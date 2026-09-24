from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from jose import jwt
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.app.config import settings, Settings
from backend.app.db.database import SessionLocal
from backend.app.db.models import User
from backend.app.services.auth import create_access_token
from backend.app.security import RequestGuards, _inference_slot
from tests.test_application import image


def account_headers(role="analyst"):
    email = f"{uuid4().hex}@example.com"
    with SessionLocal() as db:
        user = User(email=email, full_name="Test", hashed_password="unused", role=role)
        db.add(user)
        db.commit()
        uid = user.id
    return {"Authorization": f"Bearer {create_access_token({'sub': email})}"}, uid


@pytest.mark.parametrize(
    "path",
    [
        "/api/jobs",
        "/api/jobs/no-such-job",
        "/api/violations",
        "/api/dashboard/stats",
        "/api/cameras",
        "/api/users",
        "/api/export/violations.csv",
        "/api/violations/missing/pdf",
    ],
)
def test_private_apis_reject_anonymous(client, path):
    client.headers.pop("authorization")
    assert client.get(path).status_code == 401


def test_anonymous_detection_and_static_media_denied(client, detector):
    client.headers.pop("authorization")
    assert client.post("/api/detect/image", files={"file": image()}).status_code == 401
    assert client.get("/media/results/example.jpg").status_code == 404


def test_ownership_signed_media_and_restricted_roles(client, detector):
    first, first_id = account_headers()
    second, _ = account_headers()
    r = client.post("/api/detect/image", headers=first, files={"file": image()})
    assert r.status_code == 200, r.text
    data = r.json()
    url = data["result_url"]
    # Signed links are short-lived bearer capabilities, not anonymous public files.
    assert client.get(url).status_code == 200
    assert client.get(url.split("?")[0]).status_code == 422
    assert client.get(url.replace("annotated.jpg", "other.jpg")).status_code == 401
    token = url.split("token=")[1]
    assert (
        client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == 401
    )
    payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    payload["exp"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    expired = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
    assert client.get(url.split("token=")[0] + "token=" + expired).status_code == 401
    assert client.get(f"/api/jobs/{data['job_id']}", headers=second).status_code == 404
    assert data["job_id"] not in [
        j["job_id"] for j in client.get("/api/jobs", headers=second).json()
    ]
    for path in [
        "/api/violations",
        "/api/dashboard/stats",
        "/api/export/violations.csv",
    ]:
        assert client.get(path, headers=second).status_code == 403
    # Disabled accounts lose even unexpired signed-media access.
    with SessionLocal() as db:
        db.get(User, first_id).is_active = False
        db.commit()
    assert client.get(url).status_code == 401


def test_registration_disabled(client, monkeypatch):
    monkeypatch.setattr(settings, "registration_enabled", False)
    assert client.get("/api/auth/options").json()["registration_enabled"] is False
    assert (
        client.post(
            "/api/auth/register",
            json={
                "email": "new@example.com",
                "full_name": "Test",
                "password": "a-long-password",
            },
        ).status_code
        == 403
    )


def test_production_configuration():
    with pytest.raises(ValueError):
        Settings(environment="production", secret_key="short", demo_mode=False)
    with pytest.raises(ValueError):
        Settings(environment="production", secret_key="a" * 48, demo_mode=True)


def test_inference_overload(client, detector):
    _inference_slot.acquire()
    try:
        r = client.post("/api/detect/image", files={"file": image()})
        assert r.status_code == 503
        assert r.headers["retry-after"] == "5"
    finally:
        _inference_slot.release()


def test_pixel_limit(client, detector, monkeypatch):
    monkeypatch.setattr(settings, "max_image_pixels", 1000)
    assert client.post("/api/detect/image", files={"file": image()}).status_code == 400


def test_security_headers(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "no-store" in r.headers["cache-control"]


def test_request_limits(monkeypatch):
    small = FastAPI()
    small.add_middleware(RequestGuards)

    @small.post("/api/auth/login")
    async def login(request: Request):
        return {"size": len(await request.body())}

    @small.post("/api/detect/image")
    async def upload(request: Request):
        return {"size": len(await request.body())}

    monkeypatch.setattr(settings, "login_requests_per_minute", 2)
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    with TestClient(small) as c:
        assert c.post("/api/auth/login", content=b"ok").status_code == 200
        assert c.post("/api/auth/login", content=b"x" * 20000).status_code == 413
        assert c.post("/api/auth/login", content=b"ok").status_code == 429
        # No Content-Length: enforce the streaming limit too.
        body = (chunk for chunk in [b"x" * 1024**2] * 3)
        assert (
            c.post(
                "/api/detect/image",
                content=body,
                headers={
                    "Authorization": f"Bearer {create_access_token({'sub': 'test@example.com'})}"
                },
            ).status_code
            == 413
        )


def test_cleanup_dry_run_and_apply(client, tmp_path, monkeypatch):
    from backend.app.manage import cleanup
    import os
    import time

    root = tmp_path / "uploads"
    root.mkdir()
    old = root / "orphan.jpg"
    old.write_bytes(b"private bytes")
    os.utime(old, (time.time() - 40 * 86400,) * 2)
    placeholder = root / ".gitkeep"
    placeholder.touch()
    monkeypatch.setattr(settings, "upload_dir", root)
    assert cleanup(30)["applied"] is False
    assert old.exists()
    assert cleanup(30, apply=True)["applied"] is True
    assert not old.exists()
    assert placeholder.exists()


def test_bad_auth_rejected_before_large_body(client):
    # Invalid credentials should never cause a large multipart spool.
    r = client.post(
        "/api/detect/image",
        headers={
            "Authorization": "Bearer invalid",
            "Content-Length": str(100 * 1024**2),
        },
        content=b"not a body",
    )
    assert r.status_code == 401


def test_media_path_escape_blocked(client, tmp_path):
    from backend.app.services.media import media_url
    from backend.app.db.models import User

    with pytest.raises(ValueError):
        media_url(tmp_path / "outside.txt", User(id=1))


def test_production_startup_missing_model_fails(client, monkeypatch):
    from backend.app.main import app

    monkeypatch.setattr(settings, "environment", "production")
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass


def test_management_disable_account(client, monkeypatch):
    import sys
    from backend.app.manage import main

    header, user_id = account_headers()
    with SessionLocal() as db:
        email = db.get(User, user_id).email
    monkeypatch.setattr(sys, "argv", ["manage", "disable-user", "--email", email])
    main()
    assert client.get("/api/auth/me", headers=header).status_code == 401


def test_management_validation_does_not_print_password(client, monkeypatch, capsys):
    import sys
    from backend.app import manage

    sensitive_invalid_password = (
        "secret"  # Too short; should never appear in error output.
    )
    monkeypatch.setattr(manage, "getpass", lambda *a: sensitive_invalid_password)
    monkeypatch.setattr(
        sys,
        "argv",
        ["manage", "create-user", "--email", "cli@example.com", "--name", "Test"],
    )
    with pytest.raises(SystemExit):
        manage.main()
    captured = capsys.readouterr()
    assert sensitive_invalid_password not in captured.err
    assert sensitive_invalid_password not in captured.out
