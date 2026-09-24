import io

import cv2
import numpy as np
import pytest

from backend.app.config import settings
from backend.app.main import app
from backend.app.services.detector import Det, YOLODetector
from backend.app.services.reports import build_violation_ticket_pdf, violations_to_csv


def image():
    ok, encoded = cv2.imencode(".jpg", np.zeros((80, 100, 3), dtype=np.uint8))
    assert ok
    return ("traffic.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/app",
        "/dashboard",
        "/violations",
        "/live",
        "/login",
        "/about",
        "/styles.css",
        "/app.js",
        "/api/docs",
        "/api/openapi.json",
    ],
)
def test_pages(client, path):
    assert client.get(path).status_code == 200


def test_entrypoint():
    from api.server import app as deploy_app

    assert deploy_app is app


def test_missing_weights(client):
    health = client.get("/api/health").json()
    assert health["model_loaded"] is False
    assert health["status"] == "degraded"
    assert client.post("/api/detect/image", files={"file": image()}).status_code == 503


@pytest.mark.parametrize("endpoint", ["image", "frame"])
def test_image_contract(client, detector, endpoint):
    r = client.post(f"/api/detect/{endpoint}", files={"file": image()})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["object_count"] == 2
    assert data["violation_count"] == 1
    assert data["detections"][0]["class_name"] == "NoHelmet"
    assert data["detections"][0]["box"]["x1"] == 10
    assert data["detections"][1]["is_violation"] is False
    assert client.get(data["result_url"]).headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/jobs/{data['job_id']}").json()["status"] == "done"


def test_validation(client, detector, monkeypatch):
    for conf in ["-1", "1.1", "nan", "inf"]:
        assert (
            client.post(
                "/api/detect/image",
                files={"file": image()},
                data={"conf_threshold": conf},
            ).status_code
            == 422
        )
    assert (
        client.post("/api/detect/image", files={"file": ("x.exe", b"bad")}).status_code
        == 400
    )
    assert (
        client.post("/api/detect/image", files={"file": ("x.jpg", b"bad")}).status_code
        == 400
    )
    assert (
        client.post("/api/detect/image", files={"file": ("x.jpg", b"")}).status_code
        == 400
    )
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    before = set((settings.upload_dir / "images").iterdir())
    assert (
        client.post(
            "/api/detect/image", files={"file": ("x.jpg", b"x" * (1024**2 + 1))}
        ).status_code
        == 413
    )
    assert set((settings.upload_dir / "images").iterdir()) == before
    assert client.get("/api/jobs?limit=-1").status_code == 422
    assert client.get("/api/violations?offset=-1").status_code == 422


def test_auth_no_self_promotion(client, detector):
    from uuid import uuid4

    email = f"{uuid4().hex}@example.com"
    r = client.post(
        "/api/auth/register",
        json={
            "email": email,
            "full_name": "Test Analyst",
            "password": "good-password",
            "role": "admin",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["role"] == "analyst"
    token = client.post(
        "/api/auth/login", json={"email": email, "password": "good-password"}
    ).json()["access_token"]
    h = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/auth/me", headers=h).status_code == 200
    assert client.get("/api/users", headers=h).status_code == 403
    assert (
        client.post(
            "/api/detect/image",
            files={"file": image()},
            data={"create_tickets": "true"},
            headers=h,
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/detect/image",
            files={"file": image()},
            headers={"Authorization": "Bearer invalid"},
        ).status_code
        == 401
    )


def test_class_names_and_flags(detector):
    assert (
        YOLODetector._validate_names({0: "license_plate", 1: "no-helmet", 2: "helmet"})[
            1
        ]
        == "NoHelmet"
    )
    for names in [
        ["person", "car", "motorcycle"],
        ["Helmet"],
        ["Helmet", "NoHelmet", "NoHelmet"],
    ]:
        with pytest.raises(ValueError):
            YOLODetector._validate_names(names)
    dets, flags = detector.analyze_violations(
        [Det(0, "Helmet", 0.9, 1, 1, 20, 20), Det(2, "LicensePlate", 0.9, 1, 1, 20, 20)]
    )
    assert flags == []
    assert all(not d.is_violation for d in dets)


def test_video(client, detector, tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 80)
    )
    assert writer.isOpened()
    for _ in range(6):
        writer.write(np.zeros((80, 100, 3), dtype=np.uint8))
    writer.release()
    r = client.post(
        "/api/detect/video",
        files={"file": ("source.avi", source.read_bytes())},
        data={"max_frames": 2},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["summary"]["frames_processed"] == 2
    assert data["summary"]["violation_observations"] == 2
    assert client.get(data["summary"]["video_url"]).status_code == 200
    assert (
        client.post(
            "/api/detect/video",
            files={"file": ("source.avi", source.read_bytes())},
            data={"create_tickets": "true"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/detect/video",
            files={"file": ("source.avi", source.read_bytes())},
            data={"max_frames": 0},
        ).status_code
        == 422
    )


def test_reports():
    assert build_violation_ticket_pdf({"ticket_id": "test"}).startswith(b"%PDF")
    csv = violations_to_csv([{"notes": '=HYPERLINK("evil")'}])
    assert "'=HYPERLINK" in csv


def test_git_exclusions():
    import subprocess

    for name in [
        "models/best.pt",
        "backend/safecity.db-wal",
        ".env",
        "dataset/images/train/private.jpg",
        "dataset/labels/train/private.txt",
        "backend/uploads/results/output.jpg",
    ]:
        assert subprocess.run(["git", "check-ignore", "-q", name]).returncode == 0, name
    for name in [".env.example", "dataset/images/train/.gitkeep", "models/README.md"]:
        assert subprocess.run(["git", "check-ignore", "-q", name]).returncode == 1, name


def test_officer_review_flow(client, detector):
    from backend.app.db.database import SessionLocal
    from backend.app.db.models import User
    from backend.app.services.auth import create_access_token, hash_password
    from uuid import uuid4

    email = f"{uuid4().hex}@example.com"
    with SessionLocal() as db:
        db.add(
            User(
                email=email,
                full_name="Test Officer",
                hashed_password=hash_password("good-password"),
                role="officer",
            )
        )
        db.commit()
    h = {"Authorization": f"Bearer {create_access_token({'sub': email})}"}
    r = client.post(
        "/api/detect/image",
        files={"file": image()},
        data={"create_tickets": "true"},
        headers=h,
    )
    assert r.status_code == 200, r.text
    rows = client.get("/api/violations").json()
    ticket = rows[0]["ticket_id"]
    assert (
        client.patch(
            f"/api/violations/{ticket}", json={"status": "reviewed"}, headers=h
        ).status_code
        == 200
    )
    assert (
        client.patch(
            f"/api/violations/{ticket}", json={"status": "<script>"}, headers=h
        ).status_code
        == 422
    )
    assert (
        client.patch(
            f"/api/violations/{ticket}", json={"fine_amount": -1}, headers=h
        ).status_code
        == 422
    )
    assert client.get(f"/api/violations/{ticket}/pdf").content.startswith(b"%PDF")
    assert client.get("/api/export/violations.csv", headers=h).status_code == 200
    assert client.get("/api/dashboard/stats").json()["total_violations"] >= 1


def test_utf8_password_limit(client):
    assert (
        client.post(
            "/api/auth/register",
            json={
                "email": "unicode@example.com",
                "full_name": "Test",
                "password": "é" * 40,
            },
        ).status_code
        == 422
    )


def test_readiness_reports_missing_model(client):
    result = client.get("/api/ready")
    assert result.status_code == 503
    assert result.json()["model_status"] == "missing_model"
    assert "Training cannot be inferred" in result.json()["message"]


def test_video_encoder_fallback_keeps_result(detector, tmp_path, monkeypatch):
    import subprocess

    raw, output = tmp_path / "raw.mp4", tmp_path / "output.mp4"
    raw.write_bytes(b"test-mpeg4-content")

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 60)

    monkeypatch.setattr(subprocess, "run", fail)
    warning = detector._encode_browser_video(raw, output)
    assert warning and "download" in warning
    assert output.read_bytes() == b"test-mpeg4-content"
    assert not raw.exists()
