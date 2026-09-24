"""CI-only container smoke check with a temporary UNTRAINED checkpoint.

Requires a Docker daemon and a built safecity-ci image. Optional explicit
PILOT_CHECKPOINT/PILOT_IMAGE exercise a supplied artifact; otherwise weights are
untrained fixtures. Successful integration does not certify model accuracy.
"""

import io
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

import httpx
from PIL import Image


def docker(*args, **kwargs):
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True, **kwargs
    ).stdout.strip()


def main():
    with tempfile.TemporaryDirectory(prefix="safecity-container-") as directory:
        root = Path(directory)
        root.chmod(0o755)
        pilot_checkpoint = os.getenv("PILOT_CHECKPOINT")
        pilot_image = os.getenv("PILOT_IMAGE")
        if bool(pilot_checkpoint) != bool(pilot_image):
            raise ValueError("Supply both explicit pilot paths")
        checkpoint = root / ("pilot-test-only.pt" if pilot_checkpoint else "untrained-test-only.pt")
        if pilot_checkpoint:
            shutil.copyfile(Path(pilot_checkpoint).resolve(), checkpoint)
        else:
            subprocess.run(
                [sys.executable, "-m", "tests.checkpoint_factory", str(checkpoint)],
                check=True,
            )
        checkpoint.chmod(0o644)
        env = {**os.environ, "SECRET_KEY": secrets.token_urlsafe(48)}
        container = docker(
            "run",
            "-d",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--tmpfs",
            "/app/data:rw,uid=10001,gid=10001,size=128m",
            "--tmpfs",
            "/app/backend/uploads:rw,uid=10001,gid=10001,size=128m",
            "-e",
            "SECRET_KEY",
            "-e",
            "ENVIRONMENT=production",
            "-e",
            "DEMO_MODE=false",
            "-e",
            f"MODEL_PATH=/app/models/{checkpoint.name}",
            "--mount",
            f"type=bind,source={root},target=/app/models,readonly",
            "-p",
            "127.0.0.1::8000",
            "safecity-ci",
            env=env,
        )
        try:
            port = docker("port", container, "8000/tcp").rsplit(":", 1)[1]
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=30
            ) as client:
                deadline = time.monotonic() + 120
                while True:
                    try:
                        if client.get("/api/health").json().get("model_loaded"):
                            break
                    except (httpx.HTTPError, ValueError):
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Container failed readiness (check docker logs)"
                        )
                    time.sleep(1)
                assert docker("exec", container, "id", "-u") == "10001"
                assert client.get("/").status_code == 200
                assert client.get("/api/jobs").status_code == 401
                password = secrets.token_urlsafe(20)
                seed = """
import os
from backend.app.db.database import SessionLocal
from backend.app.db.models import User
from backend.app.services.auth import hash_password
with SessionLocal() as db:
    db.add(User(email="container-test@example.com", full_name="CI-only account",
                role="officer", hashed_password=hash_password(os.environ["TEST_PASSWORD"])))
    db.commit()
"""
                docker(
                    "exec",
                    "-i",
                    "-e",
                    "TEST_PASSWORD",
                    container,
                    "python",
                    "-",
                    input=seed,
                    env={**os.environ, "TEST_PASSWORD": password},
                )
                login = client.post(
                    "/api/auth/login",
                    json={"email": "container-test@example.com", "password": password},
                )
                login.raise_for_status()
                headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
                if pilot_image:
                    source = Path(pilot_image)
                    image_file = (source.name, source.read_bytes(), "image/png" if source.suffix.lower() == ".png" else "image/jpeg")
                else:
                    image = io.BytesIO()
                    Image.new("RGB", (120, 80), "black").save(image, format="JPEG")
                    image_file = ("ci.jpg", image.getvalue(), "image/jpeg")
                result = client.post(
                    "/api/detect/image",
                    headers=headers,
                    files={"file": image_file},
                )
                result.raise_for_status()
                assert client.get(result.json()["result_url"]).status_code == 200
                assert client.get("/api/jobs", headers=headers).status_code == 200
                health = json.loads(
                    docker("inspect", "--format", "{{json .State.Health}}", container)
                )
                print(
                    f"Container inference/auth/media smoke passed (health state: {health['Status']})."
                )
                print(
                    "Supplied pilot checkpoint exercised; this is not accuracy certification."
                    if pilot_checkpoint else
                    "Random weights only: trained-model compatibility and accuracy remain unverified."
                )
        finally:
            # Do not print access tokens or the environment when collecting failures.
            docker("rm", "-f", container)


if __name__ == "__main__":
    main()
