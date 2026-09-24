"""Isolated DB/uploads: tests never touch developer data or real checkpoints."""

import os
from pathlib import Path
import tempfile

TEMP = tempfile.TemporaryDirectory(prefix="safecity-tests-")
ROOT = Path(TEMP.name)
os.environ.update(
    DATABASE_URL=f"sqlite:///{ROOT / 'test.db'}",
    UPLOAD_DIR=str(ROOT / "uploads"),
    MODEL_PATH=str(ROOT / "missing.pt"),
    MODEL_URL="",
    MODEL_SHA256="",
    DEMO_MODE="false",
    ENVIRONMENT="local",
    REGISTRATION_ENABLED="true",
    LOGIN_REQUESTS_PER_MINUTE="100",
    DETECTION_REQUESTS_PER_MINUTE="600",
    SECRET_KEY="test-only-secret-not-for-deployment-1234567890",
)


import pytest
from fastapi.testclient import TestClient
from backend.app.api import routes
from backend.app.main import app
from backend.app.services.detector import Det, YOLODetector


@pytest.fixture
def client():
    from uuid import uuid4
    from backend.app.db.database import SessionLocal
    from backend.app.db.models import User
    from backend.app.services.auth import create_access_token

    with TestClient(app) as c:
        email = f"{uuid4().hex}@example.com"
        with SessionLocal() as db:
            db.add(
                User(
                    email=email,
                    full_name="Test Officer",
                    hashed_password="not-a-password-hash",
                    role="officer",
                )
            )
            db.commit()
        c.headers["Authorization"] = f"Bearer {create_access_token({'sub': email})}"
        yield c


@pytest.fixture
def detector(monkeypatch):
    # Exercise real drawing, image/video IO and summaries, only predictions are stubbed.
    det = YOLODetector.__new__(YOLODetector)
    det.model = object()
    det.names = {0: "Helmet", 1: "NoHelmet", 2: "LicensePlate"}
    det.model_path = Path("test-only.pt")
    monkeypatch.setattr(
        det,
        "detect_image",
        lambda *a, **k: (
            [
                Det(1, "NoHelmet", 0.9, 10, 10, 30, 30),
                Det(2, "LicensePlate", 0.8, 40, 40, 60, 50),
            ],
            1.0,
        ),
    )
    monkeypatch.setattr(routes, "get_detector", lambda: det)
    return det
