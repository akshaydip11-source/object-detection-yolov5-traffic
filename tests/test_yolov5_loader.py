"""Smoke test actual local Hub loader. Random weights are NOT accuracy evidence."""

import os
import subprocess
import sys
from pathlib import Path


def test_local_checkpoint_loader(tmp_path):
    script = r"""
import sys
from pathlib import Path
import numpy as np
import torch
from backend.app.config import settings
sys.path.insert(0, str(settings.yolov5_dir))
from models.yolo import DetectionModel
from backend.app.services.detector import YOLODetector

torch.set_num_threads(2)
model = DetectionModel(str(settings.yolov5_dir / "models/yolov5n.yaml"), nc=3)
model.names = {0: "Helmet", 1: "NoHelmet", 2: "LicensePlate"}
settings.model_path = Path(sys.argv[1])
torch.save({"model": model}, settings.model_path)
detector = YOLODetector()
assert detector.loaded
assert detector.names == model.names
results, ms = detector.detect_image(np.zeros((80, 120, 3), dtype=np.uint8))
assert isinstance(results, list) and ms >= 0

# Full authenticated ASGI path with the actual loaded random network (no stub predictions).
import cv2
from uuid import uuid4
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services import detector as runtime
from backend.app.services.auth import create_access_token
from backend.app.db.database import SessionLocal
from backend.app.db.models import User
runtime._detector = detector
with TestClient(app) as client:
    email = f"{uuid4().hex}@example.com"
    with SessionLocal() as db:
        db.add(User(email=email, full_name="Runtime test", role="analyst", hashed_password="unused"))
        db.commit()
    auth = {"Authorization": f"Bearer {create_access_token({'sub': email})}"}
    ok, encoded = cv2.imencode(".jpg", np.zeros((80, 120, 3), dtype=np.uint8))
    assert ok
    response = client.post("/api/detect/image", headers=auth,
        files={"file": ("runtime-test.jpg", encoded.tobytes(), "image/jpeg")})
    assert response.status_code == 200, response.text
    assert response.json()["summary"]["model"] == "custom YOLOv5 best.pt"
    assert client.get(response.json()["result_url"]).status_code == 200
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "random-test-only.pt")],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "YOLOv5_AUTOINSTALL": "false"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
