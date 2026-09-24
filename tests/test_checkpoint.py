import hashlib
import io
from unittest.mock import Mock

import pytest

from backend.app.services import checkpoint


def mock_download(monkeypatch, content):
    response = io.BytesIO(content)
    response.geturl = lambda: "https://example.test/artifact"
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(checkpoint, "build_opener", lambda *args: opener)
    return opener


def test_verified_download_then_no_network(tmp_path, monkeypatch):
    content = b"fixture bytes, not a real checkpoint"
    digest = hashlib.sha256(content).hexdigest()
    opener = mock_download(monkeypatch, content)
    target = tmp_path / "models/best.pt"
    checkpoint.provision_checkpoint(target, "https://example.test/artifact", digest)
    assert target.read_bytes() == content
    checkpoint.provision_checkpoint(target, "https://example.test/artifact", digest)
    assert opener.open.call_count == 1
    assert not list(target.parent.glob("*.partial"))


@pytest.mark.parametrize("mode", ["checksum", "oversized", "empty"])
def test_bad_download_preserves_existing(tmp_path, monkeypatch, mode):
    content = b"" if mode == "empty" else b"new content"
    mock_download(monkeypatch, content)
    target = tmp_path / "best.pt"
    target.write_bytes(b"existing trusted weights")
    digest = "0" * 64 if mode == "checksum" else hashlib.sha256(content).hexdigest()
    with pytest.raises(RuntimeError) as error:
        checkpoint.provision_checkpoint(
            target,
            "https://example.test/file?secret=do-not-log",
            digest,
            1 if mode == "oversized" else 100,
        )
    assert "do-not-log" not in str(error.value)
    assert target.read_bytes() == b"existing trusted weights"
    assert not list(tmp_path.glob("*.partial"))


def test_config_requires_https_and_hash(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        checkpoint.provision_checkpoint(
            tmp_path / "best.pt", "http://example.test/file", "0" * 64
        )
    with pytest.raises(ValueError, match="MODEL_SHA256"):
        checkpoint.provision_checkpoint(
            tmp_path / "best.pt", "https://example.test/file", None
        )


def test_no_automatic_download(tmp_path, monkeypatch):
    opener = mock_download(monkeypatch, b"unused")
    target = tmp_path / "best.pt"
    checkpoint.provision_checkpoint(target, None, None)
    assert not target.exists()
    opener.open.assert_not_called()


def test_integrity_check_before_deserialization(tmp_path, monkeypatch):
    from backend.app.config import settings
    from backend.app.services.detector import YOLODetector

    path = tmp_path / "best.pt"
    path.write_bytes(b"not a trusted pickle")
    monkeypatch.setattr(settings, "model_path", path)
    monkeypatch.setattr(settings, "model_sha256", "0" * 64)
    with pytest.raises(ValueError, match="integrity"):
        YOLODetector()
