"""FastAPI entry point for local runs and production ASGI servers."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.main import app

__all__ = ["app"]
