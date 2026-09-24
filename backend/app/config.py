"""Repository-relative configuration; environment variables override .env."""

import secrets
from pathlib import Path

from pydantic import Field, model_validator, SecretStr
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")
    app_name: str = "SafeCityAI"
    app_tagline: str = "Custom YOLOv5 Traffic Detection — Internship Demo"
    # Stable SECRET_KEY is required across restarts/multiple workers.
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(48))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    database_url: str = f"sqlite:///{BASE_DIR / 'safecity.db'}"
    upload_dir: Path = BASE_DIR / "uploads"
    model_path: Path = ROOT_DIR / "models" / "best.pt"
    model_url: SecretStr | None = None
    model_sha256: str | None = None
    yolov5_dir: Path = ROOT_DIR / "yolov5"
    device: str = "cpu"
    max_upload_mb: int = Field(50, ge=1, le=500)
    conf_threshold: float = Field(0.35, ge=0, le=1)
    iou_threshold: float = Field(0.45, ge=0, le=1)
    environment: Literal["local", "production"] = "local"
    demo_mode: bool = False
    registration_enabled: bool = False
    media_token_minutes: int = Field(10, ge=1, le=60)
    max_image_pixels: int = Field(12_000_000, ge=1024, le=50_000_000)
    max_video_seconds: int = Field(120, ge=1, le=600)
    login_requests_per_minute: int = Field(10, ge=1, le=100)
    detection_requests_per_minute: int = Field(60, ge=1, le=600)
    cors_origins: list[str] = []  # Same-origin frontend needs no CORS allowance.

    @model_validator(mode="after")
    def production_configuration(self):
        if self.environment == "production":
            if (
                "secret_key" not in self.model_fields_set
                or len(self.secret_key) < 32
                or self.secret_key.startswith("replace-")
            ):
                raise ValueError(
                    "Production requires an explicit random SECRET_KEY of at least 32 characters"
                )
            if self.demo_mode:
                raise ValueError("DEMO_MODE must be false in production")
        return self


settings = Settings()
for key in ("upload_dir", "model_path", "yolov5_dir"):
    value = getattr(settings, key)
    if not value.is_absolute():
        value = ROOT_DIR / value
    setattr(settings, key, value.resolve())
for sub in ("images", "videos", "results", "reports"):
    (settings.upload_dir / sub).mkdir(parents=True, exist_ok=True)
