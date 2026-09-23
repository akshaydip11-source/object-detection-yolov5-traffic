from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "SafeCityAI"
    app_tagline: str = "AI-Powered Traffic Rule Enforcement"
    secret_key: str = "safecity-ai-dev-secret-change-in-production-2026"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7
    database_url: str = f"sqlite:///{BASE_DIR / 'safecity.db'}"
    upload_dir: Path = BASE_DIR / "uploads"
    weights_dir: Path = BASE_DIR / "weights"
    model_path: Path = BASE_DIR / "weights" / "yolo11n.onnx"
    coco_names_path: Path = BASE_DIR / "weights" / "coco.names"
    max_upload_mb: int = 50
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    default_admin_email: str = "admin@safecity.ai"
    default_admin_password: str = "admin123"
    cors_origins: list[str] = ["*"]


settings = Settings()
for sub in ("images", "videos", "results", "reports"):
    (settings.upload_dir / sub).mkdir(parents=True, exist_ok=True)
