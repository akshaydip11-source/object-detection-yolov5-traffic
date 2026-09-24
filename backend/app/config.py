from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    app_name: str = "SafeCityAI"
    app_tagline: str = "AI-Powered Traffic Rule Enforcement"

    # Authentication
    secret_key: str = "safecity-ai-dev-secret-change-in-production-2026"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7

    # Database
    database_url: str = f"sqlite:///{BASE_DIR / 'safecity.db'}"

    # Storage
    upload_dir: Path = BASE_DIR / "uploads"

    # YOLOv5 custom traffic model
    # The trained model will be placed here after Colab training.
    model_path: Path = ROOT_DIR / "models" / "best.pt"

    # Required SafeCityAI classes
    class_names: list[str] = [
        "Helmet",
        "NoHelmet",
        "LicensePlate",
    ]

    # Detection settings
    max_upload_mb: int = 50
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45

    # Default development admin
    default_admin_email: str = "admin@safecity.ai"
    default_admin_password: str = "admin123"

    # API
    cors_origins: list[str] = ["*"]


settings = Settings()

# Create runtime upload directories
for sub in ("images", "videos", "results", "reports"):
    (settings.upload_dir / sub).mkdir(parents=True, exist_ok=True)

# Ensure the model directory exists
settings.model_path.parent.mkdir(parents=True, exist_ok=True)
