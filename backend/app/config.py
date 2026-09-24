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

    # YOLOv5 custom traffic model, exported to ONNX (models/best.onnx).
    # Export once with:  python scripts/export_onnx.py --weights models/best.pt
    # Override with the MODEL_PATH env var (e.g. /app/models/best.onnx).
    model_path: Path = ROOT_DIR / "models" / "best.onnx"

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

    # Inference tuning (free Render instances have ~0.1 CPU)
    # MODEL_IMGSZ: 640 = most accurate, 512 default, 448 ~2x faster, 320 ~4x faster
    model_imgsz: int = 512
    # ORT_THREADS: keep at 1 on a CPU-capped instance (more threads = contention)
    ort_threads: int = 1
    # VIDEO_MAX_FRAMES: frames actually analysed per video (sampled across the clip)
    video_max_frames: int = 32
    # VIDEO_IMGSZ: resolution used for video frames (smaller = faster, less memory).
    # 0 = same as MODEL_IMGSZ.
    video_imgsz: int = 448
    # The model sometimes reports Helmet *and* NoHelmet for the same head.
    # When two boxes of different classes overlap heavily only the more
    # confident one is kept, so no bogus "no helmet" ticket is created.
    resolve_class_conflicts: bool = True

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
