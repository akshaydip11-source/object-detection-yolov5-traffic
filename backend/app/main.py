"""
SafeCityAI — FastAPI application
AI-powered traffic rule enforcement platform.
"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from app.api.routes import router
from app.config import settings
from app.db.database import Base, engine
from app.seed import seed
from app.services.detector import get_detector

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(
    title=settings.app_name,
    description=settings.app_tagline,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=800)

app.include_router(router, prefix="/api")

# Uploaded / result media
settings.upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(settings.upload_dir)), name="media")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    seed()
    try:
        det = get_detector()
        print(f"YOLO model loaded: {det.model_path}")
    except Exception as e:
        print(f"⚠ Model load deferred/failed: {e}")


# ---- Frontend static SPA-ish multi-page ----
if FRONTEND_DIR.exists():
    # assets / samples folders if present
    assets = FRONTEND_DIR / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")
    samples = FRONTEND_DIR / "samples"
    if samples.exists():
        app.mount("/samples", StaticFiles(directory=str(samples)), name="samples")

    @app.get("/")
    async def landing():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/app")
    @app.get("/app.html")
    async def app_page():
        return FileResponse(FRONTEND_DIR / "app.html")

    @app.get("/dashboard")
    @app.get("/dashboard.html")
    async def dashboard_page():
        return FileResponse(FRONTEND_DIR / "dashboard.html")

    @app.get("/violations")
    @app.get("/violations.html")
    async def violations_page():
        return FileResponse(FRONTEND_DIR / "violations.html")

    @app.get("/live")
    @app.get("/live.html")
    async def live_page():
        return FileResponse(FRONTEND_DIR / "live.html")

    @app.get("/login")
    @app.get("/login.html")
    async def login_page():
        return FileResponse(FRONTEND_DIR / "login.html")

    @app.get("/about")
    @app.get("/about.html")
    async def about_page():
        return FileResponse(FRONTEND_DIR / "about.html")

    @app.get("/styles.css")
    async def styles():
        return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")

    @app.get("/app.js")
    async def app_js():
        return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


@app.get("/api")
def api_root():
    return {
        "name": settings.app_name,
        "tagline": settings.app_tagline,
        "docs": "/api/docs",
        "health": "/api/health",
        "endpoints": [
            "POST /api/auth/register",
            "POST /api/auth/login",
            "POST /api/detect/image",
            "POST /api/detect/video",
            "POST /api/detect/frame",
            "GET  /api/dashboard/stats",
            "GET  /api/violations",
            "GET  /api/cameras",
            "GET  /api/jobs",
        ],
    }
