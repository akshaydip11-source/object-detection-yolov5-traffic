"""
SafeCityAI — FastAPI application
AI-powered traffic rule enforcement platform.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from backend.app.api.routes import router
from backend.app.config import ROOT_DIR, settings
from backend.app.db.database import Base, engine
from backend.app.seed import seed
from backend.app.services.detector import get_detector
from backend.app.services.checkpoint import provision_checkpoint
from backend.app.services.media import router as media_router
from backend.app.security import RequestGuards

FRONTEND_DIR = ROOT_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    seed()
    if settings.environment == "production":
        from backend.app.db.database import SessionLocal
        from backend.app.db.models import User

        with SessionLocal() as db:
            if (
                db.query(User)
                .filter(
                    User.email.in_(
                        [
                            "admin@safecity.ai",
                            "officer@safecity.ai",
                            "analyst@safecity.ai",
                        ]
                    )
                )
                .first()
            ):
                raise RuntimeError(
                    "Remove public demo accounts before starting in production"
                )
    try:
        provision_checkpoint(
            settings.model_path,
            settings.model_url.get_secret_value() if settings.model_url else None,
            settings.model_sha256,
        )
        get_detector()
    except Exception:
        if settings.environment == "production":
            raise
        logging.getLogger(__name__).warning(
            "Custom model unavailable; detection will return 503", exc_info=True
        )
    yield


app = FastAPI(
    lifespan=lifespan,
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=800)
app.add_middleware(RequestGuards)

app.include_router(router, prefix="/api")
app.include_router(media_router, prefix="/api")

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
        return FileResponse(
            FRONTEND_DIR / "app.js", media_type="application/javascript"
        )


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
