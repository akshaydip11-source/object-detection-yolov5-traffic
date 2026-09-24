"""Private media served only through short-lived, path-scoped signed links."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from jose import JWTError, jwt
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db.database import get_db
from backend.app.db.models import DetectionJob, User

router = APIRouter()


def media_url(path: str | Path | None, user: User) -> str | None:
    if not path:
        return None
    rel = Path(path).resolve().relative_to(settings.upload_dir.resolve()).as_posix()
    token = jwt.encode(
        {
            "purpose": "media",
            "path": rel,
            "sub": str(user.id),
            "exp": datetime.now(timezone.utc)
            + timedelta(minutes=settings.media_token_minutes),
        },
        settings.secret_key,
        algorithm=settings.algorithm,
    )
    return f"/api/media/{quote(rel)}?token={token}"


@router.get("/media/{relative_path:path}", include_in_schema=False)
def private_media(
    relative_path: str, token: str = Query(...), db: Session = Depends(get_db)
):
    try:
        data = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        if data.get("purpose") != "media" or data.get("path") != relative_path:
            raise ValueError("Wrong token purpose/path")
        user = db.get(User, int(data["sub"]))
        if not user or not user.is_active:
            raise ValueError("Inactive user")
    except (JWTError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(401, "Invalid or expired media link") from exc
    root = settings.upload_dir.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "Media not found")
    candidates = [str(path)]
    if path.suffix == ".jpg":  # Video previews use the video's basename.
        candidates.append(str(path.with_suffix(".mp4")))
    job = (
        db.query(DetectionJob)
        .filter(
            or_(
                DetectionJob.input_path == str(path),
                DetectionJob.result_path.in_(candidates),
            )
        )
        .first()
    )
    if not job or (user.role not in {"admin", "officer"} and job.user_id != user.id):
        raise HTTPException(404, "Media not found")
    return FileResponse(
        path, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
    )
