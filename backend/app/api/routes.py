from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
)
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.db.models import AuditLog, CameraZone, DetectionJob, User, Violation
from app.models.schemas import (
    CameraOut, DashboardStats, DetectResponse, Detection, Token,
    UserCreate, UserLogin, UserOut, ViolationOut, ViolationUpdate, HealthOut,
)
from app.services.auth import (
    authenticate_user, create_access_token, get_current_user,
    hash_password, require_admin,
)
from app.services.detector import FINE_SCHEDULE, VIOLATION_LABELS, get_detector, new_job_id
from app.services.reports import build_violation_ticket_pdf, violations_to_csv

router = APIRouter()

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _audit(db: Session, user_id: int | None, action: str, detail: str = "") -> None:
    db.add(AuditLog(user_id=user_id, action=action, detail=detail[:2000]))
    db.commit()


def _save_upload(file: UploadFile, subdir: str) -> tuple[Path, str]:
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    name = f"{uuid.uuid4().hex}{suffix}"
    dest = settings.upload_dir / subdir / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest, file.filename or name


def _public_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        rel = p.relative_to(settings.upload_dir)
        return f"/media/{rel.as_posix()}"
    except Exception:
        return f"/media/{p.name}"


# ---------- Health ----------
@router.get("/health", response_model=HealthOut)
def health():
    try:
        det = get_detector()
        loaded = det.loaded
    except Exception:
        loaded = False
    if not loaded:
        raise HTTPException(status_code=503, detail="Detection model is unavailable")
    return HealthOut(
        status="ok",
        app=settings.app_name,
        model_loaded=loaded,
        model_path=str(settings.model_path),
    )


# ---------- Auth ----------
@router.post("/auth/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(400, "Email already registered")
    # Public registration must never grant a requested privileged role.
    # The first account on a fresh local database remains the initial admin.
    role = "admin" if db.query(User).count() == 0 else "officer"
    user = User(
        email=payload.email.lower().strip(),
        full_name=payload.full_name.strip(),
        hashed_password=hash_password(payload.password),
        role=role,
        organization=payload.organization,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    _audit(db, user.id, "register", user.email)
    return user


@router.post("/auth/login", response_model=Token)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, payload.email.lower().strip(), payload.password)
    if not user:
        raise HTTPException(401, "Invalid email or password")
    token = create_access_token({"sub": user.email, "role": user.role, "uid": user.id})
    _audit(db, user.id, "login")
    return Token(access_token=token)


@router.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


# ---------- Detection ----------
@router.post("/detect/image", response_model=DetectResponse)
async def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.35),
    camera_id: str = Form("CAM-01"),
    location: str = Form("HQ Upload Desk"),
    create_tickets: bool = Form(True),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE:
        raise HTTPException(400, f"Unsupported image type. Allowed: {sorted(ALLOWED_IMAGE)}")

    # size guard
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {settings.max_upload_mb}MB)")
    job_id = new_job_id()
    in_path = settings.upload_dir / "images" / f"{job_id}{suffix}"
    in_path.write_bytes(data)
    out_path = settings.upload_dir / "results" / f"{job_id}_annotated.jpg"

    job = DetectionJob(
        job_id=job_id,
        user_id=user.id if user else None,
        source_type="image",
        original_filename=file.filename or in_path.name,
        input_path=str(in_path),
        status="processing",
        conf_threshold=conf_threshold,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        detector = get_detector()
        result = detector.process_image_file(in_path, out_path, conf_thr=conf_threshold)
        job.status = "done"
        job.result_path = result["result_path"]
        job.object_count = result["summary"].get("object_count", 0)
        job.violation_count = result["summary"].get("violation_count", 0)
        job.processing_ms = result["processing_ms"]
        job.detections_json = {"items": result["detections"]}
        job.summary_json = result["summary"]
        job.completed_at = datetime.utcnow()
        db.commit()

        if create_tickets:
            _create_violations_from_result(
                db, job, result["violations"], camera_id, location, user
            )

        # bump camera count
        cam = db.query(CameraZone).filter(CameraZone.camera_id == camera_id).first()
        if cam:
            cam.violation_count += len(result["violations"])
            db.commit()

        _audit(db, user.id if user else None, "detect_image", job_id)

        return DetectResponse(
            job_id=job_id,
            status="done",
            source_type="image",
            object_count=job.object_count,
            violation_count=job.violation_count,
            processing_ms=job.processing_ms,
            detections=[Detection(**d) for d in result["detections"]],
            summary=result["summary"],
            result_url=_public_url(out_path),
            original_filename=job.original_filename,
            conf_threshold=conf_threshold,
            created_at=job.created_at,
        )
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise HTTPException(500, f"Detection failed: {e}") from e


@router.post("/detect/video", response_model=DetectResponse)
async def detect_video(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.35),
    camera_id: str = Form("CAM-01"),
    location: str = Form("HQ Upload Desk"),
    max_frames: int = Form(120),
    create_tickets: bool = Form(True),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO:
        raise HTTPException(400, f"Unsupported video type. Allowed: {sorted(ALLOWED_VIDEO)}")

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {settings.max_upload_mb}MB)")

    job_id = new_job_id()
    in_path = settings.upload_dir / "videos" / f"{job_id}{suffix}"
    in_path.write_bytes(data)
    out_path = settings.upload_dir / "results" / f"{job_id}_annotated.mp4"

    job = DetectionJob(
        job_id=job_id,
        user_id=user.id if user else None,
        source_type="video",
        original_filename=file.filename or in_path.name,
        input_path=str(in_path),
        status="processing",
        conf_threshold=conf_threshold,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        detector = get_detector()
        result = detector.process_video_file(
            in_path, out_path, conf_thr=conf_threshold, max_frames=min(max_frames, 300), skip=2
        )
        job.status = "done"
        job.result_path = result["result_path"]
        job.object_count = result["summary"].get("total_raw_detections", 0)
        job.violation_count = result["summary"].get("unique_violation_flags", 0)
        job.processing_ms = result["processing_ms"]
        job.detections_json = {"items": result["detections"]}
        job.summary_json = result["summary"]
        job.completed_at = datetime.utcnow()
        db.commit()

        if create_tickets:
            _create_violations_from_result(
                db, job, result["violations"], camera_id, location, user
            )
        cam = db.query(CameraZone).filter(CameraZone.camera_id == camera_id).first()
        if cam:
            cam.violation_count += len(result["violations"])
            db.commit()

        _audit(db, user.id if user else None, "detect_video", job_id)

        # Prefer snapshot for result_url preview; video still at result path
        snap = result.get("snapshot_path")
        return DetectResponse(
            job_id=job_id,
            status="done",
            source_type="video",
            object_count=job.object_count,
            violation_count=job.violation_count,
            processing_ms=job.processing_ms,
            detections=[Detection(**d) for d in result["detections"][:100]],
            summary={**result["summary"], "video_url": _public_url(out_path)},
            result_url=_public_url(snap or out_path),
            original_filename=job.original_filename,
            conf_threshold=conf_threshold,
            created_at=job.created_at,
        )
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise HTTPException(500, f"Video detection failed: {e}") from e


@router.post("/detect/frame", response_model=DetectResponse)
async def detect_frame(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.35),
    camera_id: str = Form("CAM-LIVE"),
    location: str = Form("Live Webcam"),
    create_tickets: bool = Form(False),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Single webcam/browser frame (jpeg/png) — optimized for live UI."""
    return await detect_image(
        file=file,
        conf_threshold=conf_threshold,
        camera_id=camera_id,
        location=location,
        create_tickets=create_tickets,
        user=user,
        db=db,
    )


def _create_violations_from_result(
    db: Session,
    job: DetectionJob,
    violations: list[dict],
    camera_id: str,
    location: str,
    user: User | None,
) -> None:
    for v in violations:
        ticket = f"SC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        row = Violation(
            ticket_id=ticket,
            job_id=job.id,
            reporter_id=user.id if user else None,
            violation_type=v["violation_type"],
            severity=v.get("severity", "medium"),
            confidence=float(v.get("confidence", 0)),
            location=location,
            camera_id=camera_id,
            vehicle_class=v.get("vehicle_class", "unknown"),
            plate_text=None,
            bbox_json=v.get("bbox"),
            snapshot_path=job.result_path,
            status="open",
            fine_amount=float(v.get("fine_amount", FINE_SCHEDULE.get(v["violation_type"], 500))),
            notes=VIOLATION_LABELS.get(v["violation_type"], ""),
        )
        db.add(row)
    db.commit()


@router.get("/jobs")
def list_jobs(
    limit: int = Query(30, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(DetectionJob).order_by(DetectionJob.created_at.desc()).limit(limit)
    rows = q.all()
    return [
        {
            "job_id": j.job_id,
            "source_type": j.source_type,
            "original_filename": j.original_filename,
            "status": j.status,
            "object_count": j.object_count,
            "violation_count": j.violation_count,
            "processing_ms": j.processing_ms,
            "result_url": _public_url(j.result_path),
            "summary": j.summary_json,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "conf_threshold": j.conf_threshold,
        }
        for j in rows
    ]


@router.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    j = db.query(DetectionJob).filter(DetectionJob.job_id == job_id).first()
    if not j:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": j.job_id,
        "source_type": j.source_type,
        "original_filename": j.original_filename,
        "status": j.status,
        "object_count": j.object_count,
        "violation_count": j.violation_count,
        "processing_ms": j.processing_ms,
        "detections": (j.detections_json or {}).get("items", []),
        "summary": j.summary_json,
        "result_url": _public_url(j.result_path),
        "error_message": j.error_message,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        "conf_threshold": j.conf_threshold,
    }


# ---------- Violations ----------
@router.get("/violations", response_model=list[ViolationOut])
def list_violations(
    status_filter: Optional[str] = Query(None, alias="status"),
    violation_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(Violation).order_by(Violation.created_at.desc())
    if status_filter:
        q = q.filter(Violation.status == status_filter)
    if violation_type:
        q = q.filter(Violation.violation_type == violation_type)
    if severity:
        q = q.filter(Violation.severity == severity)
    rows = q.offset(offset).limit(limit).all()
    return rows


@router.get("/violations/{ticket_id}", response_model=ViolationOut)
def get_violation(ticket_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    v = db.query(Violation).filter(Violation.ticket_id == ticket_id).first()
    if not v:
        raise HTTPException(404, "Violation not found")
    return v


@router.patch("/violations/{ticket_id}", response_model=ViolationOut)
def update_violation(
    ticket_id: str,
    payload: ViolationUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    v = db.query(Violation).filter(Violation.ticket_id == ticket_id).first()
    if not v:
        raise HTTPException(404, "Violation not found")
    data = payload.model_dump(exclude_unset=True)
    for k, val in data.items():
        setattr(v, k, val)
    v.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(v)
    _audit(db, user.id, "update_violation", f"{ticket_id} {data}")
    return v


@router.get("/violations/{ticket_id}/pdf")
def violation_pdf(
    ticket_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    v = db.query(Violation).filter(Violation.ticket_id == ticket_id).first()
    if not v:
        raise HTTPException(404, "Violation not found")
    payload = {
        "ticket_id": v.ticket_id,
        "violation_type": v.violation_type,
        "label": VIOLATION_LABELS.get(v.violation_type, v.violation_type),
        "severity": v.severity,
        "confidence": v.confidence,
        "vehicle_class": v.vehicle_class,
        "plate_text": v.plate_text,
        "camera_id": v.camera_id,
        "location": v.location,
        "status": v.status,
        "fine_amount": v.fine_amount,
        "notes": v.notes,
        "created_at": v.created_at.isoformat() if v.created_at else "",
    }
    pdf = build_violation_ticket_pdf(payload)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{ticket_id}.pdf"'},
    )


@router.get("/export/violations.csv")
def export_csv(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    rows = db.query(Violation).order_by(Violation.created_at.desc()).limit(2000).all()
    data = [
        {
            "ticket_id": r.ticket_id,
            "violation_type": r.violation_type,
            "severity": r.severity,
            "confidence": r.confidence,
            "location": r.location,
            "camera_id": r.camera_id,
            "vehicle_class": r.vehicle_class,
            "plate_text": r.plate_text,
            "status": r.status,
            "fine_amount": r.fine_amount,
            "created_at": r.created_at,
            "notes": r.notes,
        }
        for r in rows
    ]
    csv_text = violations_to_csv(data)
    _audit(db, user.id, "export_csv", f"{len(data)} rows")
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="safecity_violations.csv"'},
    )


# ---------- Cameras ----------
@router.get("/cameras", response_model=list[CameraOut])
def list_cameras(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(CameraZone).order_by(CameraZone.camera_id).all()


@router.get("/meta/violation-types")
def violation_types():
    return {
        "types": [
            {"id": k, "label": VIOLATION_LABELS[k], "fine": FINE_SCHEDULE.get(k, 0)}
            for k in VIOLATION_LABELS
        ],
        "fines": FINE_SCHEDULE,
    }


# ---------- Dashboard ----------
@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    total_det = db.query(func.count(DetectionJob.id)).scalar() or 0
    total_viol = db.query(func.count(Violation.id)).scalar() or 0
    open_viol = db.query(func.count(Violation.id)).filter(Violation.status == "open").scalar() or 0
    issued = (
        db.query(func.count(Violation.id))
        .filter(Violation.status.in_(["issued", "paid"]))
        .scalar()
        or 0
    )
    cams_total = db.query(func.count(CameraZone.id)).scalar() or 0
    cams_online = (
        db.query(func.count(CameraZone.id)).filter(CameraZone.status == "online").scalar() or 0
    )
    avg_ms = db.query(func.avg(DetectionJob.processing_ms)).filter(DetectionJob.status == "done").scalar()
    avg_ms = float(avg_ms or 0)

    # by type
    type_rows = (
        db.query(Violation.violation_type, func.count(Violation.id))
        .group_by(Violation.violation_type)
        .all()
    )
    by_type = {t: c for t, c in type_rows}

    sev_rows = (
        db.query(Violation.severity, func.count(Violation.id))
        .group_by(Violation.severity)
        .all()
    )
    by_sev = {s: c for s, c in sev_rows}

    recent = (
        db.query(Violation)
        .order_by(Violation.created_at.desc())
        .limit(8)
        .all()
    )
    recent_activity = [
        {
            "ticket_id": r.ticket_id,
            "type": r.violation_type,
            "severity": r.severity,
            "status": r.status,
            "camera_id": r.camera_id,
            "fine": r.fine_amount,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in recent
    ]

    # daily trend last 7 days
    daily_trend = []
    for i in range(6, -1, -1):
        day = datetime.utcnow().date() - timedelta(days=i)
        start = datetime.combine(day, datetime.min.time())
        end = start + timedelta(days=1)
        c = (
            db.query(func.count(Violation.id))
            .filter(Violation.created_at >= start, Violation.created_at < end)
            .scalar()
            or 0
        )
        d = (
            db.query(func.count(DetectionJob.id))
            .filter(DetectionJob.created_at >= start, DetectionJob.created_at < end)
            .scalar()
            or 0
        )
        daily_trend.append({"date": day.isoformat(), "violations": c, "detections": d})

    top_cams = (
        db.query(CameraZone)
        .order_by(CameraZone.violation_count.desc())
        .limit(5)
        .all()
    )
    top_cameras = [
        {
            "camera_id": c.camera_id,
            "name": c.name,
            "location": c.location,
            "violations": c.violation_count,
            "status": c.status,
        }
        for c in top_cams
    ]

    return DashboardStats(
        total_detections=total_det,
        total_violations=total_viol,
        open_violations=open_viol,
        issued_tickets=issued,
        cameras_online=cams_online,
        cameras_total=cams_total,
        avg_processing_ms=round(avg_ms, 1),
        violations_by_type=by_type,
        violations_by_severity=by_sev,
        recent_activity=recent_activity,
        daily_trend=daily_trend,
        top_cameras=top_cameras,
    )


@router.get("/users", response_model=list[UserOut])
def list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    return db.query(User).order_by(User.created_at.desc()).all()
