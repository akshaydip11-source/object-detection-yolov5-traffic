from __future__ import annotations

import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db.database import get_db
from backend.app.db.models import AuditLog, CameraZone, DetectionJob, User, Violation
from backend.app.models.schemas import (
    BriefDetection,
    BriefDetectResponse,
    CameraOut,
    DashboardStats,
    DetectResponse,
    Detection,
    Token,
    UserCreate,
    UserLogin,
    UserOut,
    ViolationOut,
    ViolationUpdate,
    HealthOut,
)
from backend.app.services.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
    require_officer,
)
from backend.app.services.detector import (
    FINE_SCHEDULE,
    VIOLATION_LABELS,
    get_detector,
    new_job_id,
)
from backend.app.services.media import media_url
from backend.app.security import inference_slot
from backend.app.services.reports import build_violation_ticket_pdf, violations_to_csv

router = APIRouter()

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _audit(db: Session, user_id: int | None, action: str, detail: str = "") -> None:
    db.add(AuditLog(user_id=user_id, action=action, detail=detail[:2000]))
    db.commit()


def _save_upload(file: UploadFile, subdir: str, job_id: str) -> Path:
    """Bounded disk copy; never read an entire video into RAM."""
    suffix = Path(file.filename or "").suffix.lower()
    dest = settings.upload_dir / subdir / f"{job_id}{suffix}"
    size = 0
    try:
        with dest.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(
                        413, f"File too large (max {settings.max_upload_mb}MB)"
                    )
                output.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty upload")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    finally:
        file.file.close()
    return dest


def _ready_detector():
    try:
        return get_detector()
    except Exception as exc:
        logging.getLogger(__name__).exception("Custom YOLOv5 load failed")
        raise HTTPException(
            503,
            "Custom YOLOv5 model unavailable. Check models/best.pt and server logs.",
        ) from exc


# ---------- Health ----------
@router.get("/health", response_model=HealthOut)
def health():
    """Readiness details; loading a checkpoint does not certify its training/accuracy."""
    names = []
    try:
        det = get_detector()
        loaded = det.loaded
        raw_names = getattr(det, "names", {})
        names = (
            list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)
        )
        model_status = "available" if loaded else "load_error"
        message = (
            "Custom checkpoint loaded. Accuracy must be validated separately."
            if loaded
            else "Checkpoint failed to load. Check server logs."
        )
    except FileNotFoundError:
        loaded, model_status = False, "missing_model"
        message = "Custom checkpoint unavailable. Supply trusted models/best.pt (or MODEL_PATH), then restart. Training cannot be inferred from a missing file."
    except Exception:
        loaded, model_status = False, "load_error"
        message = "Checkpoint could not be loaded. Verify checkpoint integrity, YOLOv5 compatibility and class metadata; check server logs."
    return HealthOut(
        status="ok" if loaded else "degraded",
        app=settings.app_name,
        model_loaded=loaded,
        model_path=settings.model_path.name,
        model_status=model_status,
        message=message,
        class_names=names,
        max_upload_mb=settings.max_upload_mb,
    )


@router.get("/ready", response_model=HealthOut)
def readiness():
    result = health()
    return JSONResponse(
        result.model_dump(), status_code=200 if result.model_loaded else 503
    )


# ---------- Auth ----------
@router.get("/auth/options")
def auth_options():
    return {
        "registration_enabled": settings.registration_enabled,
        "demo_mode": settings.demo_mode,
    }


@router.post("/auth/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if not settings.registration_enabled:
        raise HTTPException(
            403, "Registration disabled; ask the administrator for an account"
        )
    email = payload.email.lower().strip()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "Email already registered")
    role = "analyst"  # Public registration never grants officer/admin privileges.
    user = User(
        email=email,
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
@router.post(
    "/detect/image",
    response_model=DetectResponse,
    dependencies=[Depends(inference_slot)],
)
def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.35, ge=0, le=1),
    camera_id: str = Form("CAM-01", max_length=64),
    location: str = Form("HQ Upload Desk", max_length=255),
    create_tickets: bool = Form(False),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE:
        raise HTTPException(
            400, f"Unsupported image type. Allowed: {sorted(ALLOWED_IMAGE)}"
        )

    if create_tickets and (user is None or user.role not in {"admin", "officer"}):
        raise HTTPException(
            403, "An officer or admin must sign in to create review tickets"
        )
    detector = _ready_detector()
    job_id = new_job_id()
    in_path = _save_upload(file, "images", job_id)
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
        if cam and create_tickets:
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
            result_url=media_url(out_path, user),
            original_filename=job.original_filename,
            conf_threshold=conf_threshold,
            created_at=job.created_at,
        )
    except Exception as e:
        logging.getLogger(__name__).exception("Detection job %s failed", job_id)
        db.rollback()
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise HTTPException(
            400 if isinstance(e, ValueError) else 500,
            "Invalid image"
            if isinstance(e, ValueError)
            else "Detection failed; check server logs",
        ) from e


@router.post(
    "/detect/predict",
    response_model=BriefDetectResponse,
    dependencies=[Depends(inference_slot)],
)
def predict_for_case_study(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.50, ge=0, le=1),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Brief-compatible JSON. Box means [left, top, width, height] in pixels.

    Reuse the authenticated, bounded image pipeline; never create tickets here.
    Keep /detect/image's corner-coordinate contract unchanged for the web console.
    """
    result = detect_image(
        file=file, conf_threshold=conf_threshold, camera_id="CASE-STUDY",
        location="Case-study image upload", create_tickets=False, user=user, db=db,
    )
    names = {"helmet": "Helmet", "nohelmet": "No_Helmet", "licenseplate": "License_Plate"}
    items = []
    for det in result.detections:
        key = "".join(c for c in det.class_name.lower() if c.isalnum())
        box = det.box
        items.append(BriefDetection(**{
            "class": names[key], "confidence": det.confidence,
            "box": [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1],
        }))
    return BriefDetectResponse(
        job_id=result.job_id, detections=items, annotated_image_url=result.result_url,
    )


@router.post(
    "/detect/video",
    response_model=DetectResponse,
    dependencies=[Depends(inference_slot)],
)
def detect_video(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.35, ge=0, le=1),
    camera_id: str = Form("CAM-01", max_length=64),
    location: str = Form("HQ Upload Desk", max_length=255),
    max_frames: int = Form(120, ge=1, le=300),
    create_tickets: bool = Form(False),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO:
        raise HTTPException(
            400, f"Unsupported video type. Allowed: {sorted(ALLOWED_VIDEO)}"
        )

    if create_tickets:
        raise HTTPException(
            400,
            "Video ticket creation is disabled: frame observations are not unique offences",
        )
    detector = _ready_detector()
    job_id = new_job_id()
    in_path = _save_upload(file, "videos", job_id)
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
        result = detector.process_video_file(
            in_path,
            out_path,
            conf_thr=conf_threshold,
            max_frames=min(max_frames, 300),
            skip=2,
        )
        job.status = "done"
        job.result_path = result["result_path"]
        job.object_count = result["summary"].get("total_raw_detections", 0)
        job.violation_count = result["summary"].get("violation_observations", 0)
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
        if cam and create_tickets:
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
            summary={**result["summary"], "video_url": media_url(out_path, user)},
            result_url=media_url(snap or out_path, user),
            original_filename=job.original_filename,
            conf_threshold=conf_threshold,
            created_at=job.created_at,
        )
    except Exception as e:
        logging.getLogger(__name__).exception("Detection job %s failed", job_id)
        db.rollback()
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise HTTPException(
            400 if isinstance(e, ValueError) else 500,
            "Invalid video"
            if isinstance(e, ValueError)
            else "Video detection failed; check server logs",
        ) from e


@router.post(
    "/detect/frame",
    response_model=DetectResponse,
    dependencies=[Depends(inference_slot)],
)
def detect_frame(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.35, ge=0, le=1),
    camera_id: str = Form("CAM-LIVE", max_length=64),
    location: str = Form("Live Webcam", max_length=255),
    create_tickets: bool = Form(False),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Single webcam/browser frame (jpeg/png) — optimized for live UI."""
    return detect_image(
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
        ticket = (
            f"SC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        )
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
            fine_amount=float(
                v.get("fine_amount", FINE_SCHEDULE.get(v["violation_type"], 500))
            ),
            notes=VIOLATION_LABELS.get(v["violation_type"], ""),
        )
        db.add(row)
    db.commit()


def _job_media(job: DetectionJob, user: User) -> dict:
    summary = dict(job.summary_json or {})
    result_path = job.result_path
    if job.source_type == "video" and result_path:
        summary["video_url"] = media_url(result_path, user)
        snapshot = Path(result_path).with_suffix(".jpg")
        result_path = str(snapshot) if snapshot.is_file() else None
    return {"result_url": media_url(result_path, user), "summary": summary}


@router.get("/jobs")
def list_jobs(
    limit: int = Query(30, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(DetectionJob)
    if user.role not in {"admin", "officer"}:
        q = q.filter(DetectionJob.user_id == user.id)
    q = q.order_by(DetectionJob.created_at.desc()).limit(limit)
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
            **_job_media(j, user),
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "conf_threshold": j.conf_threshold,
        }
        for j in rows
    ]


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    j = db.query(DetectionJob).filter(DetectionJob.job_id == job_id).first()
    if not j or (user.role not in {"admin", "officer"} and j.user_id != user.id):
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
        **_job_media(j, user),
        "error_message": "Processing failed; contact the administrator"
        if j.error_message
        else None,
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
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_officer),
    db: Session = Depends(get_db),
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
def get_violation(
    ticket_id: str, db: Session = Depends(get_db), user: User = Depends(require_officer)
):
    v = db.query(Violation).filter(Violation.ticket_id == ticket_id).first()
    if not v:
        raise HTTPException(404, "Violation not found")
    return v


@router.patch("/violations/{ticket_id}", response_model=ViolationOut)
def update_violation(
    ticket_id: str,
    payload: ViolationUpdate,
    user: User = Depends(require_officer),
    db: Session = Depends(get_db),
):
    v = db.query(Violation).filter(Violation.ticket_id == ticket_id).first()
    if not v:
        raise HTTPException(404, "Violation not found")
    if user.role not in {"admin", "officer"}:
        raise HTTPException(403, "Officer or admin required")
    data = payload.model_dump(exclude_unset=True, exclude_none=True)
    for k, val in data.items():
        setattr(v, k, val)
    v.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(v)
    _audit(db, user.id, "update_violation", f"{ticket_id} {data}")
    return v


@router.get("/violations/{ticket_id}/pdf")
def violation_pdf(
    ticket_id: str, db: Session = Depends(get_db), user: User = Depends(require_officer)
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
    user: User = Depends(require_officer),
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
        headers={
            "Content-Disposition": 'attachment; filename="safecity_violations.csv"'
        },
    )


# ---------- Cameras ----------
@router.get("/cameras", response_model=list[CameraOut])
def list_cameras(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(CameraZone).order_by(CameraZone.camera_id).all()


@router.get("/meta/violation-types")
def violation_types(user: User = Depends(get_current_user)):
    return {
        "types": [
            {"id": k, "label": VIOLATION_LABELS[k], "fine": FINE_SCHEDULE.get(k, 0)}
            for k in VIOLATION_LABELS
        ],
        "fines": FINE_SCHEDULE,
    }


# ---------- Dashboard ----------
@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats(
    db: Session = Depends(get_db), user: User = Depends(require_officer)
):
    total_det = db.query(func.count(DetectionJob.id)).scalar() or 0
    total_viol = db.query(func.count(Violation.id)).scalar() or 0
    open_viol = (
        db.query(func.count(Violation.id)).filter(Violation.status == "open").scalar()
        or 0
    )
    issued = (
        db.query(func.count(Violation.id))
        .filter(Violation.status.in_(["issued", "paid"]))
        .scalar()
        or 0
    )
    cams_total = db.query(func.count(CameraZone.id)).scalar() or 0
    cams_online = (
        db.query(func.count(CameraZone.id))
        .filter(CameraZone.status == "online")
        .scalar()
        or 0
    )
    avg_ms = (
        db.query(func.avg(DetectionJob.processing_ms))
        .filter(DetectionJob.status == "done")
        .scalar()
    )
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

    recent = db.query(Violation).order_by(Violation.created_at.desc()).limit(8).all()
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
        db.query(CameraZone).order_by(CameraZone.violation_count.desc()).limit(5).all()
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
