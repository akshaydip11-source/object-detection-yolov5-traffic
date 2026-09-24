from datetime import datetime
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(
        String(50), default="officer"
    )  # admin, officer, analyst
    organization: Mapped[str] = mapped_column(
        String(255), default="SafeCity Traffic Unit"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    detections = relationship("DetectionJob", back_populates="user")
    violations = relationship("Violation", back_populates="reporter")


class DetectionJob(Base):
    __tablename__ = "detection_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    source_type: Mapped[str] = mapped_column(String(20))  # image | video | webcam_frame
    original_filename: Mapped[str] = mapped_column(String(512))
    input_path: Mapped[str] = mapped_column(String(1024))
    result_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), default="pending"
    )  # pending|processing|done|failed
    conf_threshold: Mapped[float] = mapped_column(Float, default=0.35)
    object_count: Mapped[int] = mapped_column(Integer, default=0)
    violation_count: Mapped[int] = mapped_column(Integer, default=0)
    processing_ms: Mapped[float] = mapped_column(Float, default=0.0)
    detections_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    summary_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user = relationship("User", back_populates="detections")
    violations = relationship(
        "Violation", back_populates="job", cascade="all, delete-orphan"
    )


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ticket_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("detection_jobs.id"), nullable=True
    )
    reporter_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    violation_type: Mapped[str] = mapped_column(String(100), index=True)
    # Custom detector currently emits only no_helmet (human review required).
    severity: Mapped[str] = mapped_column(
        String(20), default="medium"
    )  # low|medium|high|critical
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    location: Mapped[str] = mapped_column(String(255), default="Unknown Camera Zone")
    camera_id: Mapped[str] = mapped_column(String(64), default="CAM-01")
    vehicle_class: Mapped[str] = mapped_column(String(64), default="unknown")
    plate_text: Mapped[str | None] = mapped_column(String(32), nullable=True)
    bbox_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), default="open"
    )  # open|reviewed|issued|dismissed|paid
    fine_amount: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    job = relationship("DetectionJob", back_populates="violations")
    reporter = relationship("User", back_populates="violations")


class CameraZone(Base):
    __tablename__ = "camera_zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    location: Mapped[str] = mapped_column(String(255))
    city: Mapped[str] = mapped_column(String(100), default="Kolkata")
    lat: Mapped[float] = mapped_column(Float, default=22.5726)
    lng: Mapped[float] = mapped_column(Float, default=88.3639)
    status: Mapped[str] = mapped_column(String(32), default="online")
    stream_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    violation_count: Mapped[int] = mapped_column(Integer, default=0)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(100))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
