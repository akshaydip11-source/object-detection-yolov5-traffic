from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str = Field(min_length=6)
    organization: str = "SafeCity Traffic Unit"
    role: str = "officer"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    organization: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: BBox
    is_violation: bool = False
    violation_type: Optional[str] = None


class DetectResponse(BaseModel):
    job_id: str
    status: str
    source_type: str
    object_count: int
    violation_count: int
    processing_ms: float
    detections: list[Detection]
    summary: dict[str, Any]
    result_url: Optional[str] = None
    original_filename: str
    conf_threshold: float
    created_at: datetime


class ViolationOut(BaseModel):
    id: int
    ticket_id: str
    violation_type: str
    severity: str
    confidence: float
    location: str
    camera_id: str
    vehicle_class: str
    plate_text: Optional[str]
    status: str
    fine_amount: float
    notes: Optional[str]
    snapshot_path: Optional[str]
    created_at: datetime
    bbox_json: Optional[dict] = None

    class Config:
        from_attributes = True


class ViolationUpdate(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None
    fine_amount: Optional[float] = None
    plate_text: Optional[str] = None


class CameraOut(BaseModel):
    id: int
    camera_id: str
    name: str
    location: str
    city: str
    lat: float
    lng: float
    status: str
    violation_count: int

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    total_detections: int
    total_violations: int
    open_violations: int
    issued_tickets: int
    cameras_online: int
    cameras_total: int
    avg_processing_ms: float
    violations_by_type: dict[str, int]
    violations_by_severity: dict[str, int]
    recent_activity: list[dict[str, Any]]
    daily_trend: list[dict[str, Any]]
    top_cameras: list[dict[str, Any]]


class HealthOut(BaseModel):
    status: str
    app: str
    model_loaded: bool
    model_path: str
    version: str = "1.0.0"
    load_error: str | None = None
    model_name: str | None = None
    model_imgsz: int | None = None


