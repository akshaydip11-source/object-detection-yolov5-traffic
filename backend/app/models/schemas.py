from datetime import datetime
from typing import Any, Optional, Literal
from pydantic import BaseModel, EmailStr, Field, field_validator


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=8, max_length=72)
    organization: str = Field("SafeCity Traffic Unit", max_length=255)

    @field_validator("password")
    @classmethod
    def password_bytes(cls, value):
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 UTF-8 bytes")
        return value


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(max_length=72)


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    organization: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


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


class BriefDetection(BaseModel):
    class_name: Literal["Helmet", "No_Helmet", "License_Plate"] = Field(alias="class")
    confidence: float
    box: list[float] = Field(min_length=4, max_length=4)


class BriefDetectResponse(BaseModel):
    job_id: str
    detections: list[BriefDetection]
    box_format: Literal["xywh"] = "xywh"
    coordinate_system: Literal["image_pixels"] = "image_pixels"
    annotated_image_url: str | None = None


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
    created_at: datetime
    bbox_json: Optional[dict] = None

    model_config = {"from_attributes": True}


class ViolationUpdate(BaseModel):
    status: Optional[Literal["open", "reviewed", "issued", "dismissed", "paid"]] = None
    notes: Optional[str] = Field(None, max_length=2000)
    fine_amount: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    plate_text: Optional[str] = Field(None, max_length=32)


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

    model_config = {"from_attributes": True}


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
    model_name: str = "Custom YOLOv5"
    model_status: Literal["available", "missing_model", "load_error"] = "missing_model"
    message: str = ""
    class_names: list[str] = Field(default_factory=list)
    max_upload_mb: int = 50
    version: str = "1.0.0"
