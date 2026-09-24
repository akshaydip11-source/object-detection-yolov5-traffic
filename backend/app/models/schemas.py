from pydantic import BaseModel, Field
from typing import Any


class HealthOut(BaseModel):
    status: str
    app: str
    model_loaded: bool
    model_path: str
    version: str = "1.0.0"
    load_error: str | None = None


class DetectionOut(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: dict[str, float]
    is_violation: bool
    violation_type: str | None = None


class ImageProcessOut(BaseModel):
    job_id: str
    status: str
    source_type: str
    object_count: int
    violation_count: int
    processing_ms: float
    detections: list[DetectionOut]
    violations: list[dict[str, Any]] = []
    summary: dict[str, Any]
    result_url: str | None = None
    original_filename: str | None = None
    conf_threshold: float | None = None
    created_at: str | None = None


class VideoProcessOut(BaseModel):
    job_id: str
    status: str
    source_type: str
    total_frames: int
    processed_frames: int
    total_raw_detections: int
    unique_violation_flags: list[str]
    detections: list[DetectionOut]
    violations: list[dict[str, Any]] = []
    processing_ms: float
    video_url: str | None = None
    result_url: str | None = None
    summary: dict[str, Any]


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: str

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class CameraOut(BaseModel):
    id: int
    name: str
    zone: str
    location: str
    status: str
    latitude: float | None = None
    longitude: float | None = None

    class Config:
        from_attributes = True


class ViolationOut(BaseModel):
    id: int
    violation_type: str
    label: str
    severity: str
    fine: float
    confidence: float
    status: str
    camera_id: int | None = None
    created_at: str | None = None

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    total_cameras: int
    active_cameras: int
    total_violations: int
    pending_violations: int
    total_fines: float
    today_violations: int


class CameraCreate(BaseModel):
    name: str
    zone: str
    location: str
    latitude: float | None = None
    longitude: float | None = None


class CameraUpdate(BaseModel):
    name: str | None = None
    zone: str | None = None
    location: str | None = None
    status: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class ViolationUpdate(BaseModel):
    status: str


class MessageResponse(BaseModel):
    message: str