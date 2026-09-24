from pathlib import Path
import pathlib
import os
import time
import subprocess
from dataclasses import dataclass
from typing import Any

# Compatibility fix for YOLOv5 .pt files created in Linux/Colab
# when loading them on Windows with Python 3.14.
if os.name == "nt" and hasattr(pathlib, "WindowsPath"):
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import torch
import imageio_ffmpeg

from app.config import settings, ROOT_DIR


CLASS_NAMES = ["Helmet", "NoHelmet", "LicensePlate"]


VIOLATION_LABELS = {
    "no_helmet": "No Helmet",
}


FINE_SCHEDULE = {
    "no_helmet": 1000,
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]

    def to_dict(self) -> dict[str, Any]:
        is_violation = self.class_name == "NoHelmet"

        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "box": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
            },
            "is_violation": is_violation,
            "violation_type": (
                "no_helmet"
                if is_violation
                else None
            ),
        }


class SafeCityDetector:

    def __init__(self):
        self.model_path = Path(
            settings.model_path
        )

        self.input_size = 640

        self.model = None

        self.load_error = None

        try:
            yolov5_repo = (
                ROOT_DIR / "yolov5"
            )

            self.model = torch.hub.load(
                str(yolov5_repo),
                "custom",
                path=str(self.model_path),
                source="local",
            )

            self.model.conf = (
                settings.conf_threshold
            )

            self.model.iou = (
                settings.iou_threshold
            )

            self.model.max_det = 100

            print(
                f"? YOLO model loaded: "
                f"{self.model_path} "
                f"(input={self.input_size})"
            )

        except Exception as exc:
            self.load_error = str(exc)

            print(
                f"? YOLO model failed to load: "
                f"{exc}"
            )

    def _detect(
        self,
        frame,
        conf_thr: float | None = None,
    ) -> list[Detection]:

        if self.model is None:
            raise RuntimeError(
                "YOLO model is not loaded: "
                f"{self.load_error}"
            )

        confidence = (
            conf_thr
            if conf_thr is not None
            else settings.conf_threshold
        )

        self.model.conf = confidence

        results = self.model(
            frame,
            size=self.input_size,
        )

        detections: list[Detection] = []

        if (
            results.xyxy is None
            or len(results.xyxy) == 0
        ):
            return detections

        for row in results.xyxy[0].tolist():

            x1, y1, x2, y2, conf, class_id = row

            class_id = int(class_id)

            if (
                class_id < 0
                or class_id >= len(CLASS_NAMES)
            ):
                continue

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=CLASS_NAMES[
                        class_id
                    ],
                    confidence=float(conf),
                    bbox=[
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                )
            )

        return detections

    def analyze_violations(
        self,
        detections: list[Detection],
    ) -> list[dict[str, Any]]:

        violations: list[dict[str, Any]] = []

        for detection in detections:

            if (
                detection.class_name
                != "NoHelmet"
            ):
                continue

            violations.append(
                {
                    "violation_type": "no_helmet",
                    "label": (
                        VIOLATION_LABELS[
                            "no_helmet"
                        ]
                    ),
                    "confidence": (
                        detection.confidence
                    ),
                    "bbox": detection.bbox,
                    "fine": (
                        FINE_SCHEDULE[
                            "no_helmet"
                        ]
                    ),
                    "severity": "HIGH",
                    "detection": (
                        detection.to_dict()
                    ),
                }
            )

        return violations

    def _draw(
        self,
        frame,
        detections: list[Detection],
    ):

        output = frame.copy()

        for detection in detections:

            x1, y1, x2, y2 = detection.bbox

            if (
                detection.class_name
                == "NoHelmet"
            ):

                label = (
                    f"No Helmet "
                    f"{detection.confidence * 100:.1f}%"
                )

                box_color = (
                    0,
                    0,
                    255,
                )

            else:

                label = (
                    f"{detection.class_name} "
                    f"{detection.confidence * 100:.1f}%"
                )

                box_color = (
                    0,
                    255,
                    0,
                )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )

            cv2.putText(
                output,
                label,
                (
                    x1,
                    max(
                        y1 - 8,
                        20,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

        return output

    def process_image_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        conf_thr: float | None = None,
    ) -> dict[str, Any]:

        started_at = time.perf_counter()

        frame = cv2.imread(
            str(input_path)
        )

        if frame is None:
            raise ValueError(
                f"Could not read image: "
                f"{input_path}"
            )

        detections = self._detect(
            frame,
            conf_thr,
        )

        violations = (
            self.analyze_violations(
                detections
            )
        )

        output_frame = self._draw(
            frame,
            detections,
        )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not cv2.imwrite(
            str(output_path),
            output_frame,
        ):
            raise ValueError(
                f"Could not write image: "
                f"{output_path}"
            )

        processing_ms = (
            time.perf_counter()
            - started_at
        ) * 1000

        object_counts: dict[
            str, int
        ] = {}

        for detection in detections:

            object_counts[
                detection.class_name
            ] = (
                object_counts.get(
                    detection.class_name,
                    0,
                )
                + 1
            )

        violation_counts: dict[
            str, int
        ] = {}

        for violation in violations:

            violation_type = (
                violation[
                    "violation_type"
                ]
            )

            violation_counts[
                violation_type
            ] = (
                violation_counts.get(
                    violation_type,
                    0,
                )
                + 1
            )

        return {
            "result_path": str(
                output_path
            ),
            "processing_ms": (
                processing_ms
            ),
            "detections": [
                detection.to_dict()
                for detection in detections
            ],
            "violations": violations,
            "object_counts": (
                object_counts
            ),
            "violation_counts": (
                violation_counts
            ),
            "total_detections": len(
                detections
            ),
            "total_violations": len(
                violations
            ),
            "summary": {
                "object_count": len(
                    detections
                ),
                "violation_count": len(
                    violations
                ),
                "class_counts": (
                    object_counts
                ),
                "violations_by_type": (
                    violation_counts
                ),
                "total_detections": len(
                    detections
                ),
                "total_violations": len(
                    violations
                ),
                "classes": CLASS_NAMES,
                "model": "Custom YOLOv5",
            },
        }

    def process_video_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        conf_thr: float | None = None,
        max_frames: int | None = None,
        skip: int = 1,
    ) -> dict[str, Any]:

        started_at = time.perf_counter()

        cap = cv2.VideoCapture(
            str(input_path)
        )

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: "
                f"{input_path}"
            )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = (
            cap.get(
                cv2.CAP_PROP_FPS
            )
            or 25.0
        )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Temporary OpenCV output.
        temp_path = (
            output_path.parent
            / (
                f"{output_path.stem}"
                "_opencv.mp4"
            )
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(temp_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()

            raise ValueError(
                "Could not create "
                f"temporary video: "
                f"{temp_path}"
            )

        total_raw_detections = 0

        unique_violation_flags = set()

        frame_count = 0

        processed_frames = 0

        all_detections = []

        all_violations = []

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                if (
                    skip > 1
                    and frame_count % skip != 0
                ):
                    writer.write(frame)
                    continue

                if (
                    max_frames is not None
                    and processed_frames
                    >= max_frames
                ):
                    break

                detections = self._detect(
                    frame,
                    conf_thr,
                )

                violations = (
                    self.analyze_violations(
                        detections
                    )
                )

                total_raw_detections += len(
                    detections
                )

                for violation in violations:

                    unique_violation_flags.add(
                        violation[
                            "violation_type"
                        ]
                    )

                all_detections.extend(
                    [
                        detection.to_dict()
                        for detection
                        in detections
                    ]
                )

                all_violations.extend(
                    violations
                )

                writer.write(
                    self._draw(
                        frame,
                        detections,
                    )
                )

                processed_frames += 1

        finally:

            cap.release()

            writer.release()

        if not temp_path.exists():
            raise ValueError(
                "Temporary video was not created"
            )

        # Convert OpenCV mp4v output to
        # browser-friendly H.264 MP4.
        ffmpeg_exe = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

        if output_path.exists():
            output_path.unlink()

        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(temp_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ]

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception:
                pass

            raise RuntimeError(
                "FFmpeg video conversion failed: "
                f"{completed.stderr[-2000:]}"
            )

        try:
            temp_path.unlink()
        except OSError:
            pass

        if not output_path.exists():
            raise ValueError(
                "Final browser-compatible "
                "video was not created"
            )

        processing_ms = (
            time.perf_counter()
            - started_at
        ) * 1000

        return {
            "result_path": str(
                output_path
            ),
            "processing_ms": (
                processing_ms
            ),
            "total_frames": frame_count,
            "processed_frames": (
                processed_frames
            ),
            "total_raw_detections": (
                total_raw_detections
            ),
            "unique_violation_flags": list(
                unique_violation_flags
            ),
            "detections": (
                all_detections[:200]
            ),
            "violations": all_violations,
            "snapshot_path": None,
            "video_url": str(
                output_path
            ),
            "summary": {
                "total_frames": (
                    frame_count
                ),
                "processed_frames": (
                    processed_frames
                ),
                "total_raw_detections": (
                    total_raw_detections
                ),
                "unique_violation_flags": list(
                    unique_violation_flags
                ),
                "violation_count": len(
                    unique_violation_flags
                ),
                "model": "Custom YOLOv5",
                "classes": CLASS_NAMES,
            },
        }


_detector = None


def get_detector() -> SafeCityDetector:
    global _detector

    if _detector is None:
        _detector = SafeCityDetector()

    return _detector


def new_job_id() -> str:
    import uuid

    return uuid.uuid4().hex[:12]

