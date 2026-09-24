from pathlib import Path
import pathlib
import os
import sys
import time
import subprocess
from dataclasses import dataclass, asdict
from typing import Any

# Windows-only compatibility fix for YOLOv5 .pt files
# trained/saved on Linux/Colab.
#
# IMPORTANT:
# This must NOT run on Linux/Render.
if os.name == "nt" and hasattr(pathlib, "WindowsPath"):
    sys.modules["pathlib._local"] = pathlib
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import torch

from app.config import settings, ROOT_DIR


CLASS_NAMES = [
    "Helmet",
    "NoHelmet",
    "LicensePlate",
]


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
            "violation_type": "no_helmet" if is_violation else None,
        }


class SafeCityDetector:
    """SafeCityAI detector using the custom YOLOv5 traffic model."""

    def __init__(self):
        self.model_path = Path(settings.model_path)
        self.model = None
        self.loaded = False
        self.load_error = None

        self._load_model()

    def _load_model(self):
        try:
            if not self.model_path.exists():
                self.load_error = (
                    f"Custom YOLOv5 model not found: {self.model_path}. "
                    "Train the model in Google Colab and place best.pt in models/."
                )
                return

            yolov5_repo = ROOT_DIR / "yolov5"

            if not yolov5_repo.exists():
                self.load_error = (
                    f"YOLOv5 repository not found: {yolov5_repo}"
                )
                return

            self.model = torch.hub.load(
                str(yolov5_repo),
                "custom",
                path=str(self.model_path),
                source="local",
            )

            self.model.conf = settings.conf_threshold
            self.model.iou = settings.iou_threshold
            self.model.classes = None

            self.loaded = True
            self.load_error = None

            print(f"YOLO model loaded: {self.model_path}")

        except Exception as exc:
            self.loaded = False
            self.model = None
            self.load_error = str(exc)

            print(f"YOLO model load failed: {self.load_error}")

    def _detect(
        self,
        image,
        conf_thr: float | None = None,
    ) -> list[Detection]:

        if not self.loaded or self.model is None:
            return []

        if conf_thr is not None:
            self.model.conf = conf_thr

        results = self.model(image)

        predictions = (
            results.xyxy[0]
            .detach()
            .cpu()
            .numpy()
        )

        detections = []

        for x1, y1, x2, y2, confidence, class_id in predictions:
            class_id = int(class_id)

            if class_id < 0 or class_id >= len(CLASS_NAMES):
                continue

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=CLASS_NAMES[class_id],
                    confidence=float(confidence),
                    bbox=[
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                )
            )

        return detections

    def _draw(
        self,
        image,
        detections: list[Detection],
    ):
        output = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            label = (
                f"{detection.class_name} "
                f"{detection.confidence:.2f}"
            )

            if detection.class_name == "NoHelmet":
                label = f"NO HELMET | {label}"

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 220, 120),
                2,
            )

            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 120),
                2,
                cv2.LINE_AA,
            )

        return output

    def _build_summary(
        self,
        detections: list[Detection],
        violations: list[dict[str, Any]],
    ) -> dict[str, Any]:

        object_counts: dict[str, int] = {}

        for detection in detections:
            object_counts[detection.class_name] = (
                object_counts.get(detection.class_name, 0) + 1
            )

        violation_counts: dict[str, int] = {}

        for violation in violations:
            violation_type = violation.get(
                "violation_type",
                "unknown",
            )

            violation_counts[violation_type] = (
                violation_counts.get(violation_type, 0) + 1
            )

        return {
            "total_detections": len(detections),
            "total_violations": len(violations),
            "object_count": len(detections),
            "violation_count": len(violations),
            "classes": CLASS_NAMES,
            "model": "Custom YOLOv5",
            "object_counts": object_counts,
            "violation_counts": violation_counts,
        }

    def _build_violations(
        self,
        detections: list[Detection],
    ) -> list[dict[str, Any]]:

        violations = []

        for detection in detections:
            if detection.class_name != "NoHelmet":
                continue

            violations.append(
                {
                    "violation_type": "no_helmet",
                    "label": VIOLATION_LABELS["no_helmet"],
                    "severity": "high",
                    "fine": FINE_SCHEDULE["no_helmet"],
                    "confidence": detection.confidence,
                    "box": {
                        "x1": detection.bbox[0],
                        "y1": detection.bbox[1],
                        "x2": detection.bbox[2],
                        "y2": detection.bbox[3],
                    },
                }
            )

        return violations

    def process_image(
        self,
        image_path: Path,
        output_path: Path,
        conf_thr: float | None = None,
    ) -> dict[str, Any]:

        start = time.perf_counter()

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        detections = self._detect(
            image,
            conf_thr=conf_thr,
        )

        violations = self._build_violations(
            detections
        )

        annotated = self._draw(
            image,
            detections,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(output_path),
            annotated,
        )

        processing_ms = (
            time.perf_counter() - start
        ) * 1000

        return {
            "detections": [
                detection.to_dict()
                for detection in detections
            ],
            "violations": violations,
            "object_counts": self._build_summary(
                detections,
                violations,
            )["object_counts"],
            "violation_counts": self._build_summary(
                detections,
                violations,
            )["violation_counts"],
            "total_detections": len(detections),
            "total_violations": len(violations),
            "result_path": str(output_path),
            "processing_ms": processing_ms,
            "summary": self._build_summary(
                detections,
                violations,
            ),
        }

    def process_video_file(
        self,
        video_path: Path,
        output_path: Path,
        conf_thr: float | None = None,
        max_frames: int | None = None,
        skip: int = 1,
    ) -> dict[str, Any]:

        start = time.perf_counter()

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise ValueError(
                f"Unable to open video: {video_path}"
            )

        frame_width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        frame_height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25.0

        frame_count = 0
        processed_frames = 0
        total_raw_detections = 0

        all_detections = []
        all_violations = []

        unique_violation_flags = set()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (frame_width, frame_height),
        )

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
                    and processed_frames >= max_frames
                ):
                    break

                detections = self._detect(
                    frame,
                    conf_thr=conf_thr,
                )

                violations = self._build_violations(
                    detections
                )

                total_raw_detections += len(
                    detections
                )

                for violation in violations:
                    unique_violation_flags.add(
                        violation["violation_type"]
                    )

                all_detections.extend(
                    [
                        detection.to_dict()
                        for detection in detections
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

        processing_ms = (
            time.perf_counter() - start
        ) * 1000

        return {
            "total_frames": frame_count,
            "processed_frames": processed_frames,
            "total_raw_detections": total_raw_detections,
            "unique_violation_flags": list(
                unique_violation_flags
            ),
            "detections": all_detections[:200],
            "violations": all_violations,
            "snapshot_path": None,
            "video_url": str(output_path),
            "result_path": str(output_path),
            "processing_ms": processing_ms,
            "summary": {
                "total_frames": frame_count,
                "processed_frames": processed_frames,
                "total_raw_detections": total_raw_detections,
                "total_violations": len(
                    all_violations
                ),
                "unique_violation_flags": list(
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

    return uuid.uuid4().hex