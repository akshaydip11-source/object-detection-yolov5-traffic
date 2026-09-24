"""Custom YOLOv5 checkpoint inference. No pretrained fallback or COCO heuristics.

Only load trusted .pt files: PyTorch checkpoints may execute Python during load.
YOLOv5 AutoShape handles RGB preprocessing, scaling boxes and class-aware NMS.
"""

from __future__ import annotations

import os
import time
import uuid
import subprocess
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.app.config import settings

FINE_SCHEDULE = {"no_helmet": 1000.0}  # Illustrative only; human review required.
SEVERITY_MAP = {"no_helmet": "high"}
VIOLATION_LABELS = {"no_helmet": "No Helmet — Review Required"}


@dataclass
class Det:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    is_violation: bool = False
    violation_type: str | None = None

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box": {
                "x1": round(self.x1, 1),
                "y1": round(self.y1, 1),
                "x2": round(self.x2, 1),
                "y2": round(self.y2, 1),
            },
            "is_violation": self.is_violation,
            "violation_type": self.violation_type,
        }


class YOLODetector:
    def __init__(self) -> None:
        self.model_path = Path(settings.model_path)
        self.input_size = 640
        self._lock = Lock()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                "Custom YOLOv5 weights missing. Place your trusted checkpoint at "
                "models/best.pt or set MODEL_PATH; no fallback model is used."
            )
        if settings.model_sha256:
            from backend.app.services.checkpoint import checksum

            if checksum(self.model_path) != settings.model_sha256.lower():
                raise ValueError("Checkpoint integrity check failed (MODEL_SHA256)")
        if self.model_path.suffix.lower() != ".pt":
            raise ValueError("MODEL_PATH must be a custom YOLOv5 .pt checkpoint")
        if not (settings.yolov5_dir / "hubconf.py").is_file():
            raise FileNotFoundError("YOLOv5 source missing; check YOLOV5_DIR")
        # Never install packages implicitly at inference time.
        os.environ["YOLOv5_AUTOINSTALL"] = "false"
        import torch

        self.model = torch.hub.load(
            str(settings.yolov5_dir),
            "custom",
            path=str(self.model_path),
            source="local",
            device=settings.device,
            _verbose=False,
        )
        self.names = self._validate_names(self.model.names)
        self.model.eval()

    @staticmethod
    def _validate_names(names) -> dict[int, str]:
        aliases = {
            "helmet": "Helmet",
            "nohelmet": "NoHelmet",
            "licenseplate": "LicensePlate",
        }
        items = names.items() if isinstance(names, dict) else enumerate(names)
        result = {}
        for idx, name in items:
            normalized = "".join(c for c in str(name).lower() if c.isalnum())
            if normalized not in aliases:
                raise ValueError(
                    f"Unexpected model class {name!r}; expected Helmet, NoHelmet, LicensePlate"
                )
            result[int(idx)] = aliases[normalized]
        if (
            len(result) != 3
            or set(result.values()) != set(aliases.values())
            or set(result) != {0, 1, 2}
        ):
            raise ValueError(
                "Checkpoint must contain exactly Helmet, NoHelmet, LicensePlate"
            )
        return result

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def detect_image(
        self,
        img_bgr: np.ndarray,
        conf_thr: float | None = None,
        iou_thr: float | None = None,
    ) -> tuple[list[Det], float]:
        import torch

        conf = settings.conf_threshold if conf_thr is None else conf_thr
        iou = settings.iou_threshold if iou_thr is None else iou_thr
        if not 0 <= conf <= 1 or not 0 <= iou <= 1:
            raise ValueError("Confidence and IoU must be between 0 and 1")
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image")
        start = time.perf_counter()
        # AutoShape thresholds are mutable: serialize calls to the shared model.
        with self._lock, torch.inference_mode():
            self.model.conf, self.model.iou = conf, iou
            results = self.model(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), size=self.input_size
            )
            rows = results.xyxy[0].detach().cpu().numpy()
        dets = [
            Det(
                int(cid),
                self.names[int(cid)],
                float(score),
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            )
            for x1, y1, x2, y2, score, cid in rows
        ]
        return dets, (time.perf_counter() - start) * 1000

    def analyze_violations(self, dets: list[Det]) -> tuple[list[Det], list[dict]]:
        violations = []
        for d in dets:
            d.is_violation = d.class_name == "NoHelmet"
            d.violation_type = "no_helmet" if d.is_violation else None
            if d.is_violation:
                violations.append(
                    {
                        "violation_type": "no_helmet",
                        "label": VIOLATION_LABELS["no_helmet"],
                        "severity": "high",
                        "confidence": round(d.confidence, 4),
                        "fine_amount": FINE_SCHEDULE["no_helmet"],
                        "vehicle_class": "unknown",
                        "bbox": {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2},
                        "extra": {"requires_human_review": True},
                    }
                )
        return dets, violations

    # ---- Drawing ----
    COLOR_OK = (16, 185, 129)  # green
    COLOR_VIOL = (67, 56, 239)  # red-ish (BGR)
    COLOR_VEH = (245, 158, 11)  # amber
    COLOR_TEXT_BG = (15, 23, 42)

    def draw(
        self, img_bgr: np.ndarray, dets: list[Det], violations: list[dict] | None = None
    ) -> np.ndarray:
        out = img_bgr.copy()
        for d in dets:
            if d.is_violation:
                color = self.COLOR_VIOL
            elif d.class_name == "LicensePlate":
                color = self.COLOR_VEH
            else:
                color = self.COLOR_OK
            x1, y1, x2, y2 = map(int, [d.x1, d.y1, d.x2, d.y2])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            if d.violation_type:
                label = f"⚠ {d.violation_type} | {label}"
            self._draw_label(out, label, x1, max(0, y1 - 4), color)

        # banner
        h, w = out.shape[:2]
        banner_h = 42
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        vcount = sum(1 for d in dets if d.is_violation)
        text = f"SafeCityAI  |  objects: {len(dets)}  violations: {vcount if violations is None else len(violations)}"
        cv2.putText(
            out,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def _draw_label(
        self, img: np.ndarray, text: str, x: int, y: int, color: tuple
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.5, 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        y = max(th + 4, y)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + baseline - 2), color, -1)
        cv2.putText(
            img,
            text,
            (x + 4, y - 4),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def process_image_file(
        self,
        input_path: Path,
        result_path: Path,
        conf_thr: float | None = None,
    ) -> dict[str, Any]:
        try:
            with Image.open(input_path) as source:
                width, height = source.size
                if (
                    width * height > settings.max_image_pixels
                    or getattr(source, "n_frames", 1) != 1
                ):
                    raise ValueError("Image too large or animated")
                source.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Invalid image") from exc
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")
        dets, ms = self.detect_image(img, conf_thr=conf_thr)
        dets, violations = self.analyze_violations(dets)
        annotated = self.draw(img, dets, violations)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(result_path), annotated):
            raise OSError("Could not write annotated image")
        summary = self._summary(dets, violations, ms)
        return {
            "detections": [d.to_dict() for d in dets],
            "violations": violations,
            "summary": summary,
            "processing_ms": ms,
            "result_path": str(result_path),
        }

    def process_video_file(
        self,
        input_path: Path,
        result_path: Path,
        conf_thr: float | None = None,
        max_frames: int = 300,
        skip: int = 2,
    ) -> dict[str, Any]:
        if max_frames < 1 or skip < 1:
            raise ValueError("max_frames and skip must be positive")
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = (
            int(frame_count) if np.isfinite(frame_count) and frame_count > 0 else 0
        )
        if total_frames:
            skip = max(skip, math.ceil(total_frames / max_frames))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (
            w < 1
            or h < 1
            or w * h > settings.max_image_pixels
            or not np.isfinite(fps)
            or fps <= 0
        ):
            cap.release()
            raise ValueError("Invalid video dimensions or frame rate")
        raw_path = result_path.with_name(result_path.stem + "_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(raw_path), fourcc, max(0.01, fps / skip), (w + w % 2, h + h % 2)
        )

        if not writer.isOpened():
            cap.release()
            writer.release()
            raise ValueError("Could not create video output")

        all_dets: list[dict] = []
        all_violations: list[dict] = []
        class_counts: dict[str, int] = {}
        t0 = time.perf_counter()
        frame_i = 0
        processed = 0
        last_annotated = None

        try:
            while processed < max_frames:
                if time.perf_counter() - t0 > settings.max_video_seconds:
                    raise ValueError(
                        "Video processing time limit exceeded; use a shorter clip"
                    )
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.shape[0] * frame.shape[1] > settings.max_image_pixels:
                    raise ValueError("Video frame too large")
                if frame_i % skip != 0:
                    frame_i += 1
                    continue
                dets, _ = self.detect_image(frame, conf_thr=conf_thr)
                dets, violations = self.analyze_violations(dets)
                annotated = self.draw(frame, dets, violations)
                padded = cv2.copyMakeBorder(
                    annotated, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT
                )
                writer.write(padded)
                last_annotated = annotated
                for d in dets:
                    class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
                    all_dets.append(d.to_dict())
                for v in violations:
                    v = dict(v)
                    v["frame"] = frame_i
                    all_violations.append(v)
                processed += 1
                frame_i += 1

        finally:
            cap.release()
            writer.release()
        if processed == 0:
            raw_path.unlink(missing_ok=True)
            raise ValueError("Video contains no decodable frames")
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            raise OSError("Video encoder produced no output")
        video_warning = self._encode_browser_video(raw_path, result_path)

        # also save a snapshot jpeg of last frame
        snap_path = result_path.with_suffix(".jpg")
        if last_annotated is not None:
            if not cv2.imwrite(str(snap_path), last_annotated):
                raise OSError("Could not save video preview")

        ms = (time.perf_counter() - t0) * 1000.0
        # Preserve per-frame observations; no tracking/identity claims.
        uniq = all_violations  # Per-frame observations, not unique people or offences.
        summary = {
            "frames_processed": processed,
            "video_frames_total": total_frames or None,
            "frame_sampling_interval": skip,
            "video_codec": "mpeg4" if video_warning else "h264",
            "video_warning": video_warning,
            "total_raw_detections": len(all_dets),
            "violation_observations": len(uniq),
            "counting_note": "Per-frame observations; no tracking or unique-offender counting",
            "class_counts": class_counts,
            "processing_ms": round(ms, 1),
        }
        return {
            "detections": all_dets[:200],  # cap payload
            "violations": uniq,
            "summary": summary,
            "processing_ms": ms,
            "result_path": str(result_path),
            "snapshot_path": str(snap_path) if last_annotated is not None else None,
        }

    @staticmethod
    def _encode_browser_video(raw_path: Path, result_path: Path) -> str | None:
        """H.264/yuv420p/faststart; keep a downloadable fallback if conversion fails."""
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            subprocess.run(
                [
                    get_ffmpeg_exe(),
                    "-nostdin",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw_path),
                    "-an",
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
                    str(result_path),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            if not result_path.is_file() or result_path.stat().st_size == 0:
                raise OSError("Empty H.264 output")
            raw_path.unlink()
            return None
        except (OSError, RuntimeError, subprocess.SubprocessError):
            logging.getLogger(__name__).warning(
                "H.264 conversion unavailable; preserving MPEG-4 download",
                exc_info=True,
            )
            result_path.unlink(missing_ok=True)
            raw_path.replace(result_path)
            return "Browser video conversion unavailable. Use the annotated still or download the video for a compatible player."

    def _summary(self, dets: list[Det], violations: list[dict], ms: float) -> dict:
        class_counts: dict[str, int] = {}
        for d in dets:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        v_by_type: dict[str, int] = {}
        for v in violations:
            v_by_type[v["violation_type"]] = v_by_type.get(v["violation_type"], 0) + 1
        return {
            "object_count": len(dets),
            "violation_count": len(violations),
            "class_counts": class_counts,
            "violations_by_type": v_by_type,
            "helmets": sum(1 for d in dets if d.class_name == "Helmet"),
            "license_plates": sum(1 for d in dets if d.class_name == "LicensePlate"),
            "processing_ms": round(ms, 1),
            "model": "custom YOLOv5 best.pt",
        }


# Singleton initialization and mutable model settings are thread-safe.
_detector: YOLODetector | None = None
_detector_lock = Lock()


def get_detector() -> YOLODetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = YOLODetector()
    return _detector


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]
