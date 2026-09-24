"""
ONNX object detection for the SafeCityAI YOLOv5 traffic case study.

Loads the configured ONNX detector. The checked-in fallback is a COCO-pretrained
YOLO11 model; a trained YOLOv5 ONNX model can be selected with MODEL_PATH and
CLASS_NAMES_PATH. Custom No_Helmet detections are reported directly, while the
COCO fallback reports objects only. COCO does not detect helmet or seatbelt
status, so its detections must never be turned into traffic violations.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from app.config import settings


# COCO class indices we care about for traffic
PERSON = 0
BICYCLE = 1
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7
TRAFFIC_LIGHT = 9
STOP_SIGN = 11

VEHICLE_IDS = {BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK}
RIDER_VEHICLE_IDS = {BICYCLE, MOTORCYCLE}

# Fine schedule (INR) — illustrative Indian MV Act style amounts
FINE_SCHEDULE = {
    "no_helmet": 1000.0,
    "no_seatbelt": 1000.0,
    "triple_riding": 2000.0,
    "overcrowded_vehicle": 1500.0,
    "red_light_suspect": 5000.0,
    "stop_sign_suspect": 2000.0,
    "unattended_child_proxy": 500.0,
    "high_risk_cluster": 500.0,
}

SEVERITY_MAP = {
    "no_helmet": "high",
    "no_seatbelt": "medium",
    "triple_riding": "high",
    "overcrowded_vehicle": "medium",
    "red_light_suspect": "critical",
    "stop_sign_suspect": "high",
    "unattended_child_proxy": "medium",
    "high_risk_cluster": "low",
}

VIOLATION_LABELS = {
    "no_helmet": "No Helmet Detected",
    "no_seatbelt": "Seatbelt Violation (Suspected)",
    "triple_riding": "Triple Riding / Overloading",
    "overcrowded_vehicle": "Overcrowded Vehicle",
    "red_light_suspect": "Red Light Jump (Suspected)",
    "stop_sign_suspect": "Stop Sign Violation (Suspected)",
    "unattended_child_proxy": "Vulnerable Road User Alert",
    "high_risk_cluster": "High-Risk Object Cluster",
}


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
        box = {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
        }
        return {
            "class_id": self.class_id,
            "class": self.class_name,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box": box,
            "box_xywh": [
                round(self.x1, 1),
                round(self.y1, 1),
                round(self.x2 - self.x1, 1),
                round(self.y2 - self.y1, 1),
            ],
            "is_violation": self.is_violation,
            "violation_type": self.violation_type,
        }


class YOLODetector:
    def __init__(self) -> None:
        self.model_path = Path(settings.model_path)
        self.names = self._load_names()
        self.custom_classes = len(self.names) != 80
        self.session: ort.InferenceSession | None = None
        self.input_name = "images"
        self.input_size = 640
        self._load()

    def _load_names(self) -> list[str]:
        path = Path(settings.class_names_path)
        if not path.is_file():
            raise FileNotFoundError(f"Class names file not found: {path}")
        names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not names:
            raise ValueError(f"Class names file is empty: {path}")
        return names

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        providers = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=so, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # [1,3,H,W] or dynamic
        if isinstance(shape[2], int):
            self.input_size = shape[2]

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def _letterbox(
        self, img: np.ndarray, new_shape: int = 640
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        h, w = img.shape[:2]
        r = min(new_shape / h, new_shape / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = new_shape - nw, new_shape - nh
        left, top = pad_w // 2, pad_h // 2
        right, bottom = pad_w - left, pad_h - top
        out = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return out, r, (left, top)

    def _preprocess(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float], tuple[int, int]]:
        h0, w0 = img_bgr.shape[:2]
        img, r, (dw, dh) = self._letterbox(img_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]  # 1x3xHxW
        return img, r, (dw, dh), (h0, w0)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return keep

    def _postprocess(
        self,
        output: np.ndarray,
        r: float,
        pad: tuple[float, float],
        orig_hw: tuple[int, int],
        conf_thr: float,
        iou_thr: float,
    ) -> list[Det]:
        """
        Supports raw YOLOv5 ONNX output (xywh, objectness, class scores) and
        YOLOv8/11 output (xywh, class scores), in either common tensor orientation.
        """
        pred = output
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim != 2:
            raise ValueError(f"Unsupported ONNX detector output shape: {output.shape}")

        nc = len(self.names)
        output_widths = {4 + nc, 5 + nc}
        if pred.shape[1] not in output_widths and pred.shape[0] in output_widths:
            pred = pred.T
        if pred.shape[1] == 5 + nc:
            objectness = pred[:, 4]
            cls_scores = pred[:, 5:] * objectness[:, None]
        elif pred.shape[1] == 4 + nc:
            cls_scores = pred[:, 4:]
        else:
            raise ValueError(
                f"ONNX output has {pred.shape[1]} values per box, but class names define "
                f"{nc} classes (expected {4 + nc} or {5 + nc}). Check MODEL_PATH and CLASS_NAMES_PATH."
            )

        boxes_xywh = pred[:, :4]
        cls_ids = cls_scores.argmax(axis=1)
        confs = cls_scores.max(axis=1)

        mask = confs >= conf_thr
        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        if len(confs) == 0:
            return []

        # xywh (center) -> xyxy in letterbox space
        x, y, w, h = boxes_xywh.T
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # undo letterbox
        dw, dh = pad
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= max(r, 1e-6)
        oh, ow = orig_hw
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)

        # class-wise NMS
        final: list[Det] = []
        for cid in np.unique(cls_ids):
            idxs = np.where(cls_ids == cid)[0]
            keep = self._nms(boxes[idxs], confs[idxs], iou_thr)
            for k in keep:
                i = idxs[k]
                name = self.names[int(cid)] if int(cid) < len(self.names) else f"class_{cid}"
                final.append(
                    Det(
                        class_id=int(cid),
                        class_name=name,
                        confidence=float(confs[i]),
                        x1=float(boxes[i, 0]),
                        y1=float(boxes[i, 1]),
                        x2=float(boxes[i, 2]),
                        y2=float(boxes[i, 3]),
                    )
                )
        final.sort(key=lambda d: d.confidence, reverse=True)
        return final

    def detect_image(
        self,
        img_bgr: np.ndarray,
        conf_thr: float | None = None,
        iou_thr: float | None = None,
    ) -> tuple[list[Det], float]:
        conf_thr = conf_thr if conf_thr is not None else settings.conf_threshold
        iou_thr = iou_thr if iou_thr is not None else settings.iou_threshold
        t0 = time.perf_counter()
        inp, r, pad, hw = self._preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: inp})  # type: ignore
        dets = self._postprocess(outputs[0], r, pad, hw, conf_thr, iou_thr)
        ms = (time.perf_counter() - t0) * 1000.0
        return dets, ms

    @staticmethod
    def _center(d: Det) -> tuple[float, float]:
        return ((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2)

    @staticmethod
    def _area(d: Det) -> float:
        return max(0.0, d.x2 - d.x1) * max(0.0, d.y2 - d.y1)

    @staticmethod
    def _iou(a: Det, b: Det) -> float:
        xx1 = max(a.x1, b.x1)
        yy1 = max(a.y1, b.y1)
        xx2 = min(a.x2, b.x2)
        yy2 = min(a.y2, b.y2)
        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
        if inter <= 0:
            return 0.0
        return inter / (YOLODetector._area(a) + YOLODetector._area(b) - inter + 1e-6)

    @staticmethod
    def _person_near_vehicle(person: Det, vehicle: Det, expand: float = 0.35) -> bool:
        """Person overlaps vehicle, or sits in an expanded box (riders sit above bikes)."""
        vw = vehicle.x2 - vehicle.x1
        vh = vehicle.y2 - vehicle.y1
        ex1 = vehicle.x1 - vw * expand * 0.25
        ex2 = vehicle.x2 + vw * expand * 0.25
        ey1 = vehicle.y1 - vh * expand  # expand upward for rider torso/head
        ey2 = vehicle.y2 + vh * 0.1
        cx, cy = YOLODetector._center(person)
        inside = ex1 <= cx <= ex2 and ey1 <= cy <= ey2
        return inside or YOLODetector._iou(person, vehicle) > 0.03

    def analyze_violations(self, dets: list[Det]) -> tuple[list[Det], list[dict[str, Any]]]:
        """
        Report only violations supported by custom traffic classes.

        COCO's person and vehicle classes cannot establish helmet status,
        seatbelt use, or a traffic-rule violation. The fallback must not turn
        those object detections into enforcement flags or tickets.
        """
        if not self.custom_classes:
            return dets, []

        violations: list[dict[str, Any]] = []
        for det in dets:
            class_name = det.class_name.strip().lower().replace("-", "_").replace(" ", "_")
            if class_name in {"no_helmet", "nohelmet"}:
                det.is_violation = True
                det.violation_type = "no_helmet"
                violations.append(
                    self._vrec(
                        "no_helmet",
                        det.confidence,
                        det,
                        vehicle_class="motorcycle",
                        extra={"detected_class": det.class_name},
                    )
                )
        return dets, self._dedupe_violations(violations)

    def _vrec(
        self,
        vtype: str,
        conf: float,
        det: Det,
        vehicle_class: str = "unknown",
        extra: dict | None = None,
    ) -> dict[str, Any]:
        return {
            "violation_type": vtype,
            "label": VIOLATION_LABELS.get(vtype, vtype),
            "severity": SEVERITY_MAP.get(vtype, "medium"),
            "confidence": round(float(conf), 4),
            "fine_amount": FINE_SCHEDULE.get(vtype, 500.0),
            "vehicle_class": vehicle_class,
            "bbox": {"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2},
            "extra": extra or {},
        }

    @staticmethod
    def _dedupe_violations(violations: list[dict]) -> list[dict]:
        seen: set[tuple] = set()
        out = []
        for v in violations:
            b = v.get("bbox") or {}
            key = (
                v["violation_type"],
                int(b.get("x1", 0) // 10),
                int(b.get("y1", 0) // 10),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
        return out

    # ---- Drawing ----
    COLOR_OK = (16, 185, 129)       # green
    COLOR_VIOL = (67, 56, 239)      # red-ish (BGR)
    COLOR_VEH = (245, 158, 11)      # amber
    COLOR_TEXT_BG = (15, 23, 42)

    def draw(self, img_bgr: np.ndarray, dets: list[Det], violations: list[dict] | None = None) -> np.ndarray:
        out = img_bgr.copy()
        for d in dets:
            if d.is_violation:
                color = self.COLOR_VIOL
            elif d.class_id in VEHICLE_IDS:
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
        cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _draw_label(self, img: np.ndarray, text: str, x: int, y: int, color: tuple) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.5, 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        y = max(th + 4, y)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + baseline - 2), color, -1)
        cv2.putText(img, text, (x + 4, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def process_image_file(
        self,
        input_path: Path,
        result_path: Path,
        conf_thr: float | None = None,
    ) -> dict[str, Any]:
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")
        dets, ms = self.detect_image(img, conf_thr=conf_thr)
        dets, violations = self.analyze_violations(dets)
        annotated = self.draw(img, dets, violations)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(result_path), annotated):
            raise RuntimeError(f"Could not write annotated image: {result_path}")
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
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0:
            fps = 15.0
        if w <= 0 or h <= 0:
            cap.release()
            raise ValueError("Video has invalid frame dimensions")
        max_frames = int(max_frames)
        skip = max(1, int(skip))
        if max_frames <= 0:
            cap.release()
            raise ValueError("max_frames must be greater than zero")

        # Sample across the entire clip. Previously this loop stopped after the
        # first max_frames*skip source frames, silently ignoring the rest of a
        # longer upload. Keep the output duration close to the source duration.
        if total_frames > 0:
            skip = max(skip, math.ceil(total_frames / max_frames))

        # OpenCV's mp4v output is not supported by every browser. Render installs
        # FFmpeg below and converts this intermediate into H.264 for playback.
        raw_path = result_path.with_name(f"{result_path.stem}_opencv.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(raw_path), fourcc, max(0.1, fps / skip), (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not initialize the MP4 video encoder")

        all_dets: list[dict] = []
        all_violations: list[dict] = []
        class_counts: dict[str, int] = {}
        t0 = time.perf_counter()
        frame_i = 0
        processed = 0
        last_annotated = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if processed >= max_frames:
                    break
                if frame_i % skip != 0:
                    frame_i += 1
                    continue
                dets, _ = self.detect_image(frame, conf_thr=conf_thr)
                dets, violations = self.analyze_violations(dets)
                annotated = self.draw(frame, dets, violations)
                writer.write(annotated)
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

        if processed == 0 or not raw_path.exists() or raw_path.stat().st_size == 0:
            raise ValueError("No frames could be decoded from the uploaded video")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw_path),
                    "-an",
                    "-vf",
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "28",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(result_path),
                ],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Could not encode browser-playable MP4: {completed.stderr[-1000:]}")
            raw_path.unlink(missing_ok=True)
        else:
            raw_path.replace(result_path)

        # also save a snapshot jpeg of last frame
        snap_path = result_path.with_suffix(".jpg")
        if last_annotated is not None:
            if not cv2.imwrite(str(snap_path), last_annotated):
                raise RuntimeError("Could not write the final video preview frame")

        ms = (time.perf_counter() - t0) * 1000.0
        # unique-ish violations by type
        uniq = self._dedupe_violations(all_violations)
        summary = {
            "frames_processed": processed,
            "video_frames_total": total_frames or None,
            "frame_sampling_interval": skip,
            "total_raw_detections": len(all_dets),
            "unique_violation_flags": len(uniq),
            "class_counts": class_counts,
            "processing_ms": round(ms, 1),
            "snapshot": str(snap_path) if last_annotated is not None else None,
        }
        return {
            "detections": all_dets[:200],  # cap payload
            "violations": uniq,
            "summary": summary,
            "processing_ms": ms,
            "result_path": str(result_path),
            "snapshot_path": str(snap_path) if last_annotated is not None else None,
        }

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
            "vehicles": 0 if self.custom_classes else sum(1 for d in dets if d.class_id in VEHICLE_IDS),
            "persons": 0 if self.custom_classes else sum(1 for d in dets if d.class_id == PERSON),
            "processing_ms": round(ms, 1),
            "model": self.model_label,
        }

    @property
    def model_label(self) -> str:
        if self.custom_classes:
            if "yolov5" in self.model_path.stem.lower():
                return f"YOLOv5 custom ONNX · {len(self.names)} classes"
            return f"Custom ONNX · {len(self.names)} classes"
        return "YOLO11n COCO ONNX fallback"


# Singleton
_detector: YOLODetector | None = None


def get_detector() -> YOLODetector:
    global _detector
    if _detector is None:
        _detector = YOLODetector()
    return _detector


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]
