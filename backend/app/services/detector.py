"""
SafeCityAI detector — ONNX Runtime inference for the custom YOLOv5 traffic model.

Why ONNX Runtime instead of PyTorch + torch.hub:

* Render's free instance has 0.1 CPU / 512 MB RAM. PyTorch eager inference on that
  budget took ~12 s per image and made any real video job impossible.
* ONNX Runtime runs the same weights ~2-3x faster and needs a fraction of the memory,
  so the container ships without torch/ultralytics (much smaller image, faster builds).
* It also removes the whole class of checkpoint-unpickling problems
  (`pathlib._local`, `models.yolo.DetectionModel`, torch version drift) that broke
  earlier deploys.

The model file must be ONNX: export your trained YOLOv5 checkpoint once with

    python scripts/export_onnx.py --weights models/best.pt --out models/best.onnx

Classes: Helmet, NoHelmet, LicensePlate.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from app.config import settings, ROOT_DIR

logger = logging.getLogger(__name__)

PT_SUFFIXES = {".pt", ".pth", ".ckpt"}

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

SEVERITY = {
    "no_helmet": "high",
}

# box colours per class (BGR)
CLASS_COLORS = {
    0: (120, 220, 60),    # Helmet      -> green
    1: (80, 80, 240),     # NoHelmet    -> red
    2: (240, 180, 60),    # LicensePlate-> blue/amber
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]
    frame: int | None = None

    def to_dict(self) -> dict[str, Any]:
        is_violation = self.class_name == "NoHelmet"

        data = {
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
            "violation_type": ("no_helmet" if is_violation else None),
        }

        if self.frame is not None:
            data["frame"] = self.frame

        return data


class SafeCityDetector:
    """SafeCityAI detector backed by the custom YOLOv5 traffic model (ONNX)."""

    def __init__(self):
        self.model_path = Path(settings.model_path)
        self.imgsz = int(settings.model_imgsz)
        self.input_name = "images"

        self.session: ort.InferenceSession | None = None
        self.loaded = False
        self.load_error: str | None = None

        # number of values per anchor row: 4 box + 1 objectness + N classes
        self.row_width = 5 + len(CLASS_NAMES)

        self._load_model()

    # ------------------------------------------------------------------ load
    def _load_model(self) -> None:
        try:
            if self.model_path.suffix.lower() in PT_SUFFIXES:
                raise ValueError(
                    f"{self.model_path.name} is a PyTorch checkpoint. This service runs "
                    "ONNX Runtime — export it once and commit the .onnx file:\n"
                    "    python scripts/export_onnx.py --weights "
                    f"{self.model_path} --out models/best.onnx"
                )

            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Detection model not found: {self.model_path}. "
                    "Commit models/best.onnx (the Dockerfile copies the models/ folder)."
                )

            if self.model_path.stat().st_size < 1024 * 1024:
                raise ValueError(
                    f"{self.model_path} is only "
                    f"{self.model_path.stat().st_size} bytes — that is not a real model "
                    "(Git LFS pointer?)."
                )

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Render's free instance is CPU-capped; extra threads only add contention.
            opts.intra_op_num_threads = max(1, int(settings.ort_threads))
            opts.inter_op_num_threads = 1
            # Dynamic-shape models grow the memory arena run after run. On a
            # 512 MB instance that is what tips the process into an OOM kill, so
            # trade a little speed for a flat memory profile.
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.log_severity_level = 3

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            model_input = self.session.get_inputs()[0]
            self.input_name = model_input.name

            # If the graph has a fixed H/W, honour it instead of the configured size.
            shape = model_input.shape  # [N, 3, H, W]
            if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
                self.imgsz = int(shape[2])

            self.loaded = True
            self.load_error = None
            logger.info(
                "Detection model loaded: %s (imgsz=%d, classes=%s)",
                self.model_path,
                self.imgsz,
                CLASS_NAMES,
            )
            print(
                f"YOLO model loaded: {self.model_path} "
                f"(onnxruntime, imgsz={self.imgsz})"
            )

        except Exception as exc:  # noqa: BLE001 - reported through /api/health
            self.loaded = False
            self.session = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("YOLO model load failed: %s", self.load_error)
            print(f"YOLO model load failed: {self.load_error}")

    @property
    def model_label(self) -> str:
        if not self.loaded:
            return "model unavailable"
        return f"Custom YOLOv5 ONNX · {len(CLASS_NAMES)} classes · {self.imgsz}px"

    # --------------------------------------------------- pre / post process
    def _letterbox(
        self, img: np.ndarray, new_shape: int | None = None
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize + pad keeping aspect ratio (YOLOv5 letterbox)."""
        new_shape = int(new_shape or self.imgsz)
        h, w = img.shape[:2]
        r = min(new_shape / h, new_shape / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = new_shape - nw, new_shape - nh
        left, top = pad_w // 2, pad_h // 2
        out = cv2.copyMakeBorder(
            resized,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return out, r, (left, top)

    def _preprocess(
        self, img_bgr: np.ndarray, imgsz: int | None = None
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        boxed, r, pad = self._letterbox(img_bgr, imgsz)
        rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(x), r, pad

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
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
            order = order[np.where(iou <= iou_thr)[0] + 1]
        return keep

    def _postprocess(
        self,
        output: np.ndarray,
        r: float,
        pad: tuple[int, int],
        orig_hw: tuple[int, int],
        conf_thr: float,
        iou_thr: float,
        max_det: int = 100,
    ) -> list[Detection]:
        """Decode YOLOv5 ONNX output: (N, 4+1+nc) rows of xywh + obj + class scores."""
        pred = output
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim != 2:
            raise ValueError(f"Unsupported model output shape: {output.shape}")

        if pred.shape[1] != self.row_width and pred.shape[0] == self.row_width:
            pred = pred.T
        if pred.shape[1] != self.row_width:
            raise ValueError(
                f"Model output has {pred.shape[1]} values per box but "
                f"{len(CLASS_NAMES)} classes were configured "
                f"({self.row_width} expected). Check that the ONNX model matches "
                f"the class list."
            )

        objectness = pred[:, 4]
        cls_scores = pred[:, 5:] * objectness[:, None]
        cls_ids = cls_scores.argmax(axis=1)
        confs = cls_scores.max(axis=1)

        mask = confs >= conf_thr
        boxes_xywh = pred[mask, :4]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        if len(confs) == 0:
            return []

        x, y, w, h = boxes_xywh.T
        boxes = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)

        # undo letterbox
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes /= max(r, 1e-6)
        oh, ow = orig_hw
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)

        detections: list[Detection] = []
        for cid in np.unique(cls_ids):
            idxs = np.where(cls_ids == cid)[0]
            for k in self._nms(boxes[idxs], confs[idxs], iou_thr)[:max_det]:
                i = idxs[k]
                detections.append(
                    Detection(
                        class_id=int(cid),
                        class_name=(
                            CLASS_NAMES[int(cid)]
                            if int(cid) < len(CLASS_NAMES)
                            else f"class_{cid}"
                        ),
                        confidence=float(confs[i]),
                        bbox=[int(v) for v in boxes[i]],
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)

        if bool(getattr(settings, "resolve_class_conflicts", True)):
            detections = self._resolve_conflicts(detections)

        return detections[:max_det]

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / float(area_a + area_b - inter + 1e-6)

    def _resolve_conflicts(self, detections: list[Detection]) -> list[Detection]:
        """Drop contradictory labels on the same object (e.g. Helmet + NoHelmet).

        The checkpoint occasionally fires both classes on one head; keeping only
        the most confident box prevents false "no helmet" violations.
        """
        kept: list[Detection] = []
        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            if any(
                det.class_id != other.class_id
                and self._iou(det.bbox, other.bbox) >= 0.5
                for other in kept
            ):
                continue
            kept.append(det)
        return kept

    # -------------------------------------------------------------- inference
    def _detect(
        self,
        image: np.ndarray,
        conf_thr: float | None = None,
        imgsz: int | None = None,
    ) -> list[Detection]:
        if not self.loaded or self.session is None:
            return []

        conf_thr = (
            float(conf_thr)
            if conf_thr is not None
            else float(settings.conf_threshold)
        )

        inp, r, pad = self._preprocess(image, imgsz)
        outputs = self.session.run(None, {self.input_name: inp})
        return self._postprocess(
            outputs[0],
            r,
            pad,
            image.shape[:2],
            conf_thr,
            float(settings.iou_threshold),
        )

    # ------------------------------------------------------------------ draw
    def _draw(self, image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        output = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = CLASS_COLORS.get(detection.class_id, (120, 220, 60))

            label = f"{detection.class_name} {detection.confidence:.2f}"
            if detection.class_name == "NoHelmet":
                label = f"NO HELMET | {detection.confidence:.2f}"

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            ty = max(y1, th + 8)
            cv2.rectangle(
                output, (x1, ty - th - 6), (x1 + tw + 8, ty + baseline - 2), color, -1
            )
            cv2.putText(
                output,
                label,
                (x1 + 4, ty - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output

    # -------------------------------------------------------------- summaries
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
            vtype = violation.get("violation_type", "unknown")
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1

        return {
            "total_detections": len(detections),
            "total_violations": len(violations),
            "object_count": len(detections),
            "violation_count": len(violations),
            "classes": CLASS_NAMES,
            "model": self.model_label,
            "object_counts": object_counts,
            "class_counts": object_counts,  # the UI reads class_counts for the bars
            "violation_counts": violation_counts,
        }

    def _build_violations(
        self, detections: list[Detection]
    ) -> list[dict[str, Any]]:
        violations = []
        for detection in detections:
            if detection.class_name != "NoHelmet":
                continue
            violations.append(
                {
                    "violation_type": "no_helmet",
                    "label": VIOLATION_LABELS["no_helmet"],
                    "severity": SEVERITY["no_helmet"],
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

    # ------------------------------------------------------------- image path
    def process_image(
        self,
        image_path: Path,
        output_path: Path,
        conf_thr: float | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        detections = self._detect(image, conf_thr=conf_thr)
        violations = self._build_violations(detections)
        annotated = self._draw(image, detections)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Could not write annotated image: {output_path}")

        processing_ms = (time.perf_counter() - start) * 1000
        summary = self._build_summary(detections, violations)

        return {
            "detections": [d.to_dict() for d in detections],
            "violations": violations,
            "object_counts": summary["object_counts"],
            "violation_counts": summary["violation_counts"],
            "total_detections": len(detections),
            "total_violations": len(violations),
            "result_path": str(output_path),
            "processing_ms": processing_ms,
            "summary": summary,
        }

    def process_image_file(
        self,
        image_path: Path,
        output_path: Path,
        conf_thr: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compatibility wrapper used by the API image/upload routes."""
        if conf_thr is None:
            conf_thr = kwargs.get("conf_threshold")
        return self.process_image(
            Path(image_path), Path(output_path), conf_thr=conf_thr
        )

    # ------------------------------------------------------------- video path
    def process_video_file(
        self,
        video_path: Path,
        output_path: Path,
        conf_thr: float | None = None,
        max_frames: int | None = None,
        skip: int = 1,
    ) -> dict[str, Any]:
        """Analyse a video with a *bounded* amount of work.

        The old implementation ran the detector on every 2nd frame of the whole
        clip — on a free Render instance (0.1 CPU) a 30 s clip meant 20+ minutes
        of work, so the request never returned and users saw "nothing detected".

        Now the clip is sampled: at most `max_frames` frames are analysed, spread
        evenly across the whole video, and every source frame is still written to
        the output (annotations are held between analysed frames) so the result
        plays back at the original speed.
        """
        start = time.perf_counter()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or not math.isfinite(fps):
            fps = 25.0
        if frame_width <= 0 or frame_height <= 0:
            cap.release()
            raise ValueError("Video has invalid frame dimensions")

        budget = int(max_frames) if max_frames else int(settings.video_max_frames)
        budget = max(1, min(budget, int(settings.video_max_frames)))
        # video frames are analysed at a smaller size: faster and less memory,
        # with no measurable loss on the traffic classes
        video_imgsz = int(getattr(settings, "video_imgsz", 0) or self.imgsz)

        # analyse one frame every `step` frames, covering the entire clip
        if total_frames > 0:
            step = max(1, math.ceil(total_frames / budget))
        else:
            step = max(1, int(skip) if skip else 1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = output_path.with_name(f"{output_path.stem}_opencv.mp4")
        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, fps),
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not initialize the video encoder")

        all_detections: list[dict[str, Any]] = []
        all_violations: list[dict[str, Any]] = []
        unique_violation_flags: set[str] = set()
        class_counts: dict[str, int] = {}

        frame_index = 0
        processed_frames = 0
        last_annotated: np.ndarray | None = None
        detection_frames: list[int] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % step == 0 and processed_frames < budget:
                    detections = self._detect(
                        frame, conf_thr=conf_thr, imgsz=video_imgsz
                    )
                    violations = self._build_violations(detections)

                    for detection in detections:
                        class_counts[detection.class_name] = (
                            class_counts.get(detection.class_name, 0) + 1
                        )
                        item = detection.to_dict()
                        item["frame"] = frame_index
                        all_detections.append(item)

                    for violation in violations:
                        item = dict(violation)
                        item["frame"] = frame_index
                        all_violations.append(item)
                        unique_violation_flags.add(violation["violation_type"])

                    last_annotated = self._draw(frame, detections)
                    detection_frames.append(frame_index)
                    processed_frames += 1

                writer.write(last_annotated if last_annotated is not None else frame)
                frame_index += 1

        finally:
            cap.release()
            writer.release()

        if processed_frames == 0 or not raw_path.exists() or raw_path.stat().st_size == 0:
            raise ValueError("No frames could be decoded from the uploaded video")

        # OpenCV's mp4v is not playable in most browsers — transcode to H.264.
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            # Memory matters more than compression here: x264's default frame
            # threading peaks around 340 MB on 720p and gets the worker OOM-killed
            # on Render's free 512 MB instance. Single-threaded ultrafast with
            # lookahead disabled peaks near 90 MB and is still faster in wall time
            # because the instance has ~0.1 CPU anyway.
            completed = subprocess.run(
                [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-i", str(raw_path),
                    "-an",
                    "-filter_threads", "1",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "26",
                    "-threads", "1",
                    "-x264-params", "rc-lookahead=0:sync-lookahead=0:bframes=0",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if completed.returncode == 0 and output_path.exists():
                raw_path.unlink(missing_ok=True)
            else:
                logger.warning(
                    "ffmpeg transcode failed, serving OpenCV output: %s",
                    (completed.stderr or "")[-400:],
                )
                raw_path.replace(output_path)
        else:
            raw_path.replace(output_path)

        # snapshot of the last analysed frame (used as the result preview image)
        snapshot_path = output_path.with_suffix(".jpg")
        if last_annotated is not None:
            cv2.imwrite(str(snapshot_path), last_annotated)

        processing_ms = (time.perf_counter() - start) * 1000
        seconds = frame_index / fps if fps else 0.0

        summary = {
            "total_frames": frame_index,
            "processed_frames": processed_frames,
            "total_raw_detections": len(all_detections),
            "total_violations": len(all_violations),
            "unique_violation_flags": sorted(unique_violation_flags),
            "class_counts": class_counts,
            "frame_sampling_interval": step,
            "analysed_per_second": round(processed_frames / seconds, 2) if seconds else None,
            "model": self.model_label,
            "classes": CLASS_NAMES,
            "processing_ms": round(processing_ms, 1),
            "video_url": str(output_path),
            "note": (
                f"Analysed {processed_frames} of {frame_index} frames "
                f"(1 in {step}) to stay within the free-tier CPU budget."
            ),
        }

        return {
            "detections": all_detections[:200],
            "violations": all_violations,
            "total_frames": frame_index,
            "processed_frames": processed_frames,
            "total_raw_detections": len(all_detections),
            "unique_violation_flags": sorted(unique_violation_flags),
            "snapshot_path": str(snapshot_path) if last_annotated is not None else None,
            "video_url": str(output_path),
            "result_path": str(output_path),
            "processing_ms": processing_ms,
            "summary": summary,
        }


# Singleton
_detector: SafeCityDetector | None = None


def get_detector() -> SafeCityDetector:
    global _detector
    if _detector is None:
        _detector = SafeCityDetector()
    return _detector


def new_job_id() -> str:
    return uuid.uuid4().hex
