"""Check the actual checkpoint and optionally run supplied evaluation examples.

Usage: python -m scripts.verify_release --image /path/to/traffic.jpg [--video clip.mp4]
Does not train/download a fallback, invent accuracy, or alter the application DB.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from backend.app.config import settings
from backend.app.services.detector import YOLODetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/release-check"))
    args = parser.parse_args()
    if not settings.model_path.is_file():
        print(
            "BLOCKED: trusted custom checkpoint missing at MODEL_PATH. No fallback will be downloaded."
        )
        return 1
    try:
        detector = YOLODetector()
        with settings.model_path.open("rb") as checkpoint:
            checksum = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        report = {
            "checkpoint": settings.model_path.name,
            "sha256": checksum,
            "classes": detector.names,
            "images": [],
            "video": None,
            "note": "Inference smoke test only; manual comparison and held-out accuracy evaluation required.",
        }
        args.output.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(args.image):
            result = detector.process_image_file(
                path, args.output / f"image-{index}.jpg"
            )
            report["images"].append(
                {
                    "source": path.name,
                    "summary": result["summary"],
                    "detections": result["detections"],
                }
            )
        if args.video:
            result = detector.process_video_file(
                args.video, args.output / "video.mp4", max_frames=30
            )
            report["video"] = result["summary"]
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(f"Checkpoint loaded. Report: {args.output / 'report.json'}")
        if not args.image or not args.video:
            print(
                "INCOMPLETE: supply at least one representative image and a video for release smoke tests."
            )
            return 1
        print(
            "Smoke checks passed. Visually inspect annotations; this does not certify model accuracy."
        )
        return 0
    except Exception as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
