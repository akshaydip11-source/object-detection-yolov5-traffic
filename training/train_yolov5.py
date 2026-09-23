"""Validate the SafeCityAI dataset, fine-tune YOLOv5s, evaluate, and export."""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
YOLOV5 = ROOT / "yolov5"
DATA = ROOT / "dataset" / "traffic.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_dataset() -> tuple[list[str], int, int]:
    """Fail early on absent images, mismatched labels, or invalid YOLO boxes."""
    if not DATA.is_file():
        raise SystemExit(f"Dataset config not found: {DATA}")
    config = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    raw_names = config.get("names")
    names = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names or [])
    expected = ["Helmet", "No_Helmet", "License_Plate"]
    if names != expected:
        raise SystemExit(f"{DATA} must define classes in this order: {expected}; got {names}")

    split_counts: dict[str, int] = {}
    for split in ("train", "val"):
        images_dir = ROOT / "dataset" / "images" / split
        labels_dir = ROOT / "dataset" / "labels" / split
        images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if images_dir.exists() else []
        if not images:
            raise SystemExit(f"No {split} images found in {images_dir}. Add the annotated dataset before training.")

        seen_classes: set[int] = set()
        for image in images:
            label = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
            if not label.is_file():
                raise SystemExit(f"Missing label file for {image}: expected {label}")
            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                fields = line.split()
                if len(fields) != 5:
                    raise SystemExit(f"{label}:{line_no}: expected class x_center y_center width height")
                try:
                    class_id = int(fields[0])
                    coords = [float(value) for value in fields[1:]]
                except ValueError as exc:
                    raise SystemExit(f"{label}:{line_no}: values must be numeric") from exc
                if class_id < 0 or class_id >= len(names):
                    raise SystemExit(f"{label}:{line_no}: class id must be 0, 1, or 2")
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in coords):
                    raise SystemExit(f"{label}:{line_no}: box coordinates must be finite and normalized to [0, 1]")
                if coords[2] == 0 or coords[3] == 0:
                    raise SystemExit(f"{label}:{line_no}: box width and height must be greater than zero")
                seen_classes.add(class_id)

        missing = sorted(set(range(len(names))) - seen_classes)
        if missing:
            missing_names = [names[class_id] for class_id in missing]
            raise SystemExit(f"{split} labels contain no examples for: {', '.join(missing_names)}")
        split_counts[split] = len(images)

    return names, split_counts["train"], split_counts["val"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--weights", default="yolov5s.pt", help="COCO-pretrained YOLOv5 checkpoint")
    parser.add_argument("--video", type=Path, help="Optional held-out video for inference after training")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not (YOLOV5 / "train.py").is_file():
        raise SystemExit(f"Vendored YOLOv5 training code not found: {YOLOV5 / 'train.py'}")
    names, train_count, val_count = validate_dataset()
    print(f"Validated {train_count} train and {val_count} validation images for {names}")
    if args.validate_only:
        return

    run_name = f"safecity-yolov5-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = ROOT / "runs" / "train" / run_name
    hyp = YOLOV5 / "data" / "hyps" / "hyp.scratch-low.yaml"
    train_cmd = [
        sys.executable,
        str(YOLOV5 / "train.py"),
        "--img",
        "640",
        "--batch",
        str(args.batch),
        "--epochs",
        str(args.epochs),
        "--data",
        str(DATA),
        "--hyp",
        str(hyp),
        "--weights",
        args.weights,
        "--project",
        str(run_dir.parent),
        "--name",
        run_name,
    ]
    subprocess.run(train_cmd, check=True, cwd=YOLOV5)

    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Training finished without a best checkpoint at {best}")
    val_dir = ROOT / "runs" / "val" / run_name
    subprocess.run(
        [
            sys.executable,
            str(YOLOV5 / "val.py"),
            "--weights",
            str(best),
            "--data",
            str(DATA),
            "--img",
            "640",
            "--batch",
            str(args.batch),
            "--task",
            "val",
            "--project",
            str(val_dir.parent),
            "--name",
            run_name,
        ],
        check=True,
        cwd=YOLOV5,
    )

    export_dir = ROOT / "backend" / "weights"
    export_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(YOLOV5 / "export.py"),
            "--weights",
            str(best),
            "--include",
            "onnx",
            "--img",
            "640",
            "640",
        ],
        check=True,
        cwd=YOLOV5,
    )
    exported = best.with_suffix(".onnx")
    onnx_path = export_dir / "yolov5_custom.onnx"
    onnx_path.write_bytes(exported.read_bytes())
    names_path = export_dir / "traffic.names"
    names_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Best checkpoint: {best}")
    print(f"Validation results and plots: {val_dir}")
    print(f"ONNX export: {onnx_path}")
    print(f"Class names: {names_path}")
    print("For the API, set MODEL_PATH=weights/yolov5_custom.onnx")
    print("and CLASS_NAMES_PATH=weights/traffic.names, then restart the server.")

    if args.video:
        if not args.video.is_file():
            raise SystemExit(f"Video not found: {args.video}")
        subprocess.run(
            [
                sys.executable,
                str(YOLOV5 / "detect.py"),
                "--weights",
                str(best),
                "--source",
                str(args.video.resolve()),
                "--img",
                "640",
                "--conf-thres",
                "0.5",
                "--project",
                str(ROOT / "outputs"),
                "--name",
                run_name,
                "--save-txt",
                "--save-conf",
            ],
            check=True,
            cwd=YOLOV5,
        )


if __name__ == "__main__":
    main()
