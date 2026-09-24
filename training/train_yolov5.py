"""Validate data, train YOLOv5, evaluate the best checkpoint, optionally install it."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.dataset import DatasetError, validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/traffic.yaml")
    parser.add_argument("--weights", default="yolov5s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help="0 for GPU, cpu for CPU")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--install-model",
        action="store_true",
        help="Copy evaluated best.pt into models/ for the API",
    )
    parser.add_argument(
        "--overwrite-model",
        action="store_true",
        help="Explicitly allow replacing a local best.pt",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.img_size < 32 or args.img_size % 32:
        parser.error("epochs/batch must be positive; workers nonnegative; img-size a positive multiple of 32")
    if args.overwrite_model and not args.install_model:
        parser.error("--overwrite-model requires --install-model")
    try:
        data = validate_dataset(args.data)
    except (DatasetError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(data, indent=2))
    if args.validate_only:
        return
    target = ROOT / "models/best.pt"
    if args.install_model and target.exists() and not args.overwrite_model:
        parser.error(
            "models/best.pt exists; back it up and explicitly use --overwrite-model to replace it"
        )
    if not (ROOT / "yolov5/train.py").is_file():
        parser.error("Vendored YOLOv5 source missing")
    run_name = (
        datetime.now(timezone.utc).strftime("safecity-%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:6]
    )
    project = ROOT / "runs/train"
    run = project / run_name
    train_cmd = [
        sys.executable,
        str(ROOT / "yolov5/train.py"),
        "--img",
        str(args.img_size),
        "--batch",
        str(args.batch),
        "--epochs",
        str(args.epochs),
        "--data",
        str(args.data.resolve()),
        "--weights",
        args.weights,
        "--workers",
        str(args.workers),
        "--project",
        str(project),
        "--name",
        run_name,
    ]
    if args.device:
        train_cmd += ["--device", args.device]
    subprocess.run(train_cmd, cwd=ROOT, check=True)
    best = run / "weights/best.pt"
    if not best.is_file():
        raise SystemExit("Training did not produce best.pt; nothing installed")
    val_cmd = [
        sys.executable,
        str(ROOT / "yolov5/val.py"),
        "--weights",
        str(best),
        "--data",
        str(args.data.resolve()),
        "--img",
        str(args.img_size),
        "--batch",
        str(args.batch),
        "--project",
        str(ROOT / "runs/val"),
        "--name",
        run_name,
    ]
    if args.device:
        val_cmd += ["--device", args.device]
    subprocess.run(val_cmd, cwd=ROOT, check=True)
    report = {
        "dataset": data,
        "training_command": train_cmd,
        "validation_command": val_cmd,
        "checkpoint": str(best),
        "validation_directory": str(ROOT / "runs/val" / run_name),
        "training_metrics_csv": str(run / "results.csv"),
        "package_versions": {
            p: version(p)
            for p in ("torch", "torchvision", "ultralytics", "numpy", "pillow")
        },
        "note": "Inspect actual metrics and held-out examples; successful commands do not guarantee acceptable accuracy.",
    }
    (run / "training_report.json").write_text(json.dumps(report, indent=2))
    if args.install_model:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)
        print(
            "Installed evaluated checkpoint at models/best.pt. Restart the API and run scripts.verify_release."
        )
    print(f"Best checkpoint: {best}\nRun report: {run / 'training_report.json'}")


if __name__ == "__main__":
    main()
