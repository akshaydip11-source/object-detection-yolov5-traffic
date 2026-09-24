"""Evaluate an actual trained checkpoint on validation and test data.

Reports measured precision, recall, mAP@0.5 and mAP@0.5:0.95 per split and per
confidence threshold. The operating threshold is chosen on validation only, so the
test numbers stay a holdout; no metric here is estimated, reused from another run
or written by hand.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.dataset import DatasetError, validate_dataset  # noqa: E402 -- sys.path is set above


def load_runner():
    """Return YOLOv5's validation entry point from the vendored source."""
    yolov5 = str(ROOT / "yolov5")
    if yolov5 not in sys.path:
        sys.path.insert(0, yolov5)
    import val

    return val.run


def checkpoint_digest(weights):
    with Path(weights).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _metrics_tuple(metrics):
    return {
        "precision": float(metrics[0]),
        "recall": float(metrics[1]),
        "mAP50": float(metrics[2]),
        "mAP50_95": float(metrics[3]),
    }


def evaluate(
    weights,
    data,
    *,
    img_size=640,
    batch=16,
    device="cpu",
    thresholds=(0.25, 0.5, 0.65),
    output=None,
    workers=2,
    runner=None,
    names=("Helmet", "No_Helmet", "License_Plate"),
):
    weights = Path(weights)
    if not weights.is_file():
        raise ValueError(f"Checkpoint missing: {weights}")
    if img_size < 32 or img_size % 32:
        raise ValueError("img-size must be a positive multiple of 32")
    if batch < 1 or workers < 0:
        raise ValueError("batch must be positive and workers nonnegative")
    thresholds = [float(value) for value in thresholds]
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Confidence thresholds must each be within [0, 1]")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Confidence thresholds must be distinct")
    try:
        audit = validate_dataset(data)
    except (DatasetError, OSError) as exc:
        raise ValueError(str(exc)) from exc
    if runner is None:
        runner = load_runner()
    output = Path(output) if output else ROOT / "runs/evaluation"
    output.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in ("val", "test"):
        splits[split] = {}
        for threshold in thresholds:
            metrics, maps, _ = runner(
                data=str(Path(data).resolve()),
                weights=str(weights.resolve()),
                batch_size=batch,
                imgsz=img_size,
                conf_thres=threshold,
                iou_thres=0.6,
                task=split,
                device=device,
                workers=workers,
                project=str(output / split),
                name=f"conf-{threshold}",
                plots=False,
                verbose=False,
            )
            row = _metrics_tuple(metrics)
            row["per_class_mAP50_95"] = {
                name: float(value) for name, value in zip(names, maps)
            }
            splits[split][str(threshold)] = row
    best = max(
        thresholds,
        key=lambda value: (
            splits["val"][str(value)]["mAP50"],
            splits["val"][str(value)]["mAP50_95"],
            -value,
        ),
    )
    report = {
        "checkpoint": str(weights.resolve()),
        "checkpoint_sha256": checkpoint_digest(weights),
        "dataset_config": str(Path(data).resolve()),
        "dataset": audit,
        "img_size": img_size,
        "device": device,
        "confidence_thresholds": thresholds,
        "splits": splits,
        "selected_confidence_threshold": best,
        "selection_rule": "highest validation mAP@0.5, ties broken by validation mAP@0.5:0.95",
        "test_at_selected_threshold": splits["test"][str(best)],
        "note": "Measured on the configured splits. Selection uses validation only. A checkpoint with low mAP is reported as low; nothing here certifies accuracy, legal compliance or deployment.",
    }
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.5, 0.65])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/evaluation")
    args = parser.parse_args()
    report = evaluate(
        args.weights,
        args.data,
        img_size=args.img_size,
        batch=args.batch,
        device=args.device,
        thresholds=args.conf,
        output=args.output,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
