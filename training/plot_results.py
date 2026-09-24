"""Plot the metrics of an actual YOLOv5 run; never invent a loss or mAP curve.

Reads the `results.csv` written by `yolov5/train.py` during a real training run.
If that CSV is missing, empty or unreadable this fails loudly instead of drawing
a placeholder chart, because a chart that no run produced is not a result.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- the Agg backend must be set first

REQUIRED_COLUMNS = ("epoch", "train/box_loss", "metrics/mAP_0.5")
LOSS_KEYS = ("train/box_loss", "train/obj_loss", "train/cls_loss")
VAL_LOSS_KEYS = ("val/box_loss", "val/obj_loss", "val/cls_loss")


class ResultsError(ValueError):
    """Raised when the supplied training output cannot be charted honestly."""


def read_results(csv_path):
    path = Path(csv_path)
    if not path.is_file():
        raise ResultsError(f"Training results CSV missing: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = [column.strip() for column in (reader.fieldnames or [])]
        rows = []
        for row in reader:
            record = {}
            for key, value in row.items():
                if key is None or value is None:
                    continue
                key = key.strip()
                try:
                    record[key] = float(value)
                except ValueError:
                    record[key] = value
            if record:
                rows.append(record)
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise ResultsError(f"Training results CSV lacks {', '.join(missing)}")
    if not rows:
        raise ResultsError("Training results CSV has no epoch rows; no run to chart")
    for row in rows:
        for column in REQUIRED_COLUMNS:
            value = row.get(column)
            if not isinstance(value, float) or math.isnan(value):
                raise ResultsError(f"Non-numeric {column} in epoch row {row.get('epoch')}")
    return {"rows": rows, "columns": header, "csv_sha256": digest}


def summarize(results):
    rows = results["rows"]
    best = max(rows, key=lambda row: row["metrics/mAP_0.5"])
    return {
        "csv_sha256": results["csv_sha256"],
        "epochs_recorded": len(rows),
        "first_epoch": rows[0],
        "last_epoch": rows[-1],
        "best_epoch_by_mAP50": best.get("epoch"),
        "best_mAP50": best["metrics/mAP_0.5"],
        "note": "Values are copied from the run's own CSV; they are not estimated or smoothed.",
    }


def _columns(rows, keys):
    return [(key, [row[key] for row in rows]) for key in keys if key in rows[0]]


def plot(results, output_png):
    rows = results["rows"]
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (
        (axes[0][0], LOSS_KEYS, "Training losses"),
        (axes[0][1], VAL_LOSS_KEYS, "Validation losses"),
        (axes[1][0], ("metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95"), "Precision, recall and mAP"),
        (axes[1][1], ("x/lr0", "x/lr1", "x/lr2"), "Learning rate"),
    )
    for axis, keys, title in panels:
        for key, values in _columns(rows, keys):
            axis.plot(epochs, values, label=key)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        if axis.get_legend_handles_labels()[1]:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "not recorded by this run", ha="center", va="center", transform=axis.transAxes)
    figure.tight_layout()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=120)
    plt.close(figure)
    return output_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="results.csv of an actual training run")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the chart and report")
    args = parser.parse_args()
    results = read_results(args.csv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    chart = plot(results, output / "training_curves.png")
    summary = summarize(results)
    summary["chart"] = str(chart)
    summary["source_csv"] = str(args.csv)
    (output / "chart_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
