"""Review the recorded pilot and validation resolution, without reusing the test set."""

import csv
import hashlib
import json
from pathlib import Path
import sys

from training.dataset import validate_dataset
from training.public_dataset import annotation

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "runs/previous-pilot/public-pilot-results"
    report = json.loads((folder / "pilot_report.json").read_text())
    weights = folder / "experimental_best.pt"
    with weights.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != report["checkpoint_sha256"]:
            raise ValueError("Pilot artifact checksum mismatch; refusing deserialization")
    data = ROOT / "runs/public-pilot-data/data.yaml"
    audit = validate_dataset(data)
    if audit["fingerprint_sha256"] != report["dataset"]["fingerprint_sha256"]:
        raise ValueError("Reconstructed data differs from the original training run")
    with (folder / "training_metrics.csv").open() as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    diagnostics = {
        "source_run": 35942772831,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "epochs_recorded": len(rows),
        "training_first_epoch": rows[0],
        "training_last_epoch": rows[-1],
        "best_training_validation_mAP50": max(r.get("metrics/mAP_0.5", 0) for r in rows),
        "baseline_validation_640": report["evaluation"]["val"],
        "note": "Resolution diagnostic uses validation only. The baseline test result was already inspected; no new independent-test claim or deployment approval.",
    }
    sys.path.insert(0, str(ROOT / "yolov5"))
    import val

    metrics, maps, _ = val.run(
        data=str(data), weights=str(weights), batch_size=16, imgsz=320,
        device="cpu", workers=2, task="val", verbose=True,
        project=str(ROOT / "runs/pilot-review"), name="validation-320", plots=True,
    )
    diagnostics["validation_320"] = {
        "precision": float(metrics[0]), "recall": float(metrics[1]),
        "mAP50": float(metrics[2]), "mAP50_95": float(metrics[3]),
        "per_class_mAP50_95": dict(zip(["Helmet", "NoHelmet", "LicensePlate"], map(float, maps))),
    }
    output = ROOT / "runs/pilot-review/diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnostics, indent=2))
    annotation("notice", "Pilot training diagnostics", json.dumps(diagnostics))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        annotation("error", "Pilot review failed", f"{type(exc).__name__}: {exc}")
        raise
