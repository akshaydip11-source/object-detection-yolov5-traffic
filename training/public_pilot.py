"""Bounded CPU pilot on reviewed public data, not an automatic deployment gate."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from training.public_dataset import annotation

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / "runs/public-pilot-data/data.yaml"
    if not data.is_file():
        raise SystemExit("Prepare public data before running the pilot")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    before = set((ROOT / "runs/train").glob("safecity-*"))
    subprocess.run([
        sys.executable, "training/train_yolov5.py", "--data", str(data),
        "--weights", "yolov5n.pt", "--epochs", "20", "--img-size", "320",
        "--batch", "16", "--workers", "2", "--device", "cpu",
    ], cwd=ROOT, check=True)
    runs = set((ROOT / "runs/train").glob("safecity-*")) - before
    if len(runs) != 1:
        raise SystemExit("Expected one completed training run")
    run = runs.pop()
    best = run / "weights/best.pt"
    report = json.loads((run / "training_report.json").read_text())
    sys.path.insert(0, str(ROOT / "yolov5"))
    import val

    evaluations = {}
    for split in ("val", "test"):
        metrics, maps, timings = val.run(
            data=str(data), weights=str(best), batch_size=16, imgsz=640,
            device="cpu", workers=2, task=split, verbose=True,
            project=str(ROOT / "runs/public-pilot-evaluation"), name=split,
            plots=True,
        )
        evaluations[split] = {
            "image_size": 640, "precision": float(metrics[0]), "recall": float(metrics[1]),
            "mAP50": float(metrics[2]), "mAP50_95": float(metrics[3]),
            "per_class_mAP50_95": dict(zip(["Helmet", "NoHelmet", "LicensePlate"], map(float, maps))),
            "timings_ms": list(map(float, timings)),
        }
    with best.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report.update({
        "status": "experimental_public_data_pilot_not_approved_for_deployment",
        "checkpoint_sha256": digest,
        "evaluation": evaluations,
        "limitations": "Small source-stratified subset, 20 CPU epochs at 320px. No verified camera/video-level independence or target-domain field evaluation. No production artifact installation or enforcement approval.",
    })
    result = ROOT / "runs/public-pilot-results"
    result.mkdir(parents=True, exist_ok=False)
    (result / "pilot_report.json").write_text(json.dumps(report, indent=2))
    shutil.copy2(best, result / "experimental_best.pt")
    shutil.copy2(run / "results.csv", result / "training_metrics.csv")
    for filename in ("provenance.json", "ATTRIBUTION.md"):
        shutil.copy2(data.parent / filename, result / filename)
    print(json.dumps(report, indent=2))
    annotation("notice", "Experimental model evaluation", json.dumps({
        "status": report["status"], "checkpoint_sha256": digest,
        "evaluation": evaluations, "limitations": report["limitations"],
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        annotation("error", "Public training pilot failed", f"{type(exc).__name__}: {exc}")
        raise
