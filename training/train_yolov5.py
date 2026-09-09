"""Train SafeCityAI's custom YOLOv5 traffic detector.

Run this script from the repository root after placing the dataset under dataset/.
Training is recommended on Google Colab or another CUDA GPU.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
YOLOV5 = ROOT / "yolov5"
DATA = ROOT / "dataset" / "traffic.yaml"

if not (YOLOV5 / "train.py").exists():
    raise SystemExit("YOLOv5 repository not found. Clone ultralytics/yolov5 into the project or run the Colab notebook.")

cmd = [
    sys.executable, str(YOLOV5 / "train.py"),
    "--img", "640",
    "--batch", "16",
    "--epochs", "50",
    "--data", str(DATA),
    "--weights", "yolov5s.pt",
    "--project", str(ROOT / "runs"),
    "--name", "safecity-yolov5",
]
subprocess.run(cmd, check=True)
