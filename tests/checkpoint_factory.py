"""Generate a random test-only checkpoint. Never use it as the internship model."""

from pathlib import Path
import sys


def create(path: Path):
    import torch

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "yolov5"))
    from models.yolo import DetectionModel

    torch.set_num_threads(2)
    model = DetectionModel(str(root / "yolov5/models/yolov5n.yaml"), nc=3)
    model.names = {0: "Helmet", 1: "NoHelmet", 2: "LicensePlate"}
    torch.save({"model": model}, path)


if __name__ == "__main__":
    create(Path(sys.argv[1]))
    print("UNTRAINED test fixture written; NOT a model submission or accuracy result.")
