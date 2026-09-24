#!/usr/bin/env python
"""Export the trained YOLOv5 traffic checkpoint (models/best.pt) to ONNX.

The deployed app runs ONNX Runtime only (no PyTorch in the container), so after
retraining you must re-export the weights once:

    python scripts/export_onnx.py --weights models/best.pt --out models/best.onnx

Needs PyTorch locally or in Colab (it is NOT a runtime dependency):

    pip install -r yolov5/requirements.txt      # or: pip install torch torchvision

The exported file is what Render ships — commit it:

    git add models/best.onnx && git commit -m "update detector weights" && git push
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLOV5 = ROOT / "yolov5"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--out", default=str(ROOT / "models" / "best.onnx"))
    ap.add_argument("--imgsz", type=int, default=640, help="inference size (match training)")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--static",
        action="store_true",
        help="export a fixed-shape model (slightly faster, only runs at --imgsz)",
    )
    args = ap.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not weights.is_file():
        print(f"✗ weights not found: {weights}", file=sys.stderr)
        return 1
    if weights.stat().st_size < 1024:
        print(f"✗ {weights} is a Git LFS pointer, not real weights", file=sys.stderr)
        return 1
    if not (YOLOV5 / "export.py").is_file():
        print(f"✗ vendored YOLOv5 not found at {YOLOV5}", file=sys.stderr)
        return 1

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "✗ PyTorch is required for the export step (not needed at runtime).\n"
            "  pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cpu",
            file=sys.stderr,
        )
        return 1

    cmd = [
        sys.executable, str(YOLOV5 / "export.py"),
        "--weights", str(weights),
        "--include", "onnx",
        "--imgsz", str(args.imgsz),
        "--opset", str(args.opset),
        "--device", args.device,
    ]
    if not args.static:
        cmd.append("--dynamic")

    print("→", " ".join(cmd))
    if subprocess.run(cmd, cwd=str(YOLOV5)).returncode != 0:
        print("✗ export failed", file=sys.stderr)
        return 1

    produced = weights.with_suffix(".onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    if produced != out and produced.is_file():
        produced.replace(out)

    if not out.is_file():
        print(f"✗ expected {out} after export", file=sys.stderr)
        return 1

    mb = out.stat().st_size / 1e6
    print(f"\n✓ wrote {out} ({mb:.1f} MB)")
    print("\nNext:")
    print(f"  git add {out.relative_to(ROOT)}")
    print('  git commit -m "update detector weights" && git push   # Render redeploys')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
