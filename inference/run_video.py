"""Simple video inference entry point for the deployed SafeCityAI detector."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.services.detector import get_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Input video path")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "detected_video.mp4"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--skip", type=int, default=2)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    result = get_detector().process_video_file(
        source,
        output,
        conf_thr=args.conf,
        max_frames=args.max_frames,
        skip=args.skip,
    )
    print(result["summary"])


if __name__ == "__main__":
    main()
