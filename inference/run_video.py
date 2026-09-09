"""Simple video inference entry point for the deployed SafeCityAI detector."""
import argparse
from pathlib import Path

from backend.app.services.detector import get_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Input video path")
    parser.add_argument("--output", default="outputs/detected_video.mp4")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()
    result = get_detector().process_video_file(Path(args.source), Path(args.output), conf_thr=args.conf)
    print(result["summary"])


if __name__ == "__main__":
    main()
