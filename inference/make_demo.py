"""Make a genuine 30-second annotated clip from operator-supplied street footage.

No stock footage, generated scene, looped still or synthetic detection is substituted.
This is offline inference; encode/display FPS is not a real-time performance claim.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2
from imageio_ffmpeg import get_ffmpeg_exe


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Video could not be opened')
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not all(math.isfinite(v) and v > 0 for v in (fps, count, width, height)):
            raise ValueError('Video duration/dimensions cannot be verified')
        if width * height > 12_000_000:
            raise ValueError('Video exceeds 12 million pixels per frame')
        return {'fps': fps, 'frames': int(count), 'seconds': count / fps}
    finally:
        capture.release()


def validate_window(info, start):
    if not math.isfinite(start) or start < 0 or start + 30 > info['seconds'] + 0.01:
        raise ValueError('Provide an actual continuous 30-second source window; never loop/pad short footage')


def make_demo(source, checkpoint, credit, output, *, start=0, conf=0.5, timeout=600):
    source, checkpoint, credit, output = map(lambda p: Path(p).resolve(), (source, checkpoint, credit, output))
    if not (source.is_file() and checkpoint.is_file() and credit.is_file()):
        raise ValueError('Supply real source footage, a trusted custom checkpoint and its footage credit/license file')
    if output.exists():
        raise ValueError('Output exists; refusing overwrite')
    if not math.isfinite(conf) or not 0 <= conf <= 1 or not 1 <= timeout <= 600:
        raise ValueError('Invalid confidence threshold or offline processing time limit')
    validate_window(video_info(source), start)
    from backend.app.config import settings
    from backend.app.services.detector import YOLODetector

    # Separate CLI process only: do not increase any HTTP endpoint's resource budget.
    settings.model_path = checkpoint
    settings.max_video_seconds = timeout
    detector = YOLODetector()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.street-demo-', dir=output.parent) as temp:
        root = Path(temp)
        clip = root / 'input-30s.mp4'
        subprocess.run([
            get_ffmpeg_exe(), '-nostdin', '-v', 'error', '-n', '-ss', str(start),
            '-i', str(source), '-t', '30', '-an', '-vf', 'fps=10', '-c:v', 'libx264',
            '-threads', '2', '-pix_fmt', 'yuv420p', str(clip),
        ], check=True, timeout=120)
        info = video_info(clip)
        if info['frames'] != 300 or abs(info['seconds'] - 30) > 0.1:
            raise ValueError('Source did not decode into a continuous 30-second clip')
        bundle = root / 'bundle'
        bundle.mkdir()
        tick = time.perf_counter()
        result = detector.process_video_file(clip, bundle / 'demo-30s.mp4', conf_thr=conf, max_frames=300, skip=1)
        elapsed = time.perf_counter() - tick
        rendered = video_info(bundle / 'demo-30s.mp4')
        if result['summary']['frames_processed'] != 300 or abs(rendered['seconds'] - 30) > 0.1 or result['summary']['video_codec'] != 'h264':
            raise ValueError('Incomplete or non-browser-compatible demo output')
        with checkpoint.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        report = {
            'source_filename': source.name, 'source_start_seconds': start,
            'duration_seconds': rendered['seconds'], 'display_fps': rendered['fps'],
            'checkpoint_sha256': digest, 'confidence_threshold': conf, 'device': settings.device,
            'processing_seconds': elapsed, 'offline_frames_per_second': 300 / elapsed,
            'real_time_certified': False, 'summary': result['summary'],
            'note': 'Offline model output; manual scene/box/rights review required. A drawn No_Helmet box is not a legal finding. Display FPS is not inference throughput.',
        }
        (bundle / 'demo_report.json').write_text(json.dumps(report, indent=2))
        shutil.copyfile(credit, bundle / 'FOOTAGE_LICENSE.txt')
        if output.exists():
            raise ValueError('Output appeared during processing; refusing overwrite')
        bundle.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--source-license', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/street-demo'))
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--processing-timeout', type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(make_demo(args.source, args.weights, args.source_license, args.output,
                              start=args.start, conf=args.conf, timeout=args.processing_timeout), indent=2))


if __name__ == '__main__':
    main()
