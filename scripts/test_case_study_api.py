"""Authenticated case-study image upload. Set SAFECITY_TOKEN locally; never commit it."""

import argparse
import json
import mimetypes
import os
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--image', type=Path, required=True)
    args = parser.parse_args()
    token = os.environ.get('SAFECITY_TOKEN')
    if not token:
        parser.error('Set SAFECITY_TOKEN to an access token obtained from /api/auth/login')
    with args.image.open('rb') as stream:
        result = requests.post(args.base_url.rstrip('/') + '/api/detect/predict',
                               headers={'Authorization': f'Bearer {token}'},
                               files={'file': (args.image.name, stream, mimetypes.guess_type(args.image.name)[0] or 'image/jpeg')},
                               data={'conf_threshold': '0.5'}, timeout=120)
    if not result.ok:
        raise SystemExit(f'Inference failed with HTTP {result.status_code}; inspect API/model readiness. No tokens logged.')
    data = result.json()
    # Do not put signed private-media URLs into logs.
    print(json.dumps({k: data[k] for k in ['job_id', 'detections', 'box_format', 'coordinate_system']}, indent=2))


if __name__ == '__main__':
    main()
