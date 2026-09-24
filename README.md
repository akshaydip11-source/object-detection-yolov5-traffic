# SafeCityAI — Custom YOLOv5 Traffic Detection

> **Case-study requirements:** See [the deliverable checklist and commands](docs/CASE_STUDY.md).
> The specified final architecture is YOLOv5s/m. The earlier nano pilot is not the
> final model. Full-resolution training and a real 30-second street demo remain
> outstanding; a screenshot or synthetic test video does not fulfill them.


A FastAPI/PyTorch internship prototype for **Helmet, NoHelmet and LicensePlate**
detection, with a same-origin HTML/CSS/JS console, annotated media and review records.

> **Current status:** source-level checks pass, but **the real `best.pt` and the
> previously Postman-tested backend were not supplied**. The local adapter is not
> a verified copy of that backend. No weights were found in the workspace, repository
> history or GitHub releases. The linked Render deployment was checked and still
> reports the YOLO11 ONNX fallback. Final trained-model validation is required.
> See the [annotation failure investigation](docs/ANNOTATION_FIX.md).
> [Audit](docs/AUDIT.md) · [Deployment runbook](docs/DEPLOYMENT.md)

## Scope

- Image, video and browser-webcam frames; bounding boxes and JSON responses.
- Checkpoint class names/order are validated; no hardcoded COCO class mapping.
- Only **NoHelmet** generates a human-review flag, not proof of an offence.
- Sign-in required for detection and operational data. Analysts see their own jobs;
  officers/admins can review all jobs, dashboard, tickets, PDFs and CSV exports.
- Media is no longer publicly mounted: short-lived, path-scoped signed links are used.
- No seatbelt/triple-riding/signal/speed inference, OCR, tracking or connected CCTV.
- No pretrained fallback; the old YOLO11 ONNX detector has been removed.
- Synthetic sample illustrations test uploads only. No accuracy/latency is claimed.

## Local setup — Python 3.11

From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
# Linux/Windows CPU builds. For macOS or CUDA, install the matching PyTorch pair.
python -m pip install 'torch>=2.6,<3' 'torchvision>=0.21,<1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/setup_env.py
```

The setup script creates an ignored `.env` with a random signing secret and never
prints it or overwrites an existing file. On Debian/Ubuntu, OpenCV needs
`sudo apt-get install libgl1 libglib2.0-0` (run `apt-get update` first).

1. Supply your **trusted custom YOLOv5** checkpoint as `models/best.pt`, or set
   `MODEL_PATH` in `.env`. Relative paths resolve from the repository root.
2. Create an account (password is prompted, not passed through the command line):

```bash
python -m backend.app.manage create-user --email you@example.com --name "Project Admin" --role admin
python server.py
```

Open <http://localhost:8000/login>. Console: `/app`; API docs: `/api/docs`;
health: `/api/health`. `api.server:app` is the equivalent Uvicorn deployment entry
point; `bash scripts/run.sh` is the Unix launcher.

Without weights, **local mode** serves the UI but reports `status: degraded`,
`model_loaded: false`; authenticated inference returns **503**. **Production mode
refuses to start** without valid weights, an explicit strong secret, and demo mode
disabled. A database containing public demo identities is also refused.
Restart after replacing a checkpoint.

Never rename a YOLOv8/11/ONNX model to substitute for a YOLOv5 checkpoint. PyTorch
`.pt` deserialization can execute code: **only load trusted weights**. Vendored
YOLOv5 uses Ultralytics utility dependencies, not the YOLO11 inference API.
Runtime package auto-install is disabled. There is no default model download. A
deployment administrator can explicitly provision a trusted artifact using HTTPS
`MODEL_URL` plus mandatory `MODEL_SHA256`; the bounded download is verified before
atomic replacement, and configured integrity is checked before deserialization.

## Accounts and permissions

`REGISTRATION_ENABLED=false` and `DEMO_MODE=false` by default. Admins provision
accounts with `backend.app.manage create-user`, choosing `admin`, `officer` or
`analyst`. If explicitly enabled, public registration creates analysts only.

| Capability | Analyst | Officer/Admin |
| --- | --- | --- |
| Inference and own job history/media | Yes | Yes |
| Other users' jobs/media | No | Yes |
| Dashboard, review records, PDF/CSV | No | Yes |
| Create/update image review tickets | No | Yes |
| User list | No | Admin only |

Optional **disposable local demo only**: `DEMO_MODE=true` creates
`admin@safecity.ai / admin123`, `officer@safecity.ai / officer123`, and
`analyst@safecity.ai / analyst123`. These are public example credentials, never
production credentials. Turning the flag off does not remove accounts; use a fresh
DB before deployment. Old ONNX heuristic records are not valid custom-model results.

The browser stores access tokens in **sessionStorage**. Logout clears the browser
copy, not previously issued tokens; these expire on their configured schedule.
Media URLs are bearer links valid for 10 minutes by default, with active-account
and job ownership checks. Do not share them; refresh job history for fresh links.

## API / Postman

1. `POST /api/auth/login`, JSON `{"email":"you@example.com","password":"..."}`.
2. Use the returned `access_token` as **Authorization → Bearer Token**.
3. `POST /api/detect/image`, Body → **form-data**: `file` (File),
   `conf_threshold` (Text, `0.35`), `create_tickets` (Text, `false`).
   Let Postman set the multipart boundary.
4. Video uses `/api/detect/video`, additionally `max_frames=120`.
   Webcam frames use `/api/detect/frame`.

Unix curl example after assigning your token locally (do not commit it):

```bash
curl -f http://localhost:8000/api/health
curl -f http://localhost:8000/api/detect/image \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@/path/to/traffic.jpg' -F 'conf_threshold=0.35' -F 'create_tickets=false'
```

Responses include `job_id`, `detections` (`class_id`, `class_name`, `confidence`,
pixel `box.x1/y1/x2/y2`), counts, `summary` and a signed `result_url`.
Supported images: JPG/JPEG, PNG, BMP, WebP; videos: MP4, AVI, MOV, MKV, WebM,
subject to codecs. Empty/corrupt/oversized inputs are rejected. Defaults:
50 MB uploads, 12-megapixel frames, 120-second cooperative video time limit,
10 auth requests/minute and 60 detection requests/minute per peer IP.
One inference job is admitted at a time; overload returns 503 + Retry-After.

Video samples at least every second frame, widening the interval across a longer
clip when its frame count is known, with at most 300 processed frames. Counts are
**frame observations**, not unique offenders; automatic video ticket creation is
rejected. Output is converted to browser-compatible H.264. If conversion is
unavailable, an explicit warning accompanies a downloadable MPEG-4 fallback. Video
playback failure never hides the valid annotated still/detections. Use **Refresh
result links** to renew expired URLs without another inference job.
Webcam access requires localhost/HTTPS and browser permission. Frames persist as
jobs; use the retention command described in the deployment runbook.

## Validate the actual checkpoint

```bash
python -m scripts.verify_release --image /path/to/traffic.jpg --video /path/to/traffic.mp4
python -m inference.run_video /path/to/traffic.mp4 --output outputs/detected_video.mp4
```

Verification writes ignored `outputs/release-check/` annotations and a JSON report
including checkpoint SHA-256. Missing weights, failed inference, or missing image/
video examples return a nonzero exit. Visual comparison with the known-good backend
and held-out validation metrics are still required: a smoke test is not accuracy
validation. [Training guide](docs/TRAINING.md) · [Dataset layout](dataset/README.md)

## Deployment and tests

Use the [deployment runbook](docs/DEPLOYMENT.md) for the non-root Docker image,
Compose persistent volumes, Caddy HTTPS, account bootstrap, backup and retention.
The existing Render service was discovered through GitHub deployment metadata,
but these working-branch fixes have **not** been deployed to it. `render.yaml` is
a paid persistent-disk template, not a provisioned service; see the runbook.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/check_frontend.py  # requires Node.js
python -m ruff check backend api inference training scripts tests server.py --select F
# Optional real-browser checks (Unix; use $env:RUN_BROWSER_TESTS="1" in PowerShell):
python -m playwright install chromium
RUN_BROWSER_TESTS=1 python -m pytest tests/test_browser.py -q
```

With browser tests enabled: **99 passed, none skipped**. Chromium was installed via
an alternate package source after its normal download failed. Browser tests now use
a real HTTP server and cover images, H.264 video, fake webcam, PDF and result recovery.
One test runs actual YOLOv5 inference using an **untrained temporary checkpoint**;
this is not accuracy evidence. GitHub [application CI](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/actions/runs/35942775787) passed for commit `0be27e4`, including Docker/container and browser checks.
The fixes are published in [draft PR #1](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/pull/1); no main merge or deployment has occurred.
The [public-data pilot](docs/PUBLIC_DATA.md) has now trained and passed real-checkpoint Docker/browser smoke tests, but held-out mAP@0.5 is only **9.27%**. **Do not deploy it.** See [actual training results](docs/TRAINING_RESULTS.md). See the [investigation](docs/ANNOTATION_FIX.md).

```text
backend/app/      API, permissions, bounded inference, persistence, private media
frontend/         Same-origin web console and generated synthetic examples
api/server.py     Deployment entry point
models/           Supply trusted best.pt locally (ignored)
yolov5/           Vendored upstream YOLOv5 source/license
training/         Training helpers and experimental public-data pilot (not release-approved)
dataset/          Three-class layout/config; private dataset files ignored
deploy/           Caddy reverse-proxy configuration
scripts/          Environment setup, release check, JS check, local launcher
tests/            API/security/loader, optional browser and container smoke tests
```

This remains an internship prototype, **not a certified traffic-enforcement or
high-load production system**. TLS/access controls do not establish legal authority,
model accuracy or full security assurance. Review upstream [licensing notices](THIRD_PARTY_NOTICES.md)
and validate the actual environment/model before public deployment.
