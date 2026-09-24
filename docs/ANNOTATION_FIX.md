# Annotated-result investigation — 2026-09-24

## What is actually deployed

GitHub `main` currently points to `8159a35`. The deployment linked by GitHub is:
https://safecityai-t3ca.onrender.com

Its public `/api/health` was inspected and reported:

```json
{"model_loaded": true, "model_path": "/app/backend/weights/yolo11n.onnx", "model_name": "YOLO11n COCO ONNX fallback"}
```

This is not the custom YOLOv5 checkpoint. No `best.pt`, annotated training images,
labels or completed training metrics were found in the repository. A missing deployed
artifact does not establish whether the owner trained a model elsewhere.
Direct POST requests from this sandbox to Render failed TLS, so no successful live
upload test is claimed. The local tests below use the corrected working copy.

## Reproduced frontend defect

The GitHub HTML used one `showResultError()` handler for both detection and video
playback errors. That handler hid **both** the JPEG and video and replaced the
valid detections with an error. Thus an unsupported/broken video could erase a
successful annotation display.

A Chromium regression test against the fetched GitHub HTML reproduced the failure:
`#resultImg` became hidden after a video `error` event. The diagnostic adapts the
legacy cache-buster to the new test backend's signed-media contract; it does not
pretend to be the live Render backend. The same assertion passes on the fix.

## Changes

- Separate inference failures from image-fetch and video-playback failures.
- Preserve a valid JPEG and detections when video playback fails; offer download.
- Encode sampled video as H.264/yuv420p/faststart. Preserve a clearly marked MPEG-4
  download if conversion fails, rather than throwing away the successful annotations.
- Clear stale prior results before new inference and after a failed request.
- Refresh signed preview/video URLs from the existing job without rerunning inference.
- Display persistent model availability and actionable errors. Disable inference when
  no checkpoint is loaded; do not say a missing file proves a model was never trained.
- Add `/api/ready` (503 when unavailable), suitable for deployment health checks.
- Sample bounded frames across a clip when its frame count is known.

## End-to-end tests actually run

**77 tests passed with browser tests enabled; none skipped.** Chromium 153 runs over
real HTTP against an isolated Uvicorn server, not a mocked browser fetch layer.
Covered flows include:

- sign-in, image upload, actual annotated JPEG rendering, PDF download;
- failed second detection clearing stale results;
- image-link refresh and video-error recovery;
- video upload, real H.264 browser playback and refreshed preview links;
- fake webcam capture → frame API → annotated overlay;
- actual YOLOv5 inference → API → browser image, using a temporary **untrained**
  checkpoint (plumbing evidence, not trained-model accuracy evidence).

The remaining tests cover API permissions, media ownership, upload/resource limits,
reports, model integrity/provisioning and dataset validation. Training preflight
correctly rejects the empty repository dataset. Docker/Render execution is still
unverified in this sandbox; neither live deployment nor real-model accuracy is claimed.

## Training/deployment handoff

`training/train_yolov5.py` now validates labels/classes/images and split leakage,
trains, evaluates best.pt and optionally installs it. The unexecuted Colab notebook
uses the same pipeline. None of this substitutes for real annotated training data.
Managed deployments may provision the owner's trusted artifact using HTTPS
`MODEL_URL` plus mandatory `MODEL_SHA256`; no fallback URL is supplied.

The working branch's starting commit and the current GitHub main are separate root
histories. No force-push, reset, overwrite of main or automatic redeployment was
performed. These corrected files must be reviewed/published before they can change
the live application.
