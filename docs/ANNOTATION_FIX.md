# Annotated-result investigation — 2026-09-24

## What is actually deployed

The initial investigation inspected GitHub `main` at `8159a35`. Main subsequently
advanced to `6f472d6` (removal of false fallback violations), also incorporated into
this working branch without changing main. The deployment linked by GitHub is:
https://safecityai-t3ca.onrender.com

Its public `/api/health` was inspected and reported:

```json
{"model_loaded": true, "model_path": "/app/backend/weights/yolo11n.onnx", "model_name": "YOLO11n COCO ONNX fallback"}
```

This is not the custom YOLOv5 checkpoint. No original `best.pt`, annotated training images,
labels or completed training metrics were found during the initial repository audit.
A new experimental public-data model has since been trained; its measured accuracy
is insufficient for deployment (see the results below). A missing deployed
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

**99 tests passed with browser tests enabled; none skipped.** Chromium 153 runs over
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
correctly rejects the empty repository dataset. GitHub application CI at `0be27e4` passed the Docker build, actual container smoke
and browser checks. Local Docker remains unavailable. Live deployment is not claimed. A subsequent public-data training run measured
poor accuracy and passed trained-checkpoint integration checks; see
[the actual results](TRAINING_RESULTS.md). The added public-data tests use generated fixtures,
not actual training images.

## Training/deployment handoff

`training/train_yolov5.py` now validates labels/classes/images and split leakage,
trains, evaluates best.pt and optionally installs it. The unexecuted Colab notebook
uses the same pipeline. None of this substitutes for real annotated training data.
Managed deployments may provision the owner's trusted artifact using HTTPS
`MODEL_URL` plus mandatory `MODEL_SHA256`; no fallback URL is supplied.

The starting checkout and GitHub main had separate root histories. Both were
preserved through merge commits on the working branch, along with the newer
`6f472d6` safety fix. [Draft PR #1](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/pull/1)
is published; [application CI passed](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/actions/runs/35942775787).
No force-push, overwrite/merge of main, paid provisioning or automatic deployment
was performed. Public-data selection and the bounded training pilot are documented
in [PUBLIC_DATA.md](PUBLIC_DATA.md). The pilot completed and passed real-model Docker
and Chromium checks, but initial held-out mAP@0.5 is only **9.27%**.
[Training results](TRAINING_RESULTS.md) explain why the PR remains draft and the
experimental checkpoint must not be installed as a production detector.
