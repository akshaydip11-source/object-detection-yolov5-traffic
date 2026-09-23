# SafeCityAI — YOLO Traffic Object Detection

AI-powered traffic-rule enforcement application built around the SafeCityAI internship case study.

## Project goal

Detect traffic objects and violations from images/video and expose the inference pipeline through a FastAPI API.
The internship target classes are **`Helmet`, `No_Helmet`, and `License_Plate`**.

## Current application

The shipped UI/API currently uses the included COCO-pretrained YOLO11n ONNX fallback. It does **not** satisfy the custom detector objective yet. The website now identifies the fallback honestly. The repository has no annotated dataset, trained YOLOv5 checkpoint, validation metrics, or custom detection demo video.

The working YOLOv5s pipeline, Colab notebook, validation, video inference, and ONNX export steps are in `training/` and `docs/TRAINING.md`. Training cannot run until the annotated dataset has been added.

## Features

- Traffic image detection console
- Video inference with annotated output
- Browser live webcam frame detection
- Violation dashboard and ticket workflow
- JWT authentication with demo accounts
- PDF challan and CSV export
- FastAPI + OpenAPI documentation
- Docker deployment configuration
- YOLOv5s training, validation, video inference, and ONNX export workflow

## Quick start (Windows)

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python server.py
```

Open `http://localhost:8000`.

API docs: `http://localhost:8000/api/docs`

## Local demo accounts

- `admin@safecity.ai` / `admin123`
- `officer@safecity.ai` / `officer123`
- `analyst@safecity.ai` / `analyst123`

These demo credentials are seeded only in development mode. Render generates a separate administrator password for the production deployment, and new registrations receive the officer role.

## Deploy on Render

1. In Render, choose **New → Blueprint** and connect this GitHub repository.
2. Render reads [`render.yaml`](render.yaml), builds the Docker image, and checks `/api/health`.
3. After the first deploy, retrieve the generated `DEFAULT_ADMIN_PASSWORD` from the service environment settings and sign in as `admin@safecity.ai`. Keep that password private; the app does not currently provide password changes.

The Blueprint uses a Free web service for demonstration. Its SQLite database and uploaded files use the local filesystem, which Render does not preserve on Free services. Do not use this setup for durable records. A production deployment needs persistent storage and a database configuration designed for PostgreSQL or another managed database.

## Custom YOLOv5 training

1. Annotate 200–500+ traffic images in YOLO format with class IDs from `dataset/traffic.yaml`.
2. Split by source video into `dataset/images/{train,val}` and `dataset/labels/{train,val}`.
3. Run `python training/train_yolov5.py --validate-only`, then `python training/train_yolov5.py --epochs 50 --batch 16` on a CUDA GPU.
4. Review the generated training losses, per-class metrics, confusion matrix, and held-out video output. Do not claim mAP without the real validation results.
5. The script exports `backend/weights/yolov5_custom.onnx` and `backend/weights/traffic.names`. Set `MODEL_PATH=weights/yolov5_custom.onnx` and `CLASS_NAMES_PATH=weights/traffic.names` to use them with the API.

See [`training/safecityai_yolov5_training.ipynb`](training/safecityai_yolov5_training.ipynb) for the full Colab workflow. The actual dataset, trained weights, metrics, and demo video still need to be supplied/generated; this repository does not contain them.

## Repository structure

```text
backend/       FastAPI application, database, services, model runtime
frontend/      SafeCityAI web console
api/           deployment API entry point
inference/     video inference helper
training/      YOLOv5 training script, Colab notebook, and notes
dataset/       YOLO dataset layout + traffic.yaml
models/        local trained weights (gitignored)
outputs/       generated inference outputs
Dockerfile     container deployment
server.py      local FastAPI entry point
```

## Internship deliverables

- Annotated dataset: `dataset/` layout is ready; annotated files are not present yet
- Training notebook and pipeline: `training/`
- Video inference: `inference/` and application video console; a custom-model demo video is not present yet
- API code: `api/server.py` + `backend/app/`
- Final trained weights: not present; train after adding the annotated dataset

The original brief requires custom training and does not permit treating the pretrained fallback model as the final custom detector.
