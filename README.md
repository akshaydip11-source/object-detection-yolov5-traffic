# SafeCityAI — YOLO Traffic Object Detection

AI-powered traffic-rule enforcement application built around the SafeCityAI internship case study.

## Project goal

Detect traffic objects and violations from images/video and expose the inference pipeline through a FastAPI API.
The internship target classes are **Helmet, NoHelmet, and LicensePlate**.

## Current application

The shipped UI/API is runnable immediately using the included COCO-pretrained YOLO11n ONNX model plus traffic-rule heuristics. This provides a working end-to-end demo while the custom YOLOv5 model is trained.

The custom-model path is documented under `dataset/` and `training/`. The final internship model should be a YOLOv5s/YOLOv5m model trained on the annotated traffic dataset and saved as `best.pt`.

## Features

- Traffic image detection console
- Video inference with annotated output
- Browser live webcam frame detection
- Violation dashboard and ticket workflow
- JWT authentication with demo accounts
- PDF challan and CSV export
- FastAPI + OpenAPI documentation
- Docker deployment configuration
- YOLOv5 training/inference scaffolding

## Quick start (Windows)

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python server.py
```

Open `http://localhost:8000`.

API docs: `http://localhost:8000/api/docs`

## Demo accounts

- `admin@safecity.ai` / `admin123`
- `officer@safecity.ai` / `officer123`
- `analyst@safecity.ai` / `analyst123`

## Custom YOLOv5 training

1. Annotate approximately 200–500 traffic images in YOLO format.
2. Put them under `dataset/images` and `dataset/labels`.
3. Use `dataset/traffic.yaml`.
4. Train YOLOv5s/YOLOv5m on Google Colab GPU.
5. Evaluate mAP@0.5 and inspect loss curves.
6. Save the resulting `best.pt` under `models/` locally; do not commit large/private weights unless required.
7. Connect the trained model to the detector configuration and retest image/video/API inference.

## Repository structure

```text
backend/       FastAPI application, database, services, model runtime
frontend/      SafeCityAI web console
api/           deployment API entry point
inference/     video inference helper
training/      YOLOv5 training scripts and notes
dataset/       YOLO dataset layout + traffic.yaml
models/        local trained weights (gitignored)
outputs/       generated inference outputs
Dockerfile     container deployment
server.py      local FastAPI entry point
```

## Internship deliverables

- Annotated dataset: `dataset/`
- Training materials: `training/`
- Video inference: `inference/` and application video console
- API code: `api/server.py` + `backend/app/`
- Final trained weights: `models/best.pt`

The original brief requires custom training and does not permit treating the pretrained fallback model as the final custom detector.
