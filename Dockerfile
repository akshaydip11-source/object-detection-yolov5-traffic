FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY docs ./docs
# ONNX detector weights (models/best.onnx). The training checkpoint
# models/best.pt is kept in the repo for reference/export only and is NOT
# copied into the image.
COPY models/best.onnx ./models/best.onnx

ENV PYTHONPATH=/app/backend

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
