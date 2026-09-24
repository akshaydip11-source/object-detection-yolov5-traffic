FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 YOLOv5_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/Ultralytics XDG_CACHE_HOME=/tmp/cache MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CPU image; no CUDA libraries or remote model downloads at startup.
RUN pip install --no-cache-dir 'torch>=2.6,<3' 'torchvision>=0.21,<1' \
      --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY frontend ./frontend
COPY yolov5 ./yolov5
COPY api ./api
COPY server.py .
COPY scripts ./scripts
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models /app/backend/uploads /app/data \
    && chown -R appuser:appuser /app
ENV DATABASE_URL=sqlite:////app/data/safecity.db
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/api/ready',timeout=4)"
CMD ["python", "server.py"]
