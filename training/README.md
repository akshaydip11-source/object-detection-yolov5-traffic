# YOLOv5 Training

The internship requires transfer learning with YOLOv5s/YOLOv5m and evaluation using mAP@0.5.

1. Prepare `dataset/traffic.yaml` and YOLO-format labels.
2. Run `training/train_yolov5.py` on a CUDA GPU (Google Colab recommended).
3. The best weights are produced at `runs/safecity-yolov5/weights/best.pt`.
4. Copy the trained weights to `models/best.pt` for local experiments.
5. Record the loss curves and mAP@0.5 from the training results for the internship report.

No fabricated metrics are included; the notebook/results must be generated from the actual training run.
