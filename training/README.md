# YOLOv5 Training

Use [`safecityai_yolov5_training.ipynb`](safecityai_yolov5_training.ipynb) in Google Colab, or run the same pipeline from the project root:

```bash
python training/train_yolov5.py --validate-only
python training/train_yolov5.py --epochs 50 --batch 16
```

The target classes are `Helmet`, `No_Helmet`, and `License_Plate`. The validator requires a non-empty YOLO-format train and validation split, matching image/label files, normalized box values, and examples of every class in both splits. Training fine-tunes YOLOv5s from COCO weights, evaluates the saved best checkpoint, records standard YOLOv5 loss/mAP plots, and exports `backend/weights/yolov5_custom.onnx` plus `backend/weights/traffic.names`.

This repo has no annotated images or custom checkpoint, so it is not possible to produce real training curves, mAP metrics, or a trained demo video until the dataset is supplied. Do not report fabricated values.
