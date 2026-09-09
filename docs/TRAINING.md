# Training custom SafeCityAI weights (Helmet / NoHelmet / Plate)

This product ships with **YOLO11n COCO ONNX** + a violation heuristic layer so demos work immediately.
For production helmet/seatbelt accuracy, fine-tune on your annotated traffic dataset.

## 1. Dataset

Classes recommended:

```
0 Helmet
1 NoHelmet
2 Seatbelt
3 NoSeatbelt
4 LicensePlate
5 Motorcycle
6 Car
7 Person
```

- Collect 200–500+ frames from CCTV / dashcam
- Annotate with [Roboflow](https://roboflow.com) or LabelImg
- Export **YOLOv8** format (images + `.txt` labels)
- Apply flip / brightness / mosaic augmentation

`data/traffic.yaml` example:

```yaml
path: /datasets/traffic
train: images/train
val: images/val
names:
  0: Helmet
  1: NoHelmet
  2: Seatbelt
  3: NoSeatbelt
  4: LicensePlate
  5: Motorcycle
  6: Car
  7: Person
```

## 2. Train (Ultralytics)

```bash
pip install ultralytics
yolo detect train model=yolo11n.pt data=traffic.yaml epochs=50 imgsz=640 batch=16
```

Monitor **mAP@0.5** in `runs/detect/train/results.csv`.

## 3. Export ONNX

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12
cp runs/detect/train/weights/best.onnx backend/weights/yolo11n.onnx
```

Update `backend/weights/coco.names` (or a new names file) to match your classes, and
adjust `PERSON` / `MOTORCYCLE` / violation maps in `backend/app/services/detector.py`.

## 4. Evaluate

```bash
yolo detect val model=best.pt data=traffic.yaml
# Video demo
yolo detect predict model=best.pt source=test_street.mp4 conf=0.5
```

## 5. API contract (unchanged)

```json
{
  "class_name": "NoHelmet",
  "confidence": 0.88,
  "box": { "x1": 100, "y1": 200, "x2": 150, "y2": 260 },
  "is_violation": true,
  "violation_type": "no_helmet"
}
```
