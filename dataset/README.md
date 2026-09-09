# SafeCityAI Dataset

Place the annotated YOLO-format dataset here.

Classes:
- 0: Helmet
- 1: NoHelmet
- 2: LicensePlate

Expected structure:

```text
dataset/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
└── traffic.yaml
```

Each label file uses normalized YOLO coordinates:
`class_id center_x center_y width height`

The internship brief requires roughly 200–500 traffic images with bounding-box annotations.
