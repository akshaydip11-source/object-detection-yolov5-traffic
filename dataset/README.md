# SafeCityAI Dataset

Place the annotated YOLO-format dataset here.

Classes (exact names and IDs):
- 0: Helmet
- 1: No_Helmet
- 2: License_Plate

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

The internship brief targets roughly 200–500 traffic images with bounding-box annotations. Keep all frames from the same source video in a single split to avoid train/validation leakage. This repository currently contains no dataset images or labels; collect and annotate them before training.
