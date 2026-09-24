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

The internship brief targets roughly 200–500 traffic images with bounding-box annotations. Keep all frames from the same source video, including augmented variants, in a single split to avoid train/validation leakage. This repository contains no dataset images or labels. Use the validated training pipeline described in [Training](../docs/TRAINING.md); public dataset licensing and label coverage must be checked before importing data.
