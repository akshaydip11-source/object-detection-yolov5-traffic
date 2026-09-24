# Custom YOLOv5 training

See [the full training guide](../docs/TRAINING.md) and the unexecuted
[`safecityai_yolov5_training.ipynb`](safecityai_yolov5_training.ipynb) Colab notebook.

```bash
python training/train_yolov5.py --validate-only
python training/train_yolov5.py --epochs 50 --batch 16 --device 0 --install-model
```

Supply real labeled data first. Validation, training and best-checkpoint evaluation
are separate steps; failures stop the pipeline. No trained model or fabricated
metrics are included. The notebook's working branch must be published before it
can be cloned in Colab. Never substitute the random test fixtures for best.pt.
