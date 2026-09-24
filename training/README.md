# Custom YOLOv5 training

See [the full training guide](../docs/TRAINING.md) and the unexecuted
[`safecityai_yolov5_training.ipynb`](safecityai_yolov5_training.ipynb) Colab notebook.

```bash
python training/train_yolov5.py --validate-only
python training/train_yolov5.py --epochs 50 --batch 16 --device 0 --install-model
```

Supply real licensed labels, or use the reviewed public import described in
[Public data](../docs/PUBLIC_DATA.md). The notebook's working branch is now published
in draft PR #1. The notebook itself remains unexecuted and can select the public
pilot or your private Drive dataset; it does not deploy/install automatically.

The CPU pilot **did train**, and its checkpoint passed Docker/browser integration,
but its accuracy is inadequate for release. See [actual measured results](../docs/TRAINING_RESULTS.md).
Training completion, an untrained test fixture and production model approval are
three different things. Never substitute either an untrained fixture or the failed
accuracy pilot for an approved best.pt.
