# Custom YOLOv5 training, evaluation and installation

The dataset and trained checkpoint are not in this repository. The pipeline and
unexecuted [Colab notebook](../training/safecityai_yolov5_training.ipynb) are ready for
actual data; they are not evidence of completed training or acceptable accuracy.

## Dataset

Expected label IDs: **0 Helmet, 1 NoHelmet, 2 LicensePlate**. `No_Helmet` and
`License_Plate` are accepted name spellings, but IDs must still match their labels.
Use `images/{train,val}` and parallel `labels/{train,val}` directories with normalized
YOLO boxes (`class_id center_x center_y width height`). Empty TXT files are valid
negative examples. Every class needs examples in both train and validation.

Split by recording/scene before extracting frames; also retain unseen test data.
Confirm privacy and image/annotation licensing before using or sharing the data.
`dataset/traffic.yaml` uses `path: ../dataset`, resolved relative to the vendored
YOLOv5 root. For an external dataset, supply `--data /path/to/config.yaml` and use an
absolute `path:`. Keep all private dataset files out of Git.

## Validate before training

```bash
python training/train_yolov5.py --validate-only
# Or an external labeled dataset:
python training/train_yolov5.py --data /path/to/traffic.yaml --validate-only
```

Validation rejects missing/corrupt images, missing/orphan labels, invalid/out-of-bounds
boxes, missing classes and byte-identical train/validation leakage. It reports counts
and a dataset fingerprint. Near-duplicate frames still require a proper source split;
byte checks cannot prove absence of all leakage.

## Train and evaluate (GPU recommended)

```bash
python training/train_yolov5.py --epochs 50 --batch 16 --device 0 --install-model
```

Use `--device cpu` only if you accept a slower run; no GPU speed is claimed. The
script starts from YOLOv5s pretrained initialization (`--weights` is configurable),
trains on your labels, then runs `yolov5/val.py` on the resulting best checkpoint.
Unique output folders under `runs/train/safecity-*` and `runs/val/safecity-*` contain
the actual CSV, plots, checkpoint and validation outputs. `training_report.json`
records data counts/fingerprint, commands, environment versions and output paths.

`--install-model` copies the evaluated checkpoint to `models/best.pt`. It refuses to
replace an existing model unless `--overwrite-model` is explicitly supplied. Back up
old weights before replacing them. A successful run does **not** prove sufficient
accuracy: inspect actual mAP@0.5, mAP@0.5:0.95, per-class precision/recall, losses and
confusion matrices against the internship requirements. Do not invent metrics.

## Validate the API handoff

```bash
python -m scripts.verify_release --image /path/to/heldout.jpg --video /path/to/heldout.mp4
```

Inspect the ignored `outputs/release-check/` annotations/report, record SHA-256 and
compare outputs against the known-good Postman-tested backend. Test all three
classes, negative cases, multiple objects and different aspect ratios. Restart the
API after replacing its model; then verify health/readiness, sign-in, image/video/
webcam, signed media and reports in the browser.

**No ONNX export is needed for this backend.** Do not use a YOLOv8/11 export command
as a substitute for YOLOv5 training. Share the trusted best.pt separately, not through
Git; managed hosts may use the explicit HTTPS MODEL_URL + MODEL_SHA256 mechanism.

## Public-data pilot

See [Public data assessment](PUBLIC_DATA.md) for the pinned CC BY 4.0 source,
explicit label mapping, provenance and the bounded experimental CPU workflow.
Do not substitute that pilot for independent field validation or install its
checkpoint automatically. Generated training artifacts are excluded from Git.

The initial CPU pilot and validation-only resolution review have completed.
[Measured results](TRAINING_RESULTS.md) show insufficient class-specific accuracy,
so no model is approved for deployment. The notebook is still unexecuted; its
training results must not be confused with the recorded CPU experiment.
