# Actual training and end-to-end results — 2026-09-24

## Verdict: integration works; the model is not accurate enough to release

A **new custom YOLOv5 model was trained**, not merely a renamed COCO detector or an
untrained fixture. It passed real Docker and browser integration checks. However,
its detection accuracy is poor. **Do not deploy this experimental checkpoint.**
[Draft PR #1](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/pull/1)
remains open and unmerged. The live Render service has not been changed by this work.

## Reproducible evidence

- [Application CI](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/actions/runs/35942775787), commit `0be27e4`: passed, including Docker and browser checks.
- [Actual training/evaluation/integration run](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/actions/runs/35942772831): passed execution in 10m15s. Green execution is **not** an accuracy approval.
- [Validation-resolution review](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/actions/runs/35943981706): passed; reviewed the recorded learning curves and reused validation only.
- Local complete suite: **99 passed, no skips**, including nine real Chromium→HTTP→Uvicorn tests. One upstream Starlette deprecation warning remains.
- The pilot separately passed its **trained checkpoint** through a non-root/read-only production Docker container (authentication, image inference, signed media), and through an actual HTTP upload→YOLOv5→JPEG→Chromium flow.

The public dataset is documented in [PUBLIC_DATA.md](PUBLIC_DATA.md): publisher-declared
CC BY 4.0, pinned ZIP hash, explicit genuine class mapping, polygon-to-box conversion,
and exclusion of entire conflicting original-image groups from every split.
No generic person/motorbike label was turned into NoHelmet.

## Data actually used

| Split | Images | Helmet boxes | NoHelmet boxes | LicensePlate boxes |
| --- | ---: | ---: | ---: | ---: |
| Train | 320 | 179 | 82 | 100 |
| Validation | 64 | 48 | 13 | 12 |
| Test | 64 | 52 | 8 | 28 |

- **31 original-image groups quarantined**, below the unchanged 5% cap.
- Genuine license-plate polygons converted to enclosing boxes: 28 train, 3 validation, 7 test.
- Dataset fingerprint: `40d325c83f1dc65b2d4ce0e0fa3bf6800180ad05a7ff908a89c0641aa0beb4cd`.
- Training: official YOLOv5n initialization, **20 recorded CPU epochs at 320px**.
- Main evaluation: **640px**, matching the current application's input size.

## Measured accuracy — all values below are percentages

| Evaluation | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| Validation, 640px | 9.68 | 8.33 | 7.97 | 3.05 |
| Initial held-out test, 640px | 12.83 | 12.51 | **9.27** | **3.54** |
| Validation-only diagnostic, 320px | 26.19 | 16.65 | 16.40 | 11.88 |

Precision/recall follow YOLOv5's validation summary convention; they are not a
calibrated measurement at the application's fixed operating confidence threshold.
These are this run's actual results, not the dataset publisher's hosted-model scores.

Per-class **mAP@0.5:0.95** exposes why changing runtime resolution is not a fix:

| Class | Validation 640px | Validation 320px | Initial test 640px |
| --- | ---: | ---: | ---: |
| Helmet | 7.57 | 1.10 | 5.71 |
| NoHelmet | 1.58 | 0.54 | 2.96 |
| LicensePlate | 0.006 | 34.01 | 1.94 |

At 320px the plate score improves while the two helmet classes get worse. Changing
the app's resolution solely to raise the aggregate score would conceal a serious
class-specific failure, so the runtime default was **not** changed.

The recorded losses decreased (box 0.12279→0.044125, classification
0.040531→0.010565); the best training-time validation mAP@0.5 was only 17.186%.
Learning occurred, but the short, small-data pilot does not establish a useful detector.
A stronger training schedule, sufficient small-head examples, full-resolution evaluation,
and qualitative review are needed. More epochs alone are not guaranteed to solve it.

The initial test split was not used for fitting/checkpoint selection in this run.
It has now been inspected. Future tuning must use validation; reusing this same
small test split cannot be advertised as a new independent final test. Only eight
NoHelmet test boxes and unverified camera/video independence are major limitations.

## Experimental artifact — review only

[Download the GitHub Actions artifact](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/actions/runs/35942772831/artifacts/10786110997)
(named `public-dataset-pilot`, retained for 14 days). It includes:

- `public-pilot-results/experimental_best.pt`;
- actual training metrics, evaluation report, provenance and attribution;
- validation/test plots and annotated prediction batches.

Checkpoint SHA256:

```text
4db70c1d0b4b577a5ab66e5084b612769c33b4ea1e60bf14ffd53d2e507d35c3
```

The sandbox cannot download the Actions artifact from its storage host. Its metadata,
structured CI annotations and executed integration checks were inspected; **no local
visual inspection of the prediction plots is claimed**. The artifact can be downloaded
through GitHub for review. It is not committed into Git or installed as `models/best.pt`.
Do not use an expiring/authenticated Actions download URL as a production `MODEL_URL`.

## Still required

1. Longer, appropriately sized training and more representative annotations; use
   the published, unexecuted GPU Colab notebook or an explicitly bounded longer CPU experiment.
2. Review held-out image/video annotations and per-class false positives/negatives.
3. Validate against independent target-camera data before any enforcement use.
4. Only after acceptable results: approve a trusted artifact, preserve attribution,
   configure its SHA256 and production accounts/secrets, then explicitly authorize
   merge/deployment and repeat real-host smoke checks.
