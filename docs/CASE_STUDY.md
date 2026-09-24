# SafeCityAI — alignment with the supplied case-study brief

## Scope and current status

The required annotation classes are **Helmet, No_Helmet, License_Plate**. The
opening overview also mentions seatbelts, but the execution plan does not supply
seatbelt classes or data. No seatbelt detections, OCR, rider identity or automatic
legal conclusions are invented.

The supplied screenshot shows the old YOLO11n/ONNX console's **idle** placeholders;
it does not establish a failed completed inference. A fresh public health check
still reports `YOLO11n COCO ONNX fallback`. The draft branch is not deployed there.
The original JPEG/video error recovery defect is separately reproduced and fixed.

| Required deliverable | Implementation | Completion boundary |
| --- | --- | --- |
| Annotated dataset ZIP/link | Licensed public source, explicit class mapping, split/group audit, `training/export_dataset.py`, dataset-export CI artifact | Reuses published human annotations; does not claim we personally drew them. Review labels for your submission and target cameras. |
| YOLOv5s/m training notebook and charts | Published Colab notebook; YOLOv5s COCO initialization, 640px, proposed 100 epochs, seed, patience, explicit `training/hyps/traffic.yaml`, actual CSV/chart display cells | Notebook remains **unexecuted**. YOLOv5s weights and charts must come from that completed run; the weak YOLOv5n pilot is not a substitute. |
| 30-second street demo | `python -m inference.make_demo`, continuous source window, learned boxes, H.264 output, measured offline throughput, credit/hash report | **Real footage has not been supplied.** Synthetic unit tests are not the demo. Requires a suitable trained checkpoint and qualitative review. |
| FastAPI server and image POST JSON | `server.py`, authenticated `/api/detect/predict`, client example script, regression tests | Implemented; actual model accuracy remains a separate release gate. |

The earlier 20-epoch **YOLOv5n** pilot was only an integration experiment. Its initial
held-out mAP@0.5 was **9.27%**, and resolution diagnostics exposed poor helmet-class
performance. It must not be renamed as the final YOLOv5s/m model or deployed simply
to remove a warning. See [actual measured results](TRAINING_RESULTS.md).

## 1. Dataset handover

The [public source and licensing assessment](PUBLIC_DATA.md) records the source
URL, publisher-declared CC BY 4.0 license, pinned ZIP SHA256, conversion policy,
original-group exclusions and remaining near-duplicate/annotation-quality limitations.

```bash
python -m training.public_dataset --output runs/case-study-source \
  --polygon-boxes --quarantine-invalid
python -m training.export_dataset \
  --data runs/case-study-source/data.yaml --output runs/case-study-dataset
```

Produces `runs/case-study-dataset.zip` containing images, YOLO TXT labels, portable
`data.yaml`, attribution, original-source provenance, and the augmentation manifest.
The baseline selects **448 originals**: 320 train, 64 validation, 64 test. Training
original + horizontal flip + brightness +25% yields **960 training images**;
validation/test remain 64 each, **1,088 total files of images**. This is not 1,088
independent observations. Flip coordinates are transformed, not copied unchanged.
Whole originals and all their variants remain in one split; no hold-out augmentation.

Use the project training wrapper after relocating the ZIP. It writes an absolute,
validated runtime YAML before invoking vendored YOLOv5, avoiding root-relative path
ambiguity and forwarding no YAML download scripts. Checkpoints/data/ZIPs stay out of Git.
The **Case-study annotated dataset** Actions workflow exports a downloadable ZIP;
its artifact retention is seven days, not permanent public hosting.

**Generated and validated on 24 September 2026:**
[successful export run](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/actions/runs/35946271955)
and [downloadable annotated-dataset artifact](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/actions/runs/35946271955/artifacts/10786861674)
(~121.7 MB; expires **1 October 2026**). Its runner annotation confirms 960/64/64
exported images and train box counts **537 Helmet / 246 No_Helmet / 300 License_Plate**.
Inner `case-study-dataset.zip` SHA256:
`832cea9007b2ec8ed605d2ad92f3cb5483cee869df3e4397f08decece5908ced`.
This archive's existence is verified remotely; it was not downloaded into the Git
working tree. Keep a local copy before expiry or regenerate with the commands above.

## 2. Training configuration and actual charts

Open `training/safecityai_yolov5_training.ipynb` in Colab, enable an available GPU,
and run the cells. The public route prepares and exports the train-only augmented
set. The notebook itself remains unexecuted in this repository—no fabricated chart
outputs or metrics are included. The equivalent training command is:

```bash
python training/train_yolov5.py --data runs/case-study-dataset/data.yaml \
  --weights yolov5s.pt --img-size 640 --epochs 100 --batch 16 --device 0 \
  --hyp training/hyps/traffic.yaml --seed 0 --patience 50
```

`yolov5m.pt` is an alternative if GPU memory/latency permits. The proposed settings
are not claimed to be optimized. The hyp file retains upstream attribution, uses
mosaic, brightness/HSV augmentation and SGD settings, and disables a second online
horizontal flip because the exported training set already includes that variant.

The wrapper evaluates best.pt and records the source/runtime configs, command,
package versions, `results.csv`, `results.png` and `hyp.yaml`. Notebook cells draw
**Loss vs. Epochs and validation mAP@0.5 from that actual run's CSV**. Do not substitute
the dataset publisher's model scores or the old nano pilot's charts. Do not tune
on the already-inspected pilot test split or call it a new independent final test.
The notebook does not deploy or automatically install experimental weights.

## 3. Actual street-video demo

Supply a representative street video of at least 30 seconds and a text file naming
its source, ownership/permission or reuse license. It should be independent of the
training scenes. Uploading the screenshot does not provide this video.

```bash
python -m inference.make_demo --source /path/to/street.mp4 \
  --weights /path/to/trusted/best.pt --source-license /path/to/footage-license.txt \
  --start 0 --conf 0.50 --output outputs/street-demo
```

Outputs `demo-30s.mp4`, a JPEG preview, `demo_report.json` and `FOOTAGE_LICENSE.txt`.
The report records the model SHA256, threshold, processed frames and measured offline
throughput. A 10fps encoded video does **not** establish real-time inference. The CLI
rejects short/incomplete clips rather than looping or padding them into a fake demo.
Its separate, bounded offline time budget (up to 600 seconds) does not increase HTTP API limits.
On an available GPU, set `DEVICE=0`; the report records the selected device. It does
not certify scene content or legal violations; inspect predicted boxes manually.

## 4. Brief-compatible API

```bash
python server.py
# Obtain a private access token through POST /api/auth/login, then set it locally:
# SAFECITY_TOKEN=<your token>  (never commit or paste credentials into chat)
python -m scripts.test_case_study_api --base-url http://127.0.0.1:8000 \
  --image /path/to/traffic.jpg
```

`POST /api/detect/predict` accepts multipart `file` and `conf_threshold` (default
0.50), with `Authorization: Bearer ...`. Illustrative JSON shape, **not a measured
prediction**:

```json
{
  "job_id": "...",
  "box_format": "xywh",
  "coordinate_system": "image_pixels",
  "detections": [
    {"class": "No_Helmet", "confidence": 0.88, "box": [100, 200, 50, 60]}
  ],
  "annotated_image_url": "/api/media/..."
}
```

`box` is **left, top, width, height in original-image pixels**, not normalized YOLO
training coordinates or x1/y1/x2/y2. The existing console endpoint `/api/detect/image`
retains its corner-coordinate contract. Both share auth, upload limits, bounded
inference, job ownership and signed media. This compatibility route creates no tickets.
Unavailable weights return 503; no COCO fallback or fabricated annotation is used.

## Remaining release work

Complete the YOLOv5s/m training, review per-class errors and real held-out video,
choose a justified threshold using validation, and provide an acceptable trusted
checkpoint. Only then approve production configuration and explicitly authorize
merge/deployment. [Draft PR #1](https://github.com/akshaydip11-source/object-detection-yolov5-traffic/pull/1)
stays unmerged; the live service is not silently changed.
