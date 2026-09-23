# SafeCityAI YOLOv5 Custom Training

The target detector is YOLOv5s fine-tuned from COCO-pretrained `yolov5s.pt` on exactly these classes:

| Class ID | Name |
| --- | --- |
| 0 | `Helmet` |
| 1 | `No_Helmet` |
| 2 | `License_Plate` |

The repository currently contains no labeled training images, no custom `best.pt`, and no custom demo video. The deployed model is still the YOLO11n COCO fallback; heuristic violation flags are not evidence of a trained helmet detector. Provide the annotated dataset before claiming custom-model metrics or results.

## Dataset layout

Create a YOLO-format dataset with matching image and label paths:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── traffic.yaml
```

Each `.txt` file has one normalized row per box: `class_id x_center y_center width height`. Image and label base names must match. Keep video sources and extracted frames split by source video: do not put frames from the same recording in both train and validation, or validation mAP will be misleading. Annotate helmet and plate boxes consistently; record occluded or ambiguous examples according to one policy.

The case-study target is 200–500 annotated images. That is a starting target, not a guarantee of generalization. Add diverse scenes and inspect per-class precision/recall before using the model for enforcement.

## Validate and train

Use a CUDA GPU (Google Colab is suitable). Install the YOLOv5 requirements, then from the project root run:

```bash
python training/train_yolov5.py --validate-only
python training/train_yolov5.py --epochs 50 --batch 16
```

The script validates image/label pairs, class IDs, normalized boxes, and presence of every target class in train and validation before training. It fine-tunes `yolov5s.pt` with YOLOv5's `hyp.scratch-low.yaml`, evaluates `best.pt`, writes the standard `results.csv`/plots, and exports ONNX. Adjust `--batch` if the GPU runs out of memory. The notebook [`training/safecityai_yolov5_training.ipynb`](../training/safecityai_yolov5_training.ipynb) contains the Colab workflow, result plots, and video inference steps.

Review `runs/train/<run>/results.png`, the loss curves, confusion matrix, and `runs/val/<run>/results.txt`. Record per-class precision, recall, mAP@0.5, and mAP@0.5:0.95. Do not report results until a validation run finishes on data kept separate from training.

## Video inference

Run on a held-out video after training:

```bash
python yolov5/detect.py --weights runs/train/<run>/weights/best.pt \
  --source path/to/held-out-video.mp4 --img 640 --conf-thres 0.5 \
  --save-txt --save-conf --project outputs --name safecityai-video
```

The annotated video and frame labels are written under `outputs/safecityai-video/`. Review the result manually for false positives, missed small helmets, and plate readability.

## Run the custom model through the API

Training exports `backend/weights/yolov5_custom.onnx` and writes the matching `backend/weights/traffic.names`. For local use, set:

```text
MODEL_PATH=weights/yolov5_custom.onnx
CLASS_NAMES_PATH=weights/traffic.names
```

For Render, commit the custom ONNX and names files with the app source (or use an artifact download step) and set those same variables in the service environment. The Dockerfile copies `backend/weights` into `/app/backend/weights`. Redeploy and confirm `/api/health` reports the custom YOLOv5 model before testing `/api/detect/image` and `/api/detect/video`.

The upload API returns each detection with `class_id`, `class_name`, `confidence`, and pixel-corner `box` coordinates, plus a summary and annotated media URLs. `No_Helmet` detections create a helmet-violation flag; helmet classification and license-plate detection remain model outputs, not legal determinations.
