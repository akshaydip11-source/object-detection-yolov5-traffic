# Public training-data assessment — 2026-09-24

## Selected for a bounded pilot, not deployment approval

**MotorbikeDelivery_2.0 version 8**, published by **MotorbikeDelivery** on
[Roboflow](https://universe.roboflow.com/motorbikedelivery/motorbikedelivery_2.0/dataset/8).
The [RASYD project's dataset metadata](https://github.com/NasserAlsaqer/Detect-Violated-Motorbikes-Delivery-RASYD/blob/80169b657bd537c2f021f23399bc45e8d2e552d2/Dataset/data.yaml)
declares **CC BY 4.0** and these six classes:
`Helmet`, `License_plate`, `MotorbikeDelivery`, `MotorbikeSport`, `No_Helmet`, `Person`.
The project's README describes manual annotation and links back to this source.
This is a publisher-declared license, not a guarantee that every underlying photo's
rights have been independently established. Preserve attribution, license link and
modification notice. Do not identify people or use model predictions as legal findings.

The public GitHub mirror's ZIP is an LFS object pinned to commit
`80169b657bd537c2f021f23399bc45e8d2e552d2`, size **502,502,186 bytes**, SHA256:

```text
dab329daaa7213a3307b3677dab82e6024ffe41038870e7d970d7fa07fe9bb9d
```

The archive has not been replaced by someone else's pretrained checkpoint. Its
included/exported model files, if any, are not used. The pilot starts from the
**official Ultralytics YOLOv5n** backbone and fine-tunes new custom weights.

### Import and evaluation policy

`python -m training.public_dataset --output runs/public-pilot-data`:

1. Download the pinned export over HTTPS; verify exact length and SHA256.
2. Require exact reviewed class order and CC BY 4.0 metadata inside the ZIP.
3. Remap only genuine annotations: Helmet 0→0, No_Helmet 4→1, License_plate 1→2.
   Drop the other classes. **Never map Person/Motorbike/Rider to NoHelmet.**
4. Keep published train/validation/test assignments. Reject repeated original
   filenames across splits; retain one augmentation per named source image.
5. Deterministically select at most 320 training, 64 validation and 64 test images,
   preferentially including up to eight images of each class in each split.
   This deliberately changes prevalence: reported metrics describe this pilot subset,
   not an unbiased estimate of source/population prevalence.
6. Reject bad boxes, missing/corrupt images, missing classes and byte-identical
   duplicate selected images. Preserve a per-image source/hash manifest and attribution.

Filename grouping and exact hashes **cannot establish independent cameras/videos**.
Near-duplicate scenes, incomplete annotations and India-specific domain shift require
further review. No public-source benchmark score is presented as our model's score.

`python -m training.public_pilot` runs 20 CPU epochs at 320px, validates the selected
best checkpoint, evaluates the held-out test split, and records aggregate metrics,
per-class mAP50–95 and annotated validation batches. It does **not** install the
checkpoint in `models/`, change Render, or approve operational use. An image-size
option is now available in the general training CLI; its normal default remains 640px.

The **Public dataset pilot** GitHub workflow runs only on this session branch when
its own workflow/import/pilot files change. It is bounded to 40 minutes, has read-only
repository permissions, uses a standard CPU runner and retains experimental artifacts
for 14 days. No GPU or paid service is provisioned. Downloads/results stay outside Git.
Local direct downloads to the public media host fail TLS in this sandbox, so the
first real archive audit/training is performed by that workflow. **At this document's
initial publication, that run is pending; no trained-model accuracy is claimed.**

## Other candidates checked

| Candidate | Decision |
| --- | --- |
| [Object Detection HelmetsLicense](https://universe.roboflow.com/object-detection-helmetslicense/motorcycle-helmet-and-license-plate-detection) | Page lists helmet, plate, rider, not a genuine NoHelmet class. Reject for the complete three-class task. |
| [HCMUTE mirror](https://github.com/Khoa-CNTT/UDCNANCX5134/blob/main/Helmet-Detection-and-License-Plate-Recognition-4/data.yaml) | CC BY 4.0 export, but helmet/licenseplate/motorcyclist only. Motorcyclist must not become NoHelmet. |
| [on Gia Capital](https://universe.roboflow.com/on-gia-capital/helmet-detection-and-license-plate-recognition/dataset/1) | Promising four-class candidate, but download route requires a Roboflow account. No login gate or API credentials bypassed. |
| [HelmetViolations on Kaggle](https://www.kaggle.com/datasets/pkdarabi/helmet) | Public page declares CC BY 4.0 and Plate/WithHelmet/WithoutHelmet. However the [available mirror](https://github.com/rishikhngowdaaiml-code/Helmet-Dataset/blob/main/HelmetViolationsV2/data.yaml) says `license: Private`. This may be historic export metadata, but redistribution authority is not sufficiently clear here; leave out pending clarification. |
| [smart-traffic-monitor](https://github.com/rumbleFTW/smart-traffic-monitor) | Has images/labels but a research-only, no-redistribution license; unsuitable as the default deployment data source. |

## Release remains blocked until

- Actual source audit/training/evaluation completes successfully.
- Annotated held-out predictions and per-class errors are reviewed.
- Independent, representative target-camera data establishes acceptable performance;
  tiny validation/test subsets cannot justify enforcement claims.
- Artifact provenance and the applicable YOLOv5/data redistribution obligations are
  reviewed; a trusted checkpoint is provisioned and tested on the real host.
