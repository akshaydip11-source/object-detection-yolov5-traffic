# Synthetic upload examples

The three JPEG files here are rendered by `scripts/generate_samples.py` using only
OpenCV drawing primitives and NumPy noise. No stock photographs, downloaded images
or identifiable people are used. Reproduce with:

```
python scripts/generate_samples.py
```

These are UI/upload illustrations only. Their filenames do not establish supported
violation classes. They are not training or validation evidence for the custom model.
Previously supplied photographs with incomplete provenance were removed from the
current frontend (not rewritten out of Git history). Vendored upstream examples
remain under `yolov5/` with upstream notices; they are not exposed by the application.
