# Third-party source and assets

- `yolov5/` is vendored Ultralytics YOLOv5 source. Existing copyright notices,
  documentation and **AGPL-3.0** license at `yolov5/LICENSE` are retained unchanged.
  This audit does not relicense upstream code. Review the applicable AGPL/commercial
  terms before distributing or deploying the combined application, including any
  source-availability obligations. The owner must settle the application-level
  licensing consistently with these obligations; no blanket license grant is invented.
- Python dependencies retain their respective licenses. `ultralytics` also has
  licensing requirements; installation is not a blanket commercial license grant.
- Unattributed frontend photographs were removed from the current working tree.
  `frontend/samples/` now contains only three synthetic illustrations rendered by
  the project's OpenCV/NumPy script; see its README. These are not training data or
  evaluation evidence. Removal does not rewrite previous Git history.
- Vendored upstream sample images remain in `yolov5/data/images/` with the upstream
  tree; they are not served by the application. Do not assume this grants permission
  to reuse arbitrary photographs for your dataset.
- Google Fonts are requested remotely by the frontend. They are optional presentation
  dependencies; system-font fallbacks work without the remote font service.
- Custom weights and traffic datasets are not included. Confirm rights, privacy and
  redistribution permissions for the actual artifacts before supplying them.

- `imageio-ffmpeg` provides the FFmpeg binary used for H.264 output. Review the
  bundled FFmpeg/codec license notices in that dependency before distribution;
  adding this dependency is not a blanket commercial codec/license grant.
