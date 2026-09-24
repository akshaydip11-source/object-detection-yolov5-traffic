# Trusted custom weights (not included)

Place the validated custom YOLOv5 `best.pt` here, or configure `MODEL_PATH` in `.env`.
Expected classes: Helmet, NoHelmet, LicensePlate. Class order comes from the checkpoint.

Never commit weights without permission or load a `.pt` from an untrusted source;
checkpoint deserialization can execute code. Record the file's SHA-256, training
revision, dependency versions and validation metrics with the internship handover.
No model accuracy or tested-checkpoint compatibility has been established here.
