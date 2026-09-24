# Trusted custom weights (not included)

Place the validated custom YOLOv5 `best.pt` here, or configure `MODEL_PATH` in `.env`.
Expected classes: Helmet, NoHelmet, LicensePlate. Class order comes from the checkpoint.

Never commit weights without permission or load a `.pt` from an untrusted source;
checkpoint deserialization can execute code. Record the file's SHA-256, training
revision, dependency versions and validation metrics with the internship handover.
A new experimental public-data checkpoint now exists in GitHub Actions and has
passed real Docker/browser integration. Its measured accuracy is **too poor for
release**; it is deliberately not installed here. See [actual results](../docs/TRAINING_RESULTS.md).
Successful loading is not proof of acceptable detection accuracy. Never deploy that
pilot merely to remove the missing-model warning.
