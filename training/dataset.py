"""Validate the exact three-class YOLO dataset before starting expensive training."""

from collections import Counter
import hashlib
import math
from pathlib import Path

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
YOLOV5 = ROOT / "yolov5"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED = ["helmet", "nohelmet", "licenseplate"]


class DatasetError(ValueError):
    pass


def validate_dataset(config_path: Path) -> dict:
    """YOLOv5 resolves a relative `path` against its vendored root, not shell cwd.

    Requires train/val image directories and parallel labels directories. Empty
    label files are valid negatives, but every class must occur in each split.
    Rejects corrupt images, bad boxes, orphan labels and byte-identical split leakage.
    """
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise DatasetError(f"Dataset YAML missing: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise DatasetError("Malformed dataset YAML") from exc
    if not isinstance(config, dict):
        raise DatasetError("Dataset YAML must be a mapping")
    names = config.get("names", [])
    if isinstance(names, dict):
        if set(names) != {0, 1, 2}:
            raise DatasetError("Class IDs must be 0, 1 and 2")
        names = [names[i] for i in range(3)]
    if (
        not isinstance(names, list)
        or ["".join(c for c in str(n).lower() if c.isalnum()) for n in names]
        != EXPECTED
    ):
        raise DatasetError(
            "Class order must be Helmet, NoHelmet, LicensePlate (underscores/spaces accepted)"
        )
    if "nc" in config and config["nc"] != 3:
        raise DatasetError("nc must equal 3")
    base = Path(config.get("path") or config_path.parent)
    base = base if base.is_absolute() else (YOLOV5 / base).resolve()
    report = {"names": names, "path": str(base), "splits": {}}
    seen_hashes = {}
    combined = hashlib.sha256()
    for split in ("train", "val"):
        value = config.get(split)
        if not isinstance(value, str):
            raise DatasetError(f"{split} must name an image directory")
        folder = (base / value).resolve()
        if (
            folder.name not in {"train", "val", "valid", "images"}
            or "images" not in folder.parts
        ):
            raise DatasetError(
                f"{split} must be under an images directory; use parallel labels directories"
            )
        parts = list(folder.parts)
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        labels_folder = Path(*parts)
        images = (
            sorted(
                p
                for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            if folder.is_dir()
            else []
        )
        if not images:
            raise DatasetError(
                f"No {split} images in {folder}; supply the annotated dataset before training"
            )
        labels_seen = set()
        counts = Counter()
        for image in images:
            label = labels_folder / image.relative_to(folder).with_suffix(".txt")
            if label in labels_seen:
                raise DatasetError(f"Ambiguous image stems share a label: {label}")
            labels_seen.add(label)
            if not label.is_file():
                raise DatasetError(f"Missing label for {image.name}: {label}")
            try:
                with Image.open(image) as decoded:
                    decoded.verify()
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise DatasetError(f"Corrupt image: {image}") from exc
            with image.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            if digest in seen_hashes and seen_hashes[digest] != split:
                raise DatasetError(
                    f"Train/validation leakage: duplicate image {image.name}"
                )
            seen_hashes[digest] = split
            text = label.read_text()
            combined.update(
                f"{split}/{image.relative_to(folder)}:{digest}\n{text}".encode()
            )
            for row, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                try:
                    if len(fields) != 5:
                        raise ValueError()
                    cid = int(fields[0])
                    x, y, width, height = map(float, fields[1:])
                    if cid not in {0, 1, 2} or not all(
                        math.isfinite(v) and 0 <= v <= 1 for v in (x, y, width, height)
                    ):
                        raise ValueError()
                    if (
                        width <= 0
                        or height <= 0
                        or x - width / 2 < -1e-6
                        or y - height / 2 < -1e-6
                        or x + width / 2 > 1 + 1e-6
                        or y + height / 2 > 1 + 1e-6
                    ):
                        raise ValueError()
                except ValueError as exc:
                    raise DatasetError(
                        f"Invalid normalized box in {label}:{row}"
                    ) from exc
                counts[cid] += 1
        orphans = set(labels_folder.rglob("*.txt")) - labels_seen
        if orphans:
            raise DatasetError(f"Orphan label without an image: {sorted(orphans)[0]}")
        missing = [names[i] for i in range(3) if not counts[i]]
        if missing:
            raise DatasetError(f"{split} has no examples of: {', '.join(missing)}")
        report["splits"][split] = {
            "images": len(images),
            "boxes_per_class": {names[i]: counts[i] for i in range(3)},
        }
    report["fingerprint_sha256"] = combined.hexdigest()
    return report
