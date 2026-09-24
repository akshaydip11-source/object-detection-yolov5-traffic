"""Prepare a small, traceable RASYD/MotorbikeDelivery pilot; never invent labels.

Downloads a pinned CC BY 4.0 public export, verifies its Git LFS SHA256, selects
one augmentation per named source image, preserves published split assignments,
and remaps only Helmet, No_Helmet and License_plate. Not deployment approval.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

import yaml

from training.dataset import DatasetError, validate_dataset

SOURCE_COMMIT = "80169b657bd537c2f021f23399bc45e8d2e552d2"
SOURCE_URL = (
    "https://media.githubusercontent.com/media/NasserAlsaqer/"
    "Detect-Violated-Motorbikes-Delivery-RASYD/" + SOURCE_COMMIT
    + "/Dataset/RASYD_Datasets_YOLOv8.zip"
)
SOURCE_SHA256 = "dab329daaa7213a3307b3677dab82e6024ffe41038870e7d970d7fa07fe9bb9d"
SOURCE_SIZE = 502502186
SOURCE_NAMES = ["Helmet", "License_plate", "MotorbikeDelivery", "MotorbikeSport", "No_Helmet", "Person"]
NAMES = ["Helmet", "NoHelmet", "LicensePlate"]
CLASS_MAP = {0: 0, 4: 1, 1: 2}
ATTRIBUTION = {
    "title": "MotorbikeDelivery_2.0, version 8",
    "author": "MotorbikeDelivery; RASYD dataset contributors",
    "source": "https://universe.roboflow.com/motorbikedelivery/motorbikedelivery_2.0/dataset/8",
    "mirror": "https://github.com/NasserAlsaqer/Detect-Violated-Motorbikes-Delivery-RASYD",
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "mirror_commit": SOURCE_COMMIT,
    "archive_sha256": SOURCE_SHA256,
    "changes": "Deterministic subset; explicit polygon-to-axis-aligned-box conversion when requested; one variant per named original; explicitly quarantine invalid groups when requested; drop non-target classes; remap IDs to Helmet=0, NoHelmet=1, LicensePlate=2. No new annotations.",
    "limitations": "Publisher-declared license, not a warranty of underlying image rights. Source-video identities and near-duplicate scenes are not established; group isolation by original filename is not video-level isolation. Pilot only, not enforcement approval.",
}


def annotation(level, title, message):
    """Expose bounded audit evidence through native GitHub annotations (no write token)."""
    text = str(message)[:8000].replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level} title={title}::{text}", flush=True)


def source_group(name):
    """Roboflow variants share the filename before `.rf.`."""
    return PurePosixPath(name).name.split(".rf.")[0]


def remap_labels(text, polygon_boxes=False, statistics=None):
    rows, counts = [], Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        try:
            if not fields:
                raise ValueError()
            cid = int(fields[0])
            converted = False
            if len(fields) == 5:
                x, y, w, h = map(float, fields[1:])
                box_fields = fields[1:]
            elif polygon_boxes and len(fields) >= 7 and len(fields) % 2 == 1:
                points = list(map(float, fields[1:]))
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in points):
                    raise ValueError()
                xs, ys = points[0::2], points[1::2]
                area = sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i] for i in range(len(xs)))
                if abs(area) < 1e-12:
                    raise ValueError()
                left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
                x, y, w, h = (left + right) / 2, (top + bottom) / 2, right - left, bottom - top
                box_fields = [format(v, ".12g") for v in (x, y, w, h)]
                converted = True
            else:
                raise ValueError()
            if cid not in range(len(SOURCE_NAMES)) or not all(math.isfinite(v) for v in (x, y, w, h)):
                raise ValueError()
            if not (0 < w <= 1 and 0 < h <= 1 and w / 2 - 1e-6 <= x <= 1 - w / 2 + 1e-6 and h / 2 - 1e-6 <= y <= 1 - h / 2 + 1e-6):
                raise ValueError()
        except ValueError as exc:
            raise DatasetError("Invalid source annotation; no automatic box repair performed: " + json.dumps(fields)) from exc
        if cid in CLASS_MAP:
            target = CLASS_MAP[cid]
            rows.append(" ".join([str(target), *box_fields]))
            if converted and statistics is not None:
                statistics[target] += 1
            counts[target] += 1
    return "\n".join(rows) + ("\n" if rows else ""), counts


def select_records(records, limit):
    """Deterministic original-image sampling with minimum class coverage, not box creation."""
    ordered = sorted(records, key=lambda r: hashlib.sha256(r["image"].encode()).hexdigest())
    selected, seen = [], set()
    # Include up to eight distinct source images for each class, especially rare NoHelmet.
    for cid in range(3):
        candidates = [r for r in ordered if r["counts"].get(cid, 0)]
        if not candidates:
            raise DatasetError(f"Published split has no {NAMES[cid]} examples")
        for record in candidates[:8]:
            if record["image"] not in seen and len(selected) < limit:
                selected.append(record)
                seen.add(record["image"])
    for record in ordered:
        if len(selected) >= limit:
            break
        if record["image"] not in seen:
            selected.append(record)
            seen.add(record["image"])
    return selected


def prepare(archive, output, limits=None, quarantine_invalid=False, polygon_boxes=False):
    limits = limits or {"train": 320, "val": 64, "test": 64}
    if output.exists():
        raise DatasetError("Output already exists; use a new directory (never overwrite data)")
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SOURCE_SHA256:
            raise DatasetError("Public archive SHA256 mismatch")
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len(members) > 25000 or sum(m.file_size for m in members) > 3_000_000_000:
            raise DatasetError("Archive exceeds pilot limits")
        if any(m.file_size > 20_000_000 or ".." in PurePosixPath(m.filename).parts or m.filename.startswith("/") for m in members):
            raise DatasetError("Unsafe archive entry")
        if len({m.filename for m in members}) != len(members):
            raise DatasetError("Duplicate archive entry")
        configs = [m.filename for m in members if PurePosixPath(m.filename).name == "data.yaml"]
        if len(configs) != 1:
            raise DatasetError("Expected exactly one source data.yaml")
        config_name = configs[0]
        config = yaml.safe_load(z.read(config_name))
        if not isinstance(config, dict) or config.get("names") != SOURCE_NAMES:
            raise DatasetError("Unexpected source classes; refusing guessed remapping")
        if config.get("roboflow", {}).get("license") != "CC BY 4.0":
            raise DatasetError("Source export license is not the reviewed CC BY 4.0 license")
        root = str(PurePosixPath(config_name).parent)
        root = "" if root == "." else root + "/"
        available = {m.filename for m in members}
        records, groups, manifest, quarantined = {}, {}, [], {}
        visited = set()
        for split, source_split in (("train", "train"), ("val", "valid"), ("test", "test")):
            found = []
            for name in sorted(available):
                if not name.startswith(root + source_split + "/images/") or PurePosixPath(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                group = source_group(name)
                groups.setdefault(group, set()).add(split)
                if (split, group) in visited:
                    continue
                visited.add((split, group))
                label = str(PurePosixPath(name.replace("/images/", "/labels/")).with_suffix(".txt"))
                if label not in available:
                    raise DatasetError("Missing source label: " + label)
                polygon_counts = Counter()
                try:
                    text, counts = remap_labels(z.read(label).decode("utf-8"), polygon_boxes=polygon_boxes, statistics=polygon_counts)
                except DatasetError as exc:
                    if not quarantine_invalid:
                        raise DatasetError(f"{exc}: {label}") from exc
                    quarantined.setdefault(group, []).append({"source_label": label, "reason": str(exc)})
                    continue
                found.append({"image": name, "label": label, "group": group, "text": text, "counts": counts, "polygons": polygon_counts})
            records[split] = found
        for group, splits in groups.items():
            if len(splits) > 1:
                if not quarantine_invalid:
                    raise DatasetError("Named original/augmentation crosses published splits: " + group)
                quarantined.setdefault(group, []).append({"reason": "Named original crosses published splits", "splits": sorted(splits)})
        # Exclude whole originals from every split; never move a validation image to train.
        # Refill deterministically if differently named selected images have identical bytes.
        while True:
            if len(quarantined) > len(groups) * 0.05:
                examples = dict(list(quarantined.items())[:2])
                raise DatasetError(f"Invalid/conflicting source groups exceed 5% quarantine limit ({len(quarantined)}/{len(groups)}): " + json.dumps(examples))
            selected = {split: select_records([r for r in rows if r["group"] not in quarantined], limits[split]) for split, rows in records.items()}
            hashes, duplicates = {}, {}
            for rows in selected.values():
                for row in rows:
                    digest = hashlib.sha256(z.read(row["image"])).hexdigest()
                    if digest in hashes:
                        if not quarantine_invalid:
                            raise DatasetError("Byte-identical duplicate in selected data: " + row["image"])
                        for record in (row, hashes[digest]):
                            duplicates[record["group"]] = [{"reason": "Byte-identical selected image", "sha256": digest}]
                    hashes[digest] = row
            if not duplicates:
                records = selected
                break
            quarantined.update(duplicates)
        output.mkdir(parents=True)
        hashes = {}
        for split, rows in records.items():
            (output / "images" / split).mkdir(parents=True)
            (output / "labels" / split).mkdir(parents=True)
            for row in rows:
                image = z.read(row["image"])
                digest = hashlib.sha256(image).hexdigest()
                if digest in hashes:
                    raise DatasetError("Byte-identical duplicate in selected data: " + row["image"])
                hashes[digest] = split
                # Source basename is not trusted as a filesystem path.
                stem = hashlib.sha256(row["image"].encode()).hexdigest()[:24]
                filename = stem + PurePosixPath(row["image"]).suffix.lower()
                (output / "images" / split / filename).write_bytes(image)
                (output / "labels" / split / (stem + ".txt")).write_text(row["text"])
                manifest.append({"split": split, "file": filename, "source_image": row["image"], "source_group": row["group"], "image_sha256": digest})
        data = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "nc": 3, "names": NAMES}
        data_file = output / "data.yaml"
        data_file.write_text(yaml.safe_dump(data))
        report = validate_dataset(data_file)
        report["quarantined_source_groups"] = [{"source_group": group, "reasons": reasons} for group, reasons in sorted(quarantined.items())]
        report["polygon_boxes_enabled"] = polygon_boxes
        report["selected_polygon_conversions"] = {
            split: {NAMES[cid]: sum(row["polygons"][cid] for row in rows) for cid in range(3)}
            for split, rows in records.items()
        }
        report["attribution"] = ATTRIBUTION
        report["manifest"] = manifest
        report["source_groups_by_split"] = dict(Counter(split for splits in groups.values() for split in splits))
        (output / "provenance.json").write_text(json.dumps(report, indent=2))
        (output / "ATTRIBUTION.md").write_text("# Public pilot dataset\n\n" + "\n\n".join(f"**{k}**: {v}" for k, v in ATTRIBUTION.items()) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded, checksum-verified export")
    parser.add_argument("--download-cache", type=Path, help="Cache only the checksum-verified public ZIP for repeated CI audits")
    parser.add_argument("--polygon-boxes", action="store_true", help="Explicitly derive axis-aligned boxes from genuine normalized polygon annotations; preserve class semantics")
    parser.add_argument("--quarantine-invalid", action="store_true", help="Explicitly exclude invalid or leaking original-image groups, up to 5%; record every exclusion, never repair boxes")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    if args.archive and args.download_cache:
        parser.error("Choose --archive or --download-cache, not both")
    with tempfile.TemporaryDirectory(prefix="safecity-public-") as temp:
        archive = args.archive or args.download_cache or Path(temp) / "source.zip"
        if not args.archive and not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            partial = archive.with_suffix(".part")
            digest = hashlib.sha256()
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, partial.open("wb") as stream:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > SOURCE_SIZE:
                        raise DatasetError("Public archive exceeds expected size")
                    digest.update(chunk)
                    stream.write(chunk)
            if size != SOURCE_SIZE or digest.hexdigest() != SOURCE_SHA256:
                raise DatasetError("Public archive size/hash mismatch")
            partial.replace(archive)
        # Build transactionally: an invalid source never leaves a usable partial dataset.
        staging = Path(temp) / "prepared"
        report = prepare(archive, staging, quarantine_invalid=args.quarantine_invalid, polygon_boxes=args.polygon_boxes)
        if args.output.exists():
            raise DatasetError("Output already exists; choose a new directory")
        shutil.copytree(staging, args.output)
        for name in ("data.yaml",):
            config = yaml.safe_load((args.output / name).read_text())
            config["path"] = str(args.output.resolve())
            (args.output / name).write_text(yaml.safe_dump(config))
        report["path"] = str(args.output.resolve())
        (args.output / "provenance.json").write_text(json.dumps(report, indent=2))
        summary = {k: v for k, v in report.items() if k != "manifest"}
        exclusions = summary.pop("quarantined_source_groups")
        summary["quarantine"] = {"count": len(exclusions), "examples": exclusions[:3]}
        print(json.dumps(summary, indent=2))
        annotation("notice", "Public dataset audit", json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        annotation("error", "Public dataset audit failed", f"{type(exc).__name__}: {exc}")
        raise
