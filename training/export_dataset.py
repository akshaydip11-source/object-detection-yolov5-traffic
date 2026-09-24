"""Export a portable annotated ZIP with exactly three train-only image variants.

Original, horizontal flip (boxes transformed), brightness +25% (boxes unchanged).
Validation/test images are copied untouched. This is annotation-preserving data
augmentation, not new independent observations or additional manual annotation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from PIL import Image, ImageEnhance, ImageOps
import yaml

from training.dataset import DatasetError, IMAGE_SUFFIXES, validate_dataset
from training.public_dataset import annotation


def flip_labels(text):
    rows = []
    for line in text.splitlines():
        if line.strip():
            cid, x, y, w, h = line.split()
            rows.append(f"{cid} {1 - float(x):.12g} {y} {w} {h}")
    return '\n'.join(rows) + ('\n' if rows else '')


def export_dataset(data_file, output, attribution=None):
    data_file, output = Path(data_file).resolve(), Path(output).resolve()
    if output.exists():
        raise DatasetError('Output exists; choose a new directory')
    source_report = validate_dataset(data_file)
    config = yaml.safe_load(data_file.read_text())
    attribution = Path(attribution) if attribution else data_file.parent / 'ATTRIBUTION.md'
    if not attribution.is_file():
        raise DatasetError('Provide an attribution/license file before exporting')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dataset-export-', dir=output.parent) as temp:
        root = Path(temp) / 'dataset'
        root.mkdir()
        manifest = []
        base = Path(source_report['path'])
        for split in source_report['splits']:
            images = (base / config[split]).resolve()
            parts = list(images.parts)
            parts[len(parts) - 1 - parts[::-1].index('images')] = 'labels'
            labels = Path(*parts)
            (root / 'images' / split).mkdir(parents=True)
            (root / 'labels' / split).mkdir(parents=True)
            for image in sorted(images.rglob('*')):
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                relative = image.relative_to(images)
                text = (labels / relative.with_suffix('.txt')).read_text()
                with Image.open(image) as metadata:
                    if metadata.getexif().get(274, 1) != 1:
                        raise DatasetError('Normalize image EXIF and labels together before exporting')
                group = hashlib.sha256(f'{split}/{relative}'.encode()).hexdigest()[:24]
                original_name = group + '__original' + image.suffix.lower()
                shutil.copyfile(image, root / 'images' / split / original_name)
                (root / 'labels' / split / (Path(original_name).stem + '.txt')).write_text(text)
                names = [(original_name, 'original')]
                if split == 'train':
                    with Image.open(image) as opened:
                        # Applying EXIF rotation here would invalidate stored coordinates.
                        if opened.getexif().get(274, 1) != 1:
                            raise DatasetError('Normalize image EXIF and labels together before augmentation')
                        decoded = opened.convert('RGB')
                        variants = [
                            ('flip', ImageOps.mirror(decoded), flip_labels(text)),
                            ('brightness', ImageEnhance.Brightness(decoded).enhance(1.25), text),
                        ]
                        for variant, transformed, annotation_text in variants:
                            name = f'{group}__{variant}.jpg'
                            transformed.save(root / 'images' / split / name, quality=95)
                            (root / 'labels' / split / (Path(name).stem + '.txt')).write_text(annotation_text)
                            names.append((name, variant))
                for name, variant in names:
                    manifest.append({'split': split, 'file': name, 'source_image': str(relative),
                                     'original_group': group, 'variant': variant})
        portable = {'nc': 3, 'names': ['Helmet', 'No_Helmet', 'License_Plate']}
        portable.update({split: f'images/{split}' for split in source_report['splits']})
        (root / 'data.yaml').write_text(yaml.safe_dump(portable))
        result = validate_dataset(root / 'data.yaml')
        report = {'source_fingerprint': source_report['fingerprint_sha256'],
                  'source_splits': source_report['splits'], 'export_splits': result['splits'],
                  'fingerprint_sha256': result['fingerprint_sha256'], 'manifest': manifest,
                  'note': 'Three variants per training original only. No validation/test augmentation. Not three times as many independent images.'}
        (root / 'export_report.json').write_text(json.dumps(report, indent=2))
        shutil.copyfile(attribution, root / 'ATTRIBUTION.md')
        with (root / 'ATTRIBUTION.md').open('a') as f:
            f.write('\n\nExport modifications: training-only horizontal flips with mirrored boxes, brightness +25%, and class display aliases Helmet/No_Helmet/License_Plate. No new objects annotated.\n')
        provenance = data_file.parent / 'provenance.json'
        if provenance.is_file():
            shutil.copyfile(provenance, root / 'source_provenance.json')
        (root / 'README.md').write_text('# Annotated dataset\n\nExtract to a new directory and train using the project wrapper:\n\n`python training/train_yolov5.py --data /absolute/path/to/data.yaml --weights yolov5s.pt --img-size 640 --hyp training/hyps/traffic.yaml`\n\nThe wrapper resolves portable YAML paths before invoking vendored YOLOv5. Keep variants with their original split. Read ATTRIBUTION.md and export_report.json.\n')
        if output.exists():
            raise DatasetError('Output appeared during export; refusing overwrite')
        root.rename(output)
    return report


def archive_dataset(output):
    output = Path(output)
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise DatasetError('Archive exists; refusing overwrite')
    partial = archive.with_suffix('.zip.part')
    try:
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(output.rglob('*')):
                if file.is_file():
                    z.write(file, file.relative_to(output))
        if partial.stat().st_size > 250_000_000:
            raise DatasetError('Archive exceeds 250 MB handover limit')
        partial.rename(archive)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attribution', type=Path)
    args = parser.parse_args()
    report = export_dataset(args.data, args.output, args.attribution)
    archive = archive_dataset(args.output)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    annotation('notice', 'Annotated dataset export', json.dumps({
        'archive': archive.name, 'sha256': digest,
        'original_splits': report['source_splits'], 'export_splits': report['export_splits'],
        'note': report['note'],
    }))


if __name__ == '__main__':
    main()
