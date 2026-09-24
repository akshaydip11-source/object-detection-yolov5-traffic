import hashlib
import io
import json
import zipfile

from PIL import Image
import pytest
import yaml

from training import public_dataset as public
from training.dataset import DatasetError


def archive_fixture(tmp_path, monkeypatch, *, license_name="CC BY 4.0", classes=None, duplicate_group=False, extra=None):
    path = tmp_path / "source.zip"
    config = {"names": public.SOURCE_NAMES if classes is None else classes, "roboflow": {"license": license_name}}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data.yaml", yaml.safe_dump(config))
        for number, split in enumerate(("train", "valid", "test")):
            image = io.BytesIO()
            Image.new("RGB", (24, 24), (number * 70, 30, 70)).save(image, format="PNG")
            stem = ("shared" if duplicate_group else split) + ".rf.variant"
            z.writestr(f"{split}/images/{stem}.png", image.getvalue())
            z.writestr(f"{split}/labels/{stem}.txt", "0 0.3 0.3 0.1 0.1\n4 0.5 0.5 0.1 0.1\n1 0.7 0.7 0.1 0.1\n5 0.5 0.5 0.8 0.8\n")
        if extra:
            z.writestr(*extra)
    monkeypatch.setattr(public, "SOURCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_remap_only_explicit_classes():
    text, counts = public.remap_labels("0 0.5 0.5 0.2 0.2\n4 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n2 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.2 0.2\n5 0.5 0.5 0.2 0.2")
    assert [int(row.split()[0]) for row in text.splitlines()] == [0, 1, 2]
    assert dict(counts) == {0: 1, 1: 1, 2: 1}


@pytest.mark.parametrize("text", ["0 nan 0.5 0.1 0.1", "7 0.5 0.5 0.1 0.1", "1 0 0 1 1", "4 0.5 0.5 0 0", "5 broken"])
def test_reject_bad_source_annotations(text):
    with pytest.raises(DatasetError, match="Invalid source annotation"):
        public.remap_labels(text)


def test_prepare_validates_and_preserves_attribution(tmp_path, monkeypatch):
    archive = archive_fixture(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    report = public.prepare(archive, output)
    assert report["splits"]["train"]["boxes_per_class"] == {"Helmet": 1, "NoHelmet": 1, "LicensePlate": 1}
    assert report["splits"]["test"]["images"] == 1
    assert len(report["manifest"]) == 3
    assert json.loads((output / "provenance.json").read_text())["attribution"]["license"] == "CC BY 4.0"
    with pytest.raises(DatasetError, match="already exists"):
        public.prepare(archive, output)


@pytest.mark.parametrize("kwargs,message", [
    ({"license_name": "Private"}, "license"),
    ({"classes": ["person", "bike", "plate"]}, "classes"),
    ({"duplicate_group": True}, "crosses published splits"),
    ({"extra": ("../escape.txt", "bad")}, "Unsafe archive"),
])
def test_reject_unreviewed_or_leaking_exports(tmp_path, monkeypatch, kwargs, message):
    archive = archive_fixture(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(DatasetError, match=message):
        public.prepare(archive, tmp_path / "prepared")


def test_checksum_and_grouping(tmp_path, monkeypatch):
    archive = archive_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(public, "SOURCE_SHA256", "0" * 64)
    with pytest.raises(DatasetError, match="SHA256"):
        public.prepare(archive, tmp_path / "prepared")
    assert public.source_group("train/images/frame001_jpg.rf.abc.jpg") == public.source_group("valid/images/frame001_jpg.rf.xyz.jpg")


def test_selection_deterministic_and_covers_rare_class():
    records = [{"image": str(i), "counts": {i % 3: 1}} for i in range(100)]
    one = public.select_records(records, 30)
    assert one == public.select_records(list(reversed(records)), 30)
    assert len(one) == 30
    assert all(sum(bool(r["counts"].get(cid)) for r in one) >= 8 for cid in range(3))
