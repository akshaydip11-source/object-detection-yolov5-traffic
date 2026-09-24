from PIL import Image
import pytest
import yaml

from training.dataset import DatasetError, validate_dataset


@pytest.fixture
def dataset(tmp_path):
    for index, split in enumerate(("train", "val")):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (40, 40), (index * 100, 0, 0)).save(
            tmp_path / "images" / split / "sample.jpg"
        )
        (tmp_path / "labels" / split / "sample.txt").write_text(
            "0 0.2 0.2 0.1 0.1\n1 0.5 0.5 0.1 0.1\n2 0.8 0.8 0.1 0.1\n"
        )
    path = tmp_path / "traffic.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "names": {2: "License_Plate", 0: "Helmet", 1: "No_Helmet"},
            }
        )
    )
    return path


def test_valid_dataset_aliases(dataset):
    report = validate_dataset(dataset)
    assert report["splits"]["train"]["images"] == 1
    assert len(report["fingerprint_sha256"]) == 64


@pytest.mark.parametrize(
    "bad",
    [
        "3 0.5 0.5 0.1 0.1",
        "0 nan 0.5 0.1 0.1",
        "0 0.5 0.5 0 0.1",
        "0 0.95 0.5 0.2 0.1",
        "0 0.5 0.5 0.1",
        "car 0.5 0.5 0.1 0.1",
    ],
)
def test_invalid_box(dataset, bad):
    (dataset.parent / "labels/train/sample.txt").write_text(bad)
    with pytest.raises(DatasetError, match="Invalid normalized box"):
        validate_dataset(dataset)


def test_missing_label(dataset):
    (dataset.parent / "labels/train/sample.txt").unlink()
    with pytest.raises(DatasetError, match="Missing label"):
        validate_dataset(dataset)


def test_split_leakage(dataset):
    (dataset.parent / "images/val/sample.jpg").write_bytes(
        (dataset.parent / "images/train/sample.jpg").read_bytes()
    )
    with pytest.raises(DatasetError, match="leakage"):
        validate_dataset(dataset)


def test_orphan_label(dataset):
    (dataset.parent / "labels/train/orphan.txt").touch()
    with pytest.raises(DatasetError, match="Orphan label"):
        validate_dataset(dataset)


def test_empty_data(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "names": ["Helmet", "NoHelmet", "LicensePlate"],
            }
        )
    )
    with pytest.raises(DatasetError, match="No train images"):
        validate_dataset(path)


def test_no_examples_for_class(dataset):
    (dataset.parent / "labels/train/sample.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(DatasetError, match="no examples"):
        validate_dataset(dataset)


def test_corrupt_image(dataset):
    (dataset.parent / "images/train/sample.jpg").write_bytes(b"not an image")
    with pytest.raises(DatasetError, match="Corrupt image"):
        validate_dataset(dataset)
