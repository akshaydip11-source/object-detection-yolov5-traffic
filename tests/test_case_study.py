import hashlib
from pathlib import Path
import zipfile

from PIL import Image
import pytest
import yaml

from training.dataset import DatasetError, validate_dataset
from training.export_dataset import archive_dataset, export_dataset, flip_labels
from training.train_yolov5 import materialize_data_config
from inference.make_demo import validate_window
from tests.test_application import image


def test_brief_api_contract_and_signed_preview(client, detector):
    r = client.post('/api/detect/predict', files={'file': image()})
    assert r.status_code == 200, r.text
    result = r.json()
    assert result['box_format'] == 'xywh'
    assert result['coordinate_system'] == 'image_pixels'
    assert result['detections'][0] == {'class': 'No_Helmet', 'confidence': 0.9, 'box': [10, 10, 20, 20]}
    assert result['detections'][1]['class'] == 'License_Plate'
    assert client.get(result['annotated_image_url']).status_code == 200
    # Existing console contract remains corners, not width/height.
    old = client.post('/api/detect/image', files={'file': image()}).json()
    assert old['detections'][0]['box']['x2'] == 30


def test_brief_api_requires_access_token(client):
    client.headers.pop('Authorization')
    assert client.post('/api/detect/predict', files={'file': image()}).status_code == 401


@pytest.mark.parametrize('conf', ['nan', 'inf', '-1', '1.1'])
def test_brief_api_bad_threshold(client, detector, conf):
    assert client.post('/api/detect/predict', files={'file': image()}, data={'conf_threshold': conf}).status_code == 422


@pytest.fixture
def small_dataset(tmp_path):
    for index, split in enumerate(['train', 'val', 'test']):
        (tmp_path / 'images' / split).mkdir(parents=True)
        (tmp_path / 'labels' / split).mkdir(parents=True)
        Image.new('RGB', (32, 32), (50 + index * 70, 30, 60)).save(tmp_path / 'images' / split / 'sample.jpg')
        (tmp_path / 'labels' / split / 'sample.txt').write_text('0 0.2 0.2 0.1 0.1\n1 0.5 0.5 0.1 0.1\n2 0.8 0.8 0.1 0.1\n')
    config = tmp_path / 'data.yaml'
    config.write_text(yaml.safe_dump({'names': ['Helmet','No_Helmet','License_Plate'],
                                     'train':'images/train','val':'images/val','test':'images/test'}))
    (tmp_path / 'ATTRIBUTION.md').write_text('Generated test-only fixtures, not real training observations.')
    return config


def test_export_triples_only_training_and_is_portable(small_dataset, tmp_path):
    output = tmp_path / 'bundle'
    report = export_dataset(small_dataset, output)
    assert {s: row['images'] for s, row in report['export_splits'].items()} == {'train':3,'val':1,'test':1}
    assert len({r['original_group'] for r in report['manifest'] if r['split']=='train'}) == 1
    for split in ['val','test']:
        original = small_dataset.parent / 'images' / split / 'sample.jpg'
        copied = next((output / 'images' / split).iterdir())
        assert hashlib.sha256(copied.read_bytes()).digest() == hashlib.sha256(original.read_bytes()).digest()
    flipped = next((output / 'labels/train').glob('*__flip.txt')).read_text().splitlines()
    assert float(flipped[0].split()[1]) == pytest.approx(0.8)
    assert float(flipped[2].split()[1]) == pytest.approx(0.2)
    config = yaml.safe_load((output / 'data.yaml').read_text())
    assert 'path' not in config
    archive = archive_dataset(output)
    unpacked = tmp_path / 'relocated'
    with zipfile.ZipFile(archive) as z:
        z.extractall(unpacked)
    audit = validate_dataset(unpacked / 'data.yaml')
    runtime = materialize_data_config(unpacked / 'data.yaml', audit, tmp_path / 'runtime/data.yaml')
    resolved = yaml.safe_load(runtime.read_text())
    assert Path(resolved['train']) == unpacked / 'images/train'
    assert Path(resolved['test']) == unpacked / 'images/test'
    assert 'download' not in resolved
    with pytest.raises(DatasetError, match='exists'):
        export_dataset(small_dataset, output)
    with pytest.raises(DatasetError, match='exists'):
        archive_dataset(output)


def test_attribution_required(small_dataset, tmp_path):
    (small_dataset.parent / 'ATTRIBUTION.md').unlink()
    with pytest.raises(DatasetError, match='attribution'):
        export_dataset(small_dataset, tmp_path / 'output')


def test_flip_empty_negative():
    assert flip_labels('') == ''


@pytest.mark.parametrize('start,seconds', [(-1,30),(0,29),(2,31),(float('nan'),40)])
def test_no_fake_or_looped_30_second_demo(start, seconds):
    with pytest.raises(ValueError, match='continuous 30-second'):
        validate_window({'seconds': seconds}, start)


def test_genuine_30_second_window():
    validate_window({'seconds': 31}, 1)


def test_demo_file_contract_with_synthetic_fixture_not_real_demo(tmp_path, detector, monkeypatch):
    import cv2
    import numpy as np
    from backend.app.config import settings
    from backend.app.services import detector as runtime
    from inference.make_demo import make_demo, video_info

    source = tmp_path / 'synthetic-test-only.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 1, (80, 80))
    assert writer.isOpened()
    for i in range(30):
        writer.write(np.full((80, 80, 3), i * 4, dtype=np.uint8))
    writer.release()
    checkpoint = tmp_path / 'test-only.pt'
    checkpoint.write_bytes(b'not a checkpoint; model constructor stubbed in this test')
    credit = tmp_path / 'credit.txt'
    credit.write_text('Synthetic unit-test fixture, NOT a street-scene demo or accuracy evidence.')
    monkeypatch.setattr(runtime, 'YOLODetector', lambda: detector)
    # Restore process-local settings after exercising the standalone CLI implementation.
    monkeypatch.setattr(settings, 'model_path', settings.model_path)
    monkeypatch.setattr(settings, 'max_video_seconds', settings.max_video_seconds)
    output = tmp_path / 'test-demo'
    result = make_demo(source, checkpoint, credit, output)
    assert result['real_time_certified'] is False
    assert result['summary']['frames_processed'] == 300
    assert video_info(output / 'demo-30s.mp4')['seconds'] == pytest.approx(30, abs=0.1)
    assert (output / 'FOOTAGE_LICENSE.txt').read_text() == credit.read_text()
