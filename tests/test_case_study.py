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


def test_validation_images_with_exif_rotation_are_not_silently_mislabeled(small_dataset, tmp_path):
    file = small_dataset.parent / 'images/val/sample.jpg'
    with Image.open(file) as decoded:
        exif = Image.Exif()
        exif[274] = 6
        decoded.save(file, exif=exif)
    with pytest.raises(DatasetError, match='EXIF'):
        export_dataset(small_dataset, tmp_path / 'bundle')


def test_runtime_yaml_drops_unvalidated_download_directives(small_dataset, tmp_path):
    config = yaml.safe_load(small_dataset.read_text())
    config['download'] = 'must-not-be-forwarded-or-executed'
    small_dataset.write_text(yaml.safe_dump(config))
    report = validate_dataset(small_dataset)
    path = materialize_data_config(small_dataset, report, tmp_path / 'runtime.yaml')
    assert 'download' not in yaml.safe_load(path.read_text())


def test_archive_never_removes_another_writers_partial(small_dataset, tmp_path):
    output = tmp_path / 'bundle'
    export_dataset(small_dataset, output)
    existing = output.with_suffix('.zip.part')
    existing.write_text('another writer owns this')
    archive_dataset(output)
    assert existing.read_text() == 'another writer owns this'


def test_archive_race_cannot_clobber_completed_file(small_dataset, tmp_path, monkeypatch):
    import os
    output = tmp_path / 'bundle'
    export_dataset(small_dataset, output)
    original_link = os.link

    def competing_writer(source, target):
        Path(target).write_text('another completed archive')
        return original_link(source, target)

    monkeypatch.setattr(os, 'link', competing_writer)
    with pytest.raises(DatasetError, match='refusing overwrite'):
        archive_dataset(output)
    assert output.with_suffix('.zip').read_text() == 'another completed archive'
    assert not list(tmp_path.glob('.dataset-*.zip.part'))


def test_bundle_publication_preserves_existing_empty_directory(tmp_path):
    from scripts.file_publish import publish_directory
    source, output = tmp_path / 'staging', tmp_path / 'existing'
    source.mkdir()
    (source / 'file').write_text('new data')
    output.mkdir()
    with pytest.raises(FileExistsError):
        publish_directory(source, output)
    assert output.is_dir() and not list(output.iterdir())
    assert (source / 'file').read_text() == 'new data'


def test_failed_publication_cleans_only_its_new_destination(tmp_path, monkeypatch):
    import shutil
    from scripts.file_publish import publish_directory
    source, output = tmp_path / 'staging', tmp_path / 'new'
    source.mkdir()
    (source / 'file').write_text('original')

    def fail_copy(*args, **kwargs):
        (output / 'partial').write_text('partial')
        raise OSError('simulated disk failure')

    monkeypatch.setattr(shutil, 'copytree', fail_copy)
    with pytest.raises(OSError, match='simulated disk'):
        publish_directory(source, output)
    assert not output.exists()
    assert (source / 'file').read_text() == 'original'


def _results_csv(path, rows):
    header = 'epoch,train/box_loss,train/obj_loss,train/cls_loss,metrics/precision,metrics/recall,metrics/mAP_0.5,metrics/mAP_0.5:0.95,x/lr0'
    path.write_text('\n'.join([header] + rows) + '\n')
    return path


def test_training_curves_are_charted_from_the_runs_own_csv(tmp_path):
    from training.plot_results import plot, read_results, summarize

    csv = _results_csv(tmp_path / 'results.csv', [
        '0,0.2,0.15,0.05,0.1,0.2,0.2,0.08,0.01',
        '1,0.1,0.12,0.03,0.2,0.3,0.5,0.2,0.0098',
        '2,0.05,0.1,0.02,0.25,0.35,0.4,0.18,0.0096',
    ])
    results = read_results(csv)
    chart = plot(results, tmp_path / 'charts/training_curves.png')
    summary = summarize(results)
    assert chart.is_file() and chart.stat().st_size > 0
    assert summary['epochs_recorded'] == 3
    assert summary['best_epoch_by_mAP50'] == 1
    assert summary['best_mAP50'] == 0.5
    assert summary['first_epoch']['train/box_loss'] == 0.2
    assert summary['last_epoch']['train/box_loss'] == 0.05
    assert summary['csv_sha256'] == hashlib.sha256(csv.read_bytes()).hexdigest()


@pytest.mark.parametrize('content', [
    None,
    '',
    'epoch,train/box_loss\n0,0.2\n',
    'epoch,train/box_loss,metrics/mAP_0.5\n0,0.2,0.1\n1,nan,0.2\n',
])
def test_no_chart_without_a_real_run_csv(tmp_path, content):
    from training.plot_results import ResultsError, read_results

    csv = tmp_path / 'results.csv'
    if content is not None:
        csv.write_text(content)
    with pytest.raises(ResultsError):
        read_results(csv)


def test_checkpoint_evaluation_selects_threshold_on_validation_only(small_dataset, tmp_path):
    from training.evaluate_checkpoint import evaluate

    weights = tmp_path / 'best.pt'
    weights.write_bytes(b'checkpoint bytes for digest only')
    calls = []
    table = {
        ('val', 0.25): ((0.30, 0.40, 0.35, 0.20), [0.30, 0.40, 0.20]),
        ('val', 0.5): ((0.50, 0.40, 0.45, 0.25), [0.40, 0.50, 0.30]),
        ('val', 0.65): ((0.60, 0.20, 0.30, 0.15), [0.50, 0.20, 0.10]),
        # Test peaks at a different threshold; selection must ignore it.
        ('test', 0.25): ((0.90, 0.90, 0.95, 0.90), [0.9, 0.9, 0.9]),
        ('test', 0.5): ((0.40, 0.30, 0.35, 0.20), [0.4, 0.3, 0.2]),
        ('test', 0.65): ((0.20, 0.10, 0.15, 0.05), [0.2, 0.1, 0.1]),
    }

    def fake_runner(**kwargs):
        calls.append((kwargs['task'], kwargs['conf_thres']))
        metrics, maps = table[(kwargs['task'], kwargs['conf_thres'])]
        return metrics, maps, (0, 0, 0)

    report = evaluate(
        weights, small_dataset, img_size=320, batch=4, device='cpu',
        thresholds=[0.25, 0.5, 0.65], output=tmp_path / 'evaluation', runner=fake_runner,
    )
    assert report['selected_confidence_threshold'] == 0.5
    assert report['test_at_selected_threshold']['mAP50'] == pytest.approx(0.35)
    assert report['splits']['test']['0.25']['mAP50'] == pytest.approx(0.95)
    assert report['splits']['val']['0.5']['per_class_mAP50_95'] == {'Helmet': 0.4, 'No_Helmet': 0.5, 'License_Plate': 0.3}
    assert report['checkpoint_sha256'] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert {(split, conf) for split, conf in calls} == {('val', c) for c in (0.25, 0.5, 0.65)} | {('test', c) for c in (0.25, 0.5, 0.65)}
    assert (tmp_path / 'evaluation/evaluation_report.json').is_file()


def test_checkpoint_evaluation_rejects_unusable_arguments(small_dataset, tmp_path):
    from training.evaluate_checkpoint import evaluate

    weights = tmp_path / 'best.pt'
    weights.write_bytes(b'checkpoint bytes')
    runner = lambda **kwargs: ((0.1, 0.1, 0.1, 0.1), [0.1, 0.1, 0.1], (0, 0, 0))
    with pytest.raises(ValueError, match='Checkpoint missing'):
        evaluate(tmp_path / 'absent.pt', small_dataset, output=tmp_path / 'a', runner=runner)
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        evaluate(weights, small_dataset, thresholds=[1.5], output=tmp_path / 'b', runner=runner)
    with pytest.raises(ValueError, match='distinct'):
        evaluate(weights, small_dataset, thresholds=[0.5, 0.5], output=tmp_path / 'c', runner=runner)
    with pytest.raises(ValueError, match='multiple of 32'):
        evaluate(weights, small_dataset, img_size=100, output=tmp_path / 'd', runner=runner)


def test_credited_public_footage_matches_the_demo_workflow():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / '.github/workflows/case-study-training.yml').read_text())
    dispatch = workflow[True]['workflow_dispatch']['inputs']
    url = dispatch['demo-video-url']['default']
    credit_path = root / dispatch['demo-video-license']['default']
    credit = credit_path.read_text()
    assert url.startswith('https://upload.wikimedia.org/wikipedia/commons/')
    assert url in credit and 'CC BY-SA 4.0' in credit and 'Karel Bilek' in credit
    steps = ''.join(step.get('run', '') for job in workflow['jobs'].values() for step in job['steps'])
    assert 'install-model' not in steps
    assert '--device cpu' in steps and 'timeout "${BUDGET_MINUTES}m"' in steps
    assert '--source-license "${{ inputs.demo-video-license }}"' in steps
