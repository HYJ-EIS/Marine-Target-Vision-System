from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.evaluation.msdc_dataset_benchmark import VideoInfo, write_motchallenge_gt_sequence


def spec(tmp_path, labels, labels_at_root):
    source = tmp_path / 'source'
    (source / 'gt').mkdir(parents=True)
    gt = source / 'gt/gt.txt'
    gt.write_text('1,1,10,10,20,20,1,1,1\n', encoding='utf-8')
    labels_file = (source if labels_at_root else source / 'gt') / 'labels.txt'
    labels_file.write_text(labels, encoding='utf-8')
    return SimpleNamespace(seq_name='synthetic', gt_file=gt, labels_file=labels_file)


@pytest.mark.parametrize('labels_at_root', [True, False])
def test_gt_export_preserves_numeric_classes_and_writes_global_labels(tmp_path, labels_at_root):
    dataset = spec(tmp_path, 'USV\nfishship\nUAV\n', labels_at_root)
    original = dataset.gt_file.read_bytes()
    result = write_motchallenge_gt_sequence(dataset, tmp_path / 'out', VideoInfo(25, 100, 100, 1))
    assert (result / 'gt/gt.txt').read_bytes() == original
    assert (result / 'gt/labels.txt').read_text().splitlines() == ['USV', 'fishship', 'UAV']


@pytest.mark.parametrize('labels_at_root', [True, False])
def test_gt_export_rejects_local_label_order_before_copying(tmp_path, labels_at_root):
    dataset = spec(tmp_path, 'UAV\nUSV\nfishship\n', labels_at_root)
    with pytest.raises(ValueError, match='labels'):
        write_motchallenge_gt_sequence(dataset, tmp_path / 'out', VideoInfo(25, 100, 100, 1))
    assert not (tmp_path / 'out').exists()
