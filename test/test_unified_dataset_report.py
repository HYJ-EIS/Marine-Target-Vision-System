import json

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluation.class_mapping import class_names
from tools.evaluation.unified_dataset_report import detection_coverage, extract_metrics, speed_row


def test_native_metrics_scale_threshold_average_and_counts():
    row = {k: '0.5' for k in ('MOTA', 'MOTP', 'IDF1', 'IDP', 'IDR')}
    row['MOTA'] = '-0.2'
    for key in ('HOTA', 'DetA', 'AssA', 'LocA'):
        row.update({key + '___5': '0.2', key + '___10': '0.6', key + '___AUC': '0.4'})
    for key in ('IDSW', 'CLR_TP', 'CLR_FP', 'CLR_FN', 'Frag', 'IDTP', 'IDFP', 'IDFN'):
        row[key] = '12'
    result = extract_metrics(row)
    assert result['HOTA'] == pytest.approx(40)
    assert result['MOTA'] == -20
    assert result['MOTP'] == 50
    assert result['TP'] == 12


def test_pooled_speed_uses_total_frames_over_total_time():
    def group(n, d, t, r):
        return [n, {'total_time_s': d}, {'total_time_s': t, 'stage_seconds': {'tracker': t / 2}}, {'total_time_s': r}]
    row = speed_row('test', [group(100, 1, 2, 2), group(900, 10, 10, 25)])
    assert row['Total Time (s)'] == 50
    assert row['Overall FPS'] == 20
    assert row['Avg Time (ms/frame)'] == 50
    assert row['Tracker Time (s)'] == 6
    assert row['Tracker FPS'] == pytest.approx(1000 / 6)


def _coverage_inputs(tmp_path, detections, gt_rows=None, labels=None):
    gt = tmp_path / 'annotations' / 'gt.txt'
    gt.parent.mkdir()
    gt.write_text(gt_rows or '1,1,10,10,20,20,1,1,1\n', encoding='utf-8')
    (gt.parent / 'labels.txt').write_text(
        '\n'.join(class_names() if labels is None else labels) + '\n', encoding='utf-8',
    )
    cache = tmp_path / 'detections' / 'mini_high_low_detections.jsonl'
    cache.parent.mkdir()
    cache.write_text(json.dumps({'frame_id': 1, 'low_boxes': detections}) + '\n', encoding='utf-8')
    return tmp_path, {'sequence': 'mini', 'gt': str(gt)}


@pytest.mark.parametrize('prediction, expected_hits', [
    ({'class': 'USV'}, 1),
    ({'class_id': 0}, 1),
    ({'mot_class_id': 1}, 1),
    ({'class': 'fishship'}, 0),
    ({'class': 'UAV'}, 0),
    ({'class': 'unknown'}, 0),
    ({}, 0),
    ({'class': 'USV', 'class_id': 2}, 0),
])
def test_detection_coverage_requires_correct_class_at_same_box(tmp_path, prediction, expected_hits):
    detection = {'x': 10, 'y': 10, 'w': 20, 'h': 20, **prediction}
    root, item = _coverage_inputs(tmp_path, [detection])

    rows = detection_coverage(root, item)

    assert len(rows) == 1
    assert rows[0]['GT class'] == 'USV'
    assert rows[0]['GT boxes with a cached detection at IoU>=0.5'] == expected_hits
    assert rows[0]['Coverage (%)'] == 100 * expected_hits
    assert rows[0]['GT boxes without overlap'] == 1 - expected_hits


def test_detection_coverage_retains_nonexclusive_spatial_coverage(tmp_path):
    root, item = _coverage_inputs(
        tmp_path,
        [{'x': 10, 'y': 10, 'w': 20, 'h': 20, 'class': 'USV'}],
        gt_rows='1,1,10,10,20,20,1,1,1\n1,2,11,10,20,20,1,1,1\n',
    )

    rows = detection_coverage(root, item)

    assert rows[0]['GT boxes'] == 2
    assert rows[0]['GT boxes with a cached detection at IoU>=0.5'] == 2


def test_detection_coverage_requires_global_gt_label_order(tmp_path):
    root, item = _coverage_inputs(tmp_path, [], labels=('UAV', 'USV', 'fishship'))

    with pytest.raises(ValueError, match='configured class order'):
        detection_coverage(root, item)
