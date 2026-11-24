from tools.evaluation.stable_tracking_analysis import continuous_segments, match_frames
from collections import defaultdict
import numpy as np


def test_breaks_on_unmatched_changed_id_and_missing_gt():
    frames = [1, 2, 3, 4, 5, 6, 8]
    matches = {(1, 1): 10, (2, 1): 10, (4, 1): 10, (5, 1): 11,
               (6, 1): 11, (8, 1): 11}
    runs = continuous_segments(frames, matches, 1)
    assert [r['frames'] for r in runs] == [2, 1, 2, 1]
    assert [r['start_frame'] for r in runs] == [1, 4, 5, 8]


def test_no_matches_has_no_stable_segment():
    assert continuous_segments([1, 2, 3], {}, 1) == []


def test_matching_is_one_to_one_and_rejects_wrong_location():
    box = np.array([0., 0., 10., 10.])
    distant = np.array([100., 100., 10., 10.])
    gt = defaultdict(list, {1: [(1, box, 1), (2, distant, 1)]})
    pred = defaultdict(list, {1: [(10, box, -1), (11, box, -1)]})
    matches = match_frames(gt, pred, 1)
    assert len(matches) == 1 and (1, 1) in matches


def test_previous_identity_preference_still_requires_iou_threshold():
    box = np.array([0., 0., 10., 10.])
    near = np.array([1., 0., 10., 10.])
    far = np.array([100., 0., 10., 10.])
    gt = defaultdict(list, {f: [(1, box, 1)] for f in (1, 2, 3)})
    pred = defaultdict(list, {1: [(10, box, -1)],
                              2: [(10, near, -1), (11, box, -1)],
                              3: [(10, far, -1), (11, box, -1)]})
    assert match_frames(gt, pred, 3) == {(1, 1): 10, (2, 1): 10, (3, 1): 11}


def test_class_matching_gates_iou_before_prior_identity_bonus():
    box = np.array([0., 0., 10., 10.])
    gt = defaultdict(list, {f: [(1, box, 1)] for f in (1, 2)})
    pred = defaultdict(list, {
        1: [(10, box, 1)],
        2: [(10, box, 3), (20, box, 1)],
    })
    assert match_frames(gt, pred, 2, require_class_match=True) == {(1, 1): 10, (2, 1): 20}
    assert match_frames(gt, pred, 2) == {(1, 1): 10, (2, 1): 10}


def test_read_mot_keeps_exported_class_and_legacy_placeholder(tmp_path):
    from tools.evaluation.stable_tracking_analysis import read_mot
    path = tmp_path / 'prediction.txt'
    path.write_text('1,10,0,0,10,10,.9,3,-1,-1\n2,10,0,0,10,10,.9,-1,-1,-1\n')
    rows = read_mot(path)
    assert rows[1][0][2] == 3
    assert rows[2][0][2] == -1


def test_class_protocol_is_optional_for_historical_runs(tmp_path):
    import json
    from tools.evaluation.stable_tracking_analysis import protocol_requires_class_matching
    assert protocol_requires_class_matching(tmp_path) is False
    path = tmp_path / 'evaluation_protocol.json'
    path.write_text(json.dumps({'class_consistency_required': True}))
    assert protocol_requires_class_matching(tmp_path) is True
    path.write_text(json.dumps({'class_consistency_required': False}))
    assert protocol_requires_class_matching(tmp_path) is False


def test_stable_analysis_uses_class_protocol_when_every_prediction_is_wrong_class(tmp_path):
    import csv
    import hashlib
    import json
    from tools.evaluation.stable_tracking_analysis import analyze
    from target_module.image_detect_module.config import Config
    source = tmp_path / 'source' / 'gt'
    source.mkdir(parents=True)
    source_gt = source / 'gt.txt'
    source_gt.write_text('1,1,10,10,20,20,1,1,1\n2,1,10,10,20,20,1,1,1\n')
    (source / 'labels.txt').write_text('\n'.join(Config.CLASSES) + '\n')
    run = tmp_path / 'run'
    gt = run / 'mot_gt/seq/gt'
    gt.mkdir(parents=True)
    (gt / 'gt.txt').write_bytes(source_gt.read_bytes())
    (gt / 'labels.txt').write_bytes((source / 'labels.txt').read_bytes())
    (gt.parent / 'seqinfo.ini').write_text('[Sequence]\nframeRate=2\nseqLength=2\n')
    pred = run / 'trackers/msdc_v3_replay/data/seq.txt'
    pred.parent.mkdir(parents=True)
    pred.write_text('1,10,10,10,20,20,.9,3,-1,-1\n2,10,10,10,20,20,.9,3,-1,-1\n')
    ref = run / 'eval/msdc_v3_replay/pedestrian_detailed.csv'
    ref.parent.mkdir(parents=True)
    with ref.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['seq', 'CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW'])
        writer.writeheader()
        writer.writerow(dict(seq='seq', CLR_TP=0, CLR_FN=2, CLR_FP=2, IDSW=0))
    (run / 'eval/evaluation_protocol.json').write_text(json.dumps({'class_consistency_required': True}))
    (run / 'inventory.json').write_text(json.dumps([
        dict(sequence='seq', gt=str(source_gt), gt_sha256=hashlib.sha256(source_gt.read_bytes()).hexdigest())
    ]))
    (run / 'run_metadata.json').write_text(json.dumps(
        dict(msdc_variant='synthetic', tracker_names={'msdc_elt': 'msdc_v3_replay'})
    ))
    output = tmp_path / 'analysis'
    analyze(run, output)
    report = json.loads((output / 'analysis.json').read_text())
    assert report['class_consistency_required'] is True
    assert report['datasets'][0]['CLR_TP'] == report['targets'][0]['matched_frames'] == 0
    assert report['provenance'][0]['evaluation_protocol_sha256']
    assert (output / 'segments.csv').exists()
