import pytest

from tools.evaluation.tracking_duration_report import summarize_target


def test_missing_match_and_switch_have_different_counts():
    matches = {(1, 1): 10, (2, 1): 10, (4, 1): 10,
               (5, 1): 11, (6, 1): 11}
    row, runs = summarize_target(range(1, 7), matches, 1, fps=2, minimum_seconds=1)
    assert row['effective_seconds'] == 2.5
    assert row['observable_seconds'] == 3
    assert row['longest_seconds'] == 1
    assert row['sustained_seconds'] == 2
    assert row['missed_then_recovered_count'] == 1
    assert row['identity_switches'] == row['direct_identity_switches'] == 1
    assert [s['frames'] for s in runs] == [2, 1, 2]


def test_annotation_gap_is_not_a_miss_or_a_recovery():
    row, _ = summarize_target([1, 2, 7, 8], {(1, 1): 10, (2, 1): 10,
                             (7, 1): 10, (8, 1): 11}, 1, fps=2)
    assert row['observable_seconds'] == row['effective_seconds'] == 2
    assert row['longest_seconds'] == 1
    assert row['missed_then_recovered_count'] == 0
    assert row['longest_missed_seconds'] == 0
    assert row['identity_switches'] == 1


def test_never_tracked_target_does_not_get_zero_confirmation_delay():
    row, runs = summarize_target([2, 3, 4, 5], {}, 1, fps=2)
    assert row['first_match_delay_seconds'] is None
    assert row['longest_missed_seconds'] == row['observable_seconds'] == 2
    assert row['effective_seconds'] == row['sustained_seconds'] == 0
    assert row['segment_count'] == 0 and runs == []


def test_noninteger_frame_rate_and_distinct_targets():
    matches = {(f, 1): 5 for f in range(1, 31)}
    matches.update({(f, 2): 6 for f in range(1, 30)})
    a, _ = summarize_target(range(1, 31), matches, 1, fps=29.97)
    b, _ = summarize_target(range(1, 31), matches, 2, fps=29.97)
    assert a['sustained_seconds'] == pytest.approx(30/29.97)
    assert b['sustained_seconds'] == 0
    assert a['effective_seconds'] + b['effective_seconds'] == pytest.approx(59/29.97)


def test_initial_and_terminal_miss_are_not_recovered_gaps():
    row, _ = summarize_target(range(1, 9), {(3, 1): 5, (4, 1): 5}, 1, fps=2)
    assert row['first_match_delay_seconds'] == 1
    assert row['missed_then_recovered_count'] == 0
    assert row['longest_missed_seconds'] == 2


@pytest.mark.parametrize('require_class_match', [False, True])
def test_analyze_record_uses_saved_class_protocol_and_preserves_historical_matching(tmp_path, require_class_match):
    import csv
    import json
    from target_module.image_detect_module.config import Config
    from tools.evaluation.tracking_duration_report import analyze_record
    run = tmp_path / 'run'
    gt_root = run / 'mot_gt/seq'
    (gt_root / 'gt').mkdir(parents=True)
    (gt_root / 'seqinfo.ini').write_text('[Sequence]\nframeRate=2\nseqLength=2\n')
    (gt_root / 'gt/gt.txt').write_text('1,1,10,10,20,20,1,1,1\n2,1,10,10,20,20,1,1,1\n')
    (gt_root / 'gt/labels.txt').write_text('\n'.join(Config.CLASSES) + '\n')
    pred = run / 'trackers/msdc_v3_replay/data/seq.txt'
    pred.parent.mkdir(parents=True)
    pred.write_text('1,10,10,10,20,20,.9,3,-1,-1\n2,10,10,10,20,20,.9,3,-1,-1\n')
    ref = run / 'eval/msdc_v3_replay/pedestrian_detailed.csv'
    ref.parent.mkdir(parents=True)
    if require_class_match:
        (run / 'eval/evaluation_protocol.json').write_text(json.dumps({'class_consistency_required': True}))
    tp = 0 if require_class_match else 2
    ratio = 0 if require_class_match else 1
    reference = dict(
        seq='seq', CLR_TP=tp, CLR_FN=2-tp, CLR_FP=2-tp, IDSW=0, GT_Dets=2, Dets=2,
        MOTA=-1 if require_class_match else 1, IDF1=ratio, IDP=ratio, IDR=ratio,
        MOTP=ratio, HOTA___AUC=ratio, Frag=0, IDTP=tp, IDFP=2-tp, IDFN=2-tp,
    )
    with ref.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reference))
        writer.writeheader()
        writer.writerow(reference)
    row, targets, segments, provenance = analyze_record((run, ref, reference, gt_root, pred), 1.0)
    assert row['class_consistency_required'] is require_class_match
    assert row['CLR_TP'] == tp
    assert row['effective_seconds'] == tp / 2
    assert targets[0]['effective_frames'] == tp
    assert bool(provenance['evaluation_protocol_sha256']) is require_class_match
