import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluation import paper_statistics as ps
from tools.evaluation.paper_statistics_run import export_run
from tools.evaluation.motchallenge_eval import run_motchallenge_eval


def box(identity, x=10, cls=1):
    return identity, np.array([x, 10, 20, 20], dtype=float), cls


def test_reference_prefers_maximum_cardinality_before_iou(monkeypatch):
    monkeypatch.setattr(ps, 'iou_matrix', lambda a, b: np.array([[.99, .5], [.5, 0]]))
    assert {(a, b) for a, b, _ in ps.geometric_matching([box(1), box(2)], [box(10), box(20)])} == {(0, 1), (1, 0)}


def test_clear_empty_prediction_frame_keeps_vendored_prior_id():
    from collections import defaultdict
    gt = defaultdict(list, {frame: [box(1)] for frame in (1, 2, 3)})
    pred = defaultdict(list, {1: [box(10)], 3: [box(10, 12), box(20)]})
    matched = ps.match_frames(gt, pred, 3, require_class_match=True)
    assert matched == {(1, 1): 10, (3, 1): 10}


def test_extra_ids_are_distinct_and_zero_match_targets_remain():
    gt = {frame: [box(1), box(2, 60)] for frame in range(1, 4)}
    pred = {1: [box(3)], 2: [box(8)], 3: [box(3)]}
    matched = {(1, 1): 3, (2, 1): 8, (3, 1): 3}
    rows, summary, birth = ps.identity_statistics(gt, pred, matched)
    assert rows[0]['matched_public_ids'] == [3, 8]
    assert rows[0]['extra_public_ids'] == 1
    assert rows[1]['extra_public_ids'] == 0
    assert summary['never_correctly_output_gt_count'] == 1
    assert summary['extra_public_ids_mean'] == .5


def test_duplicates_count_boxes_gt_frames_and_events_separately():
    gt = {frame: [box(1)] for frame in range(1, 4)}
    pred = {1: [box(3), box(8), box(9)], 2: [box(3), box(8)], 3: [box(3)]}
    rows = []
    summary, events = ps.duplicate_statistics(gt, pred, 3, emit=rows.append)
    assert summary['duplicate_excess_boxes'] == 3
    assert summary['duplicate_gt_frames'] == 2
    assert summary['duplicate_events'] == 1
    assert events[0]['duration_frames'] == 2
    assert rows[0]['excess_box_count'] == 2


def test_two_real_targets_and_unanchored_overlaps_are_not_duplicates():
    summary, _ = ps.duplicate_statistics({1: [box(1), box(2, 15)]}, {1: [box(10), box(20, 15)]}, 1)
    assert summary['duplicate_excess_boxes'] == 0
    summary, _ = ps.duplicate_statistics({}, {1: [box(10), box(20)]}, 1)
    assert summary['duplicate_excess_boxes'] == 0


def test_recovery_public_birth_during_gap_and_old_new_can_coexist():
    event = {'event_id': 'e', 'gt_id': 1, 'gap_start_a': 4, 'gap_end_b': 6, 't0': 7, 'censored': False}
    matched = {(frame, 1): 10 for frame in (1, 2, 3, 10, 11, 12)}
    matched.update({(frame, 1): 20 for frame in (7, 8, 9)})
    row = ps.recovery_outcome(event, matched, {10: 1, 20: 5})
    assert row['old_id_success'] is row['new_public_id_stable'] is True
    assert row['first_public_phase'] == {20: 'during_gap'}
    summaries, events = ps.common_recovery_comparison({'Full': [row], '−R': [row]}, 'R_main')
    assert all(row['old_new_joint_11'] == row['common_pretracked_count'] == 1 for row in summaries)


def test_existing_other_id_and_no_new_latency_are_not_new_creation():
    event = {'event_id': 'e', 'gt_id': 1, 'gap_start_a': 4, 'gap_end_b': 6, 't0': 7, 'censored': False}
    matched = {(frame, 1): 10 for frame in (1, 2, 3)}
    matched.update({(frame, 1): 20 for frame in (7, 8, 9)})
    row = ps.recovery_outcome(event, matched, {10: 1, 20: 2})
    assert row['existing_other_id_stable'] is True
    assert row['new_public_id_stable'] is False
    assert row['new_id_latency_frames'] is None
    assert row['first_new_id_stable_frame'] is None
    unknown = ps.recovery_outcome(event, {(frame, 1): 20 for frame in (7, 8, 9)}, {20: 5})
    assert unknown['new_public_id_stable'] is None
    result, _ = ps.common_recovery_comparison({'a': [row], 'b': [unknown]}, 'group')
    assert all(row['common_pretracked_count'] == 0 and row['new_id_event_rate'] is None for row in result)


def test_shared_events_stop_at_gt_gaps_and_censor_short_tail():
    gt = {frame: [box(1)] for frame in range(1, 21)}
    available = {1: set(range(1, 5)) | set(range(8, 21))}
    confirm, recovery = ps.shared_events(gt, available, 's')
    assert confirm[0]['censored'] is True
    assert recovery[0]['gap_start_a'] == 5
    assert recovery[0]['censored'] is True
    del gt[6]
    _, recovery = ps.shared_events(gt, available, 's')
    assert len(recovery) == 1
    assert recovery[0]['gap_has_gt_missing'] is True
    assert ps.summarize_recovery([ps.recovery_outcome(recovery[0], {}, {})])['recovery_gt_gap_excluded'] == 1


def test_confirmation_absolute_times_do_not_use_future_id_birth_or_h40_cutoff():
    event = {'gt_id': 2, 't0': 50, 'first_gt_frame': 50, 'last_gt_frame': 110, 'no_detection': False, 'censored': False}
    # ID10 was on a different GT before this object appeared. Its global birth
    # must never become this object's first public-output time.
    matches = {(frame, 1): 10 for frame in range(1, 10)}
    matches.update({(frame, 2): 10 for frame in range(95, 101)})
    row = ps.confirmation_outcome(event, matches)
    assert row['first_public_output'] == row['first_correct_output'] == 95
    assert row['first_stable_correct_output'] == 95
    assert row['success'] is False
    assert row['penalized_latency_frames'] == 40
    event['censored'] = True
    row = ps.confirmation_outcome(event, matches)
    assert row['first_public_output'] == 95
    assert row['latency_frames'] is None


def test_censor_actual_observation_length_distinguishes_gt_gap_and_video_tail():
    gt = {frame: [box(1)] for frame in range(1, 60) if frame != 6}
    confirm, _ = ps.shared_events(gt, {1: {3}}, 's', sequence_length=60)
    assert confirm[0]['actual_observation_length_frames'] == 3
    assert confirm[0]['censor_reason'] == 'gt_gap'
    gt = {frame: [box(1)] for frame in range(1, 11)}
    confirm, _ = ps.shared_events(gt, {1: {8}}, 's', sequence_length=10)
    assert confirm[0]['actual_observation_length_frames'] == 3
    assert confirm[0]['censor_reason'] == 'sequence_tail'


def test_penalty_and_first_correct_before_stable_are_distinct():
    event = {'gt_id': 1, 't0': 1, 'no_detection': False, 'censored': False}
    matched = {(2, 1): 10, (6, 1): 20, (7, 1): 20, (8, 1): 20}
    row = ps.confirmation_outcome(event, matched)
    assert row['latency_frames'] == 5
    assert row['preconfirmation_fn'] == 1
    row = ps.confirmation_outcome(event, {})
    assert row['penalized_latency_frames'] == 40
    assert row['latency_frames'] is None


def test_bootstrap_is_paired_and_deterministic():
    a = ps.bootstrap_difference([2, 7, 4], [1, 6, 3], iterations=100)
    b = ps.bootstrap_difference([2, 7, 4], [1, 6, 3], iterations=100)
    assert a == b
    assert a['ci_low'] == a['ci_high'] == a['mean_difference'] == 1


def test_statistics_reads_actual_lifecycle_logs(tmp_path):
    from collections import defaultdict
    from target_module.image_detect_module.config import Config
    from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker

    class LocalConfig(Config):
        MSDC_DEBUG_EVENTS = False
        MSDC_LOW_CANDIDATE_ENABLE = False
        MSDC_PENDING_RECOVERY_ENABLE = False
        MSDC_CANDIDATE_CONFIRM_MODE = 'evidence'
        MSDC_CANDIDATE_CONFIRM_SCORE = 2.5

    tracker = MSDCLifecycleTracker(config=LocalConfig)
    tracker.enable_paper_logging(tmp_path, 'synthetic', 'Full')
    gt, pred = defaultdict(list), defaultdict(list)
    detection = {'x': 10, 'y': 10, 'w': 20, 'h': 20, 'confidence': .9, 'class': 'USV'}
    for frame in range(1, 15):
        gt[frame] = [box(1)]
        detections = [] if frame in (5, 6, 7) else [detection]
        emitted = tracker.update(np.zeros((64, 64, 3), np.uint8), frame-1, 'visible', detections, detections)
        pred[frame] = [(row['track_id'], np.array([row[key] for key in ('x', 'y', 'w', 'h')], float), 1) for row in emitted]
    tracker.close_paper_logging()
    matches = ps.match_frames(gt, pred, 14, require_class_match=True)
    _, _, births = ps.identity_statistics(gt, pred, matches)
    summary, actions, confirmations, recovery, first_active = ps.internal_statistics(tmp_path, gt, matches, births, 14, 'synthetic', 'Full')
    assert summary['new_identity_confirmations'] == 1
    assert summary['false_confirmations'] == 0
    assert summary['lost_reuse_actions'] == 1
    assert summary['reuse_precision'] == 1
    assert first_active[1] == 4


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def _csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def synthetic_run(tmp_path):
    # This fixture writes synthetic boxes and evaluator data, never a real video.
    core = [('paper_no_a', '−A'), ('paper_count_e', '−E'), ('paper_no_r', '−R'),
            ('paper_count_e_no_r', '−E−R'), ('paper_full_s0', 'Full,S=0'), ('paper_no_r_s0', '−R,S=0'),
            ('paper_lost_only_s0', 'Lost-only,S=0'), ('paper_removed_only_s0', 'Removed-only,S=0')]
    scans = [(f'paper_assoc_d{value}', f'distance {value}') for value in (40, 60, 100, 120)]
    scans += [(f'paper_candidate_t{int(value*10)}', f'threshold {value}') for value in (1.5, 2, 3, 3.5)]
    spec = {'sequences': [], 'counts': {}, 'core_variants': [name for name, _ in core],
            'sensitivity_variants': [name for name, _ in scans],
            'variants': [{'name': name, 'label': label, 'overrides': {}} for name, label in core+scans],
            'sensitivity': {'association_center_parameter': [40, 60, 80, 100, 120],
                            'candidate_only_confirmation_threshold': [1.5, 2, 2.5, 3, 3.5]},
            'measurement': {'window_frames': 40, 'stable_correct_id_frames': 3,
                            'minimum_detection_gap_frames': 3, 'pre_and_post_gap_detection_frames': 3,
                            'bootstrap': {'iterations': 100, 'seed': 20261030}}}
    sequence, length = 'synthetic', 60
    spec['sequences'] = [{'sequence': sequence}]
    spec['counts'].update(sequences=1, main_runs=4, core_runs=12, sensitivity_runs=8, total_unique_runs=20)
    spec['measurement']['bootstrap']['iterations'] = 100
    spec_path = tmp_path/'spec.json'
    ps.write_json(spec_path, spec)
    run = tmp_path/'run'
    metadata = {'tracker_names': {'msdc_elt': 'msdc_v3_replay', 'bytetrack': 'bytetrack_replay',
                                 'official_ocsort': 'official_ocsort_replay', 'official_botsort': 'official_botsort_replay'}}
    ps.write_json(run/'main/run_metadata.json', metadata)
    ps.write_json(run/'main/inventory.json', [{'sequence': sequence, 'frames': length, 'video': str(tmp_path/'synthetic.mp4')}])
    gt_lines = ''.join(f'{frame},1,10,10,20,20,1,1,1\n' for frame in range(1, length+1))
    for batch in ('main', 'core', 'sensitivity'):
        _write(run/batch/'mot_gt'/sequence/'gt/gt.txt', gt_lines)
        _write(run/batch/'mot_gt'/sequence/'gt/labels.txt', 'USV\nfishship\nUAV\n')
        _write(run/batch/'mot_gt'/sequence/'seqinfo.ini', f'[Sequence]\nname={sequence}\nseqLength={length}\nframeRate=25\n')
    stages = dict.fromkeys(ps.STAGES, .01)
    timing = {'processed_frames': length, 'total_time_s': .1, 'stage_seconds': stages, 'other_seconds': .04}
    detection = {'x': 10, 'y': 10, 'w': 20, 'h': 20, 'confidence': .9, 'class': 'USV'}
    cache = run/'main/detections'/f'{sequence}_high_low_detections.jsonl'
    _write(cache, ''.join(json.dumps({'frame_id': frame, 'high_boxes': [detection] if frame not in (10, 11, 12) else [],
                                     'low_boxes': [detection] if frame not in (10, 11, 12) else []})+'\n' for frame in range(1, length+1)))
    ps.write_json(cache.with_suffix('.timing.json'), timing)
    labels = {v['name']: v['label'] for v in spec['variants']}
    batches = {
        'main': [('Full' if key == 'msdc_elt' else key, tracker, key == 'msdc_elt') for key, tracker in metadata['tracker_names'].items()],
        'core': [(labels[name], name+'_replay', True) for name in spec['core_variants']],
        'sensitivity': [(labels[name], name+'_replay', True) for name in spec['sensitivity_variants']],
    }
    for batch, methods in batches.items():
        for label, tracker, internal in methods:
            no_recovery = label in ('−R', '−E−R', '−R,S=0')
            lines = []
            for frame in range(4, length+1):
                public = 20 if no_recovery and frame >= 13 else 10
                lines.append(f'{frame},{public},10,10,20,20,0.9,1,-1,-1\n')
            mot = run/batch/'trackers'/tracker/'data'/f'{sequence}.txt'
            _write(mot, ''.join(lines))
            ps.write_json(mot.with_suffix('.timing.json'), timing)
            _write(mot.with_suffix('.frames.jsonl'), ''.join(json.dumps({'frame_id': frame})+'\n' for frame in range(1, length+1)))
            video = run/batch/'visualizations'/sequence/f'{tracker}.mp4'
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b'synthetic placeholder; statistics never decode this file')
            ps.write_json(video.with_suffix('.timing.json'), timing)
            if internal:
                logs = {kind: [] for kind in ('observations', 'associations', 'states', 'events', 'outputs')}
                for frame in range(1, length+1):
                    detected = frame not in (10, 11, 12)
                    group = {'group_id': 0, 'box_xyxy': [10, 10, 30, 30], 'model_class_id': 0}
                    observations = [group] if detected else []
                    associations = [{'phase': 'ordinary', 'outcome': 'matched', 'gid': 1, 'public_id': 10,
                                     'group_id': 0, 'IoU': .9, 'distance': 0}] if detected and frame > 1 else []
                    events = []
                    if frame == 1:
                        events.append({**group, 'event_type': 'NEW_CANDIDATE', 'gid': 1, 'public_id': None})
                    if frame == 4:
                        events.append({**group, 'event_type': 'CONFIRM_ACTIVE', 'gid': 1, 'public_id': 10, 'id_origin': 'new'})
                    if frame == 13 and not no_recovery:
                        events.append({**group, 'event_type': 'LOST_REACQUIRED', 'gid': 1, 'public_id': 10,
                                       'old_public_id': 10, 'branch': 'lost', 'accepted': True})
                    for kind, records in (('observations', observations), ('associations', associations), ('events', events),
                                          ('states', []), ('outputs', [{'public_id': 10, 'emitted': True}] if frame >= 4 else [])):
                        logs[kind].append(json.dumps({'seq': sequence, 'method': tracker, 'frame': frame, 'records': records})+'\n')
                for kind, lines in logs.items():
                    _write(run/batch/'trackers'/tracker/'paper_logs'/sequence/(kind+'.jsonl'), ''.join(lines))
        run_motchallenge_eval(run/batch/'mot_gt', run/batch/'trackers', run/batch/('eval' if batch == 'main' else 'eval_all'), print_results=False)
    return run, spec_path


def test_core_and_all_export_from_synthetic_run_preserves_contract(synthetic_run):
    run, spec_path = synthetic_run
    core = export_run(run, spec_path, 'core')
    assert core['unique_method_sequence_results'] == 12
    output = run/'paper_statistics'
    assert len(_csv(output/'main_by_sequence.csv')) == 4
    assert len(_csv(output/'ablation_by_sequence.csv')) == 5
    assert len(_csv(output/'recovery_s0_by_sequence.csv')) == 4
    assert len(_csv(output/'standard_aggregate.csv')) == 12
    aggregate = next(row for row in _csv(output/'standard_aggregate.csv') if row['method'] == 'Full')
    assert float(aggregate['correct_coverage']) == float(aggregate['longest_same_id_coverage']) == 95
    macro = next(row for row in _csv(output/'macro_aggregate.csv') if row['method'] == 'Full')
    assert float(macro['longest_same_id_coverage']) == 95
    r = _csv(output/'r_recovery.csv')
    selected = next(row for row in r if row['comparison_id'] == 'R_main' and row['method'] == '−R')
    assert selected['common_pretracked_count'] == '1'
    assert selected['new_id_event_rate'] == '1.0'
    assert selected['old_new_joint_01'] == '1'
    complete = export_run(run, spec_path, 'all')
    assert complete['unique_method_sequence_results'] == 20
    assert len(_csv(output/'timing_by_sequence.csv')) == 20
    assert len(_csv(output/'sensitivity_distance.csv')) == len(_csv(output/'sensitivity_confirmation.csv')) == 5
    assert (output/'sensitivity_distance.svg').is_file()
    assert (output/'sensitivity_confirmation.png').is_file()
    assert complete['official_counts_verified'] is True
    # Re-export replaces the same keys, never appends duplicate result rows.
    assert len(_csv(output/'main_by_sequence.csv')) == 4


def test_export_rejects_official_clear_count_disagreement(synthetic_run):
    run, spec_path = synthetic_run
    path = run/'main/eval/msdc_v3_replay/pedestrian_detailed.csv'
    rows = _csv(path)
    rows[0]['CLR_TP'] = '0'
    ps.write_csv(path, rows)
    with pytest.raises(ValueError, match='CLEAR correspondence disagrees'):
        export_run(run, spec_path)
