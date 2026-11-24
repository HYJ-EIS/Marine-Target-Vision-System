"""Read-only statistics for the fixed visible-light MOT experiment contract.

No detector, tracker, renderer or video reader is invoked here. Official metrics
come from native TrackEval CSVs; output correspondences mirror vendored CLEAR.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import bisect
import configparser
import csv
import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from tools.evaluation.class_aware_mot import validate_gt_labels, validate_mot_class_file
from tools.evaluation.class_mapping import normalize_prediction_class
from tools.evaluation.stable_tracking_analysis import read_mot, match_frames


NA = None
STAGES = ('read_decode', 'low_detection', 'high_split', 'tracker', 'render', 'write_export')
METRICS = ('HOTA', 'DetA', 'AssA', 'LocA', 'MOTA', 'MOTP', 'IDF1', 'IDP', 'IDR',
           'IDSW', 'TP', 'FP', 'FN', 'Frag', 'IDTP', 'IDFP', 'IDFN')
COUNTS = {'IDSW', 'TP', 'FP', 'FN', 'Frag', 'IDTP', 'IDFP', 'IDFN'}


def ratio(numerator, denominator, scale=1.0):
    return scale * numerator / denominator if denominator else NA


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv_value(value):
    if value is None:
        return 'N/A'
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(sorted(value) if isinstance(value, set) else value, ensure_ascii=False)
    return value


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = list(fields or dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ['sequence', 'method']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


class StreamingCSV:
    def __init__(self, stack, path, fields):
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = stack.enter_context(path.open('w', encoding='utf-8-sig', newline=''))
        self.writer = csv.DictWriter(fh, fieldnames=fields)
        self.writer.writeheader()
        self.fields = fields

    def emit(self, row):
        self.writer.writerow({key: csv_value(row.get(key)) for key in self.fields})


def iou_matrix(first, second):
    """Input rows are (identity, xywh ndarray, one-based class)."""
    if not first or not second:
        return np.zeros((len(first), len(second)), dtype=float)
    a = np.asarray([row[1] for row in first], dtype=float)
    b = np.asarray([row[1] for row in second], dtype=float)
    overlap = np.maximum(0, np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
                         - np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    union = np.maximum(a[:, 2:], 0).prod(axis=1)[:, None] + np.maximum(b[:, 2:], 0).prod(axis=1)[None, :] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def geometric_matching(gt_rows, prediction_rows, threshold=.5):
    """Maximum cardinality, then total IoU; same-class edges only."""
    overlaps = iou_matrix(gt_rows, prediction_rows)
    classes_equal = np.asarray([row[2] for row in gt_rows])[:, None] == np.asarray([row[2] for row in prediction_rows])[None, :]
    valid = classes_equal & (overlaps >= threshold)
    if not overlaps.size:
        return []
    bonus = max(len(gt_rows), len(prediction_rows)) + 1
    rows, cols = linear_sum_assignment(-np.where(valid, bonus + overlaps, 0))
    return [(int(a), int(b), float(overlaps[a, b])) for a, b in zip(rows, cols) if valid[a, b]]


def box_record(record, identity):
    """Normalize the explicit project observation/event schemas to MOT tuples."""
    if 'model_class_id' in record:
        cls = normalize_prediction_class({'class_id': record['model_class_id']})
    else:
        source = dict(record)
        if 'class_name' in source and 'class' not in source:
            source['class'] = source['class_name']
        cls = normalize_prediction_class(source)
    xyxy = record.get('box_xyxy', record.get('obs_box_xyxy', record.get('exported_box_xyxy')))
    if xyxy is not None:
        x1, y1, x2, y2 = map(float, xyxy)
        box = np.array([x1, y1, x2-x1, y2-y1])
    elif all(key in record for key in ('x', 'y', 'w', 'h')):
        box = np.array([float(record[key]) for key in ('x', 'y', 'w', 'h')])
    else:
        raise ValueError(f'Missing explicitly formatted box in {record}')
    return identity, box, cls


def reference_availability(cache_path, gt, length):
    available = defaultdict(set)
    with Path(cache_path).open(encoding='utf-8') as stream:
        frames = 0
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row['frame_id'])
            frames += 1
            if frame != frames or frame > length:
                raise ValueError(f'Noncontiguous detection cache: {cache_path}, frame {frame}')
            if 'low_boxes' not in row:
                raise ValueError(f'Expected low_boxes in shared cache {cache_path}:{frame}')
            boxes = [box_record(box, i) for i, box in enumerate(row['low_boxes'])]
            for gi, _, _ in geometric_matching(gt.get(frame, []), boxes):
                available[gt[frame][gi][0]].add(frame)
    if frames != length:
        raise ValueError(f'Detection cache length {frames} != {length}: {cache_path}')
    return available


def consecutive_runs(values):
    result = []
    for value in sorted(set(values)):
        if result and value == result[-1][1] + 1:
            result[-1] = result[-1][0], value
        else:
            result.append((value, value))
    return result


def observed_window(valid_frames, t0, window, sequence_length):
    if t0 is None:
        return {'actual_observation_length_frames': 0, 'observed_end_frame': None, 'censor_reason': 'no_detection'}
    count = 0
    while count < window and t0+count <= sequence_length and t0+count in valid_frames:
        count += 1
    reason = None if count == window else 'sequence_tail' if t0+count > sequence_length else 'gt_gap'
    return {'actual_observation_length_frames': count, 'observed_end_frame': t0+count-1 if count else None, 'censor_reason': reason}


def complete_recovery(event):
    return not event['censored'] and not event.get('gap_has_gt_missing', False)


def shared_events(gt, available, sequence, window=40, stable=3, minimum_gap=3, surrounding=3, sequence_length=None):
    by_gt = defaultdict(set)
    for frame, rows in gt.items():
        for identity, _, _ in rows:
            by_gt[identity].add(frame)
    confirmations, recoveries = [], []
    sequence_length = sequence_length if sequence_length is not None else max(gt, default=0)
    for identity, valid_frames in sorted(by_gt.items()):
        detections = available.get(identity, set())
        t0 = min(detections) if detections else None
        observed = observed_window(valid_frames, t0, window, sequence_length)
        confirmations.append({'event_id': f'{sequence}:confirm:{identity}', 'sequence': sequence, 'gt_id': identity,
                              't0': t0, 'no_detection': t0 is None,
                              'first_gt_frame': min(valid_frames), 'last_gt_frame': max(valid_frames),
                              'censored': t0 is not None and observed['actual_observation_length_frames'] < window, **observed})
        # Search between observed detections over the GT's entire lifetime so
        # gaps containing missing GT are explicitly exported, never silently lost.
        missing = set(range(min(valid_frames), max(valid_frames)+1)) - detections
        for a, b in consecutive_runs(missing):
            if b-a+1 < minimum_gap:
                continue
            if not set(range(a-surrounding, a)) <= detections or not set(range(b+1, b+surrounding+1)) <= detections:
                continue
            t0 = b+1
            gt_missing = not set(range(a, b+1)) <= valid_frames
            observed = observed_window(valid_frames, t0, window, sequence_length)
            recoveries.append({'event_id': f'{sequence}:recovery:{identity}:{a}:{b}', 'sequence': sequence,
                               'gt_id': identity, 'gap_start_a': a, 'gap_end_b': b, 'gap_frames': b-a+1,
                               'gap_bin': gap_bin(b-a+1), 't0': t0, 'gap_has_gt_missing': gt_missing,
                               'excluded_reason': 'gt_gap' if gt_missing else observed['censor_reason'],
                               'censored': not gt_missing and observed['actual_observation_length_frames'] < window, **observed})
    return confirmations, recoveries


def gap_bin(frames):
    for low, high in ((3, 10), (11, 40), (41, 80), (81, 121)):
        if low <= frames <= high:
            return f'{low}-{high}'
    return '>121'


def stable_segments(matches, identity, t0, window=40, stable=3):
    segments = []
    current_id = None
    start = None
    for frame in range(t0, t0+window+1):
        public_id = matches.get((frame, identity)) if frame < t0+window else None
        if public_id != current_id:
            if current_id is not None and frame-start >= stable:
                segments.append((start, frame-1, current_id))
            current_id, start = public_id, frame
    return segments


def confirmation_outcome(event, matches, window=40, stable=3):
    row = dict(event)
    identity = event['gt_id']
    target_matches = sorted(frame for frame, gt_id in matches if gt_id == identity)
    lower = event.get('t0')
    if lower is None:
        lower = event.get('first_gt_frame', min(target_matches, default=1))
    upper = event.get('last_gt_frame', max(target_matches, default=lower-1))
    first_correct_lifetime = next((frame for frame in target_matches if lower <= frame <= upper), None)
    lifetime_segments = stable_segments(matches, identity, lower, max(upper-lower+1, 0), stable)
    row.update(first_public_output=first_correct_lifetime,
               first_correct_output=first_correct_lifetime,
               first_stable_correct_output=lifetime_segments[0][0] if lifetime_segments else None)
    if event['no_detection'] or event['censored']:
        return {**row, 'success': None, 'latency_frames': None, 'penalized_latency_frames': None, 'preconfirmation_fn': None}
    t0 = event['t0']
    segments = stable_segments(matches, identity, t0, window, stable)
    first_correct = next((frame for frame in range(t0, t0+window) if (frame, identity) in matches), None)
    first = segments[0][0] if segments else None
    return {**row, 'success': bool(segments), 'window_first_correct_output': first_correct,
            'window_first_stable_correct_output': first, 'latency_frames': first-t0 if first is not None else None,
            'penalized_latency_frames': first-t0 if first is not None else window,
            'preconfirmation_fn': first_correct-t0 if first_correct is not None else window}


def recovery_outcome(event, matches, births, window=40, stable=3, surrounding=3):
    before = [matches.get((frame, event['gt_id'])) for frame in range(event['gap_start_a']-surrounding, event['gap_start_a'])]
    old_id = before[0] if before and before[0] is not None and len(set(before)) == 1 else None
    row = {**event, 'pretracked': old_id is not None, 'old_id': old_id}
    if not complete_recovery(event):
        return {**row, 'old_id_success': None, 'new_public_id_stable': None, 'existing_other_id_stable': None,
                'old_id_latency_frames': None, 'penalized_latency_frames': None}
    segments = stable_segments(matches, event['gt_id'], event['t0'], window, stable)
    old = [segment for segment in segments if segment[2] == old_id] if old_id is not None else []
    row.update(old_id_success=bool(old), old_id_latency_frames=old[0][0]-event['t0'] if old else None,
               penalized_latency_frames=old[0][0]-event['t0'] if old else window)
    if old_id is None:
        return {**row, 'new_public_id_stable': None, 'existing_other_id_stable': None, 'new_public_ids': [],
                'new_public_id_count': None, 'first_new_id_stable_frame': None, 'new_id_latency_frames': None,
                'new_public_id_first_frames': {}, 'first_public_phase': {}}
    other = [segment for segment in segments if segment[2] != old_id]
    new = [segment for segment in other if births[segment[2]] >= event['gap_start_a']]
    ids = sorted({segment[2] for segment in new})
    first = min((segment[0] for segment in new), default=None)
    return {**row, 'new_public_id_stable': bool(new),
            'existing_other_id_stable': any(births[segment[2]] < event['gap_start_a'] for segment in other),
            'new_public_ids': ids, 'new_public_id_count': len(ids), 'first_new_id_stable_frame': first,
            'new_id_latency_frames': first-event['t0'] if first is not None else None,
            'new_public_id_first_frames': {identity: births[identity] for identity in ids},
            'first_public_phase': {identity: 'during_gap' if births[identity] < event['t0'] else 'after_reappearance' for identity in ids}}


def identity_statistics(gt, pred, matches, sequence='', method=''):
    gt_frames = defaultdict(list)
    matched_ids = defaultdict(set)
    birth = {}
    matched_public = set(matches.values())
    for frame, rows in gt.items():
        for identity, _, _ in rows:
            gt_frames[identity].append(frame)
    for frame, rows in pred.items():
        for identity, _, _ in rows:
            birth[identity] = min(frame, birth.get(identity, frame))
    for (_, identity), public in matches.items():
        matched_ids[identity].add(public)
    identity_rows = []
    longest = interruptions = trailing = 0
    for identity, frames in sorted(gt_frames.items()):
        runs = []
        gap = False
        had_match = False
        previous_frame = None
        for frame in sorted(frames):
            if previous_frame is None or frame != previous_frame+1:
                gap = had_match = False
            public = matches.get((frame, identity))
            if public is not None:
                if gap and had_match:
                    interruptions += 1
                gap = False
                had_match = True
                if runs and runs[-1][1] == frame-1 and runs[-1][2] == public:
                    runs[-1] = runs[-1][0], frame, public
                else:
                    runs.append((frame, frame, public))
            else:
                gap = True
            previous_frame = frame
        for a, b in consecutive_runs(frames):
            last = next((f for f in range(b, a-1, -1) if (f, identity) in matches), None)
            trailing += b-last if last is not None else b-a+1
        best = max((b-a+1 for a, b, _ in runs), default=0)
        longest += best
        ids = sorted(matched_ids[identity])
        identity_rows.append({'sequence': sequence, 'method': method, 'gt_id': identity,
                              'matched_public_ids': ids, 'matched_public_id_count': len(ids),
                              'extra_public_ids': max(len(ids)-1, 0), 'ever_correctly_output': bool(ids),
                              'valid_gt_frames': len(frames), 'longest_same_id_frames': best})
    gt_count = sum(len(rows) for rows in gt.values())
    summary = {'unmatched_new_public_ids': len(set(birth)-matched_public),
               'extra_public_ids_sum': sum(row['extra_public_ids'] for row in identity_rows),
               'extra_public_ids_mean': ratio(sum(row['extra_public_ids'] for row in identity_rows), len(identity_rows)),
               'never_correctly_output_gt_count': sum(not row['ever_correctly_output'] for row in identity_rows),
               'correct_coverage': ratio(len(matches), gt_count, 100),
               'longest_same_id_coverage': ratio(longest, gt_count, 100),
               'output_interruptions': interruptions, 'unrecovered_tail_frames': trailing}
    return identity_rows, summary, birth


def duplicate_statistics(gt, pred, length, sequence='', method='', emit=None):
    events, active = [], {}
    count = duplicate_frames = 0
    for frame in range(1, length+1):
        gs, ps = gt.get(frame, []), pred.get(frame, [])
        anchors = geometric_matching(gs, ps)
        used = {pi for _, pi, _ in anchors}
        overlaps = iou_matrix(gs, ps)
        attached = defaultdict(list)
        for pi, prediction in enumerate(ps):
            if pi in used:
                continue
            options = [(float(overlaps[gi, pi]), gs[gi][0], gi, anchor_pi)
                       for gi, anchor_pi, _ in anchors
                       if gs[gi][2] == prediction[2] and overlaps[gi, pi] >= .5]
            if options:
                overlap, identity, gi, anchor_pi = sorted(options, key=lambda item: (-item[0], item[1]))[0]
                attached[identity].append((prediction[0], overlap, ps[anchor_pi][0]))
        for identity in list(active):
            if identity not in attached:
                events.append(active.pop(identity))
        for identity, extras in attached.items():
            public_ids = {public for public, _, _ in extras} | {anchor for _, _, anchor in extras}
            row = {'sequence': sequence, 'method': method, 'frame': frame, 'gt_id': identity,
                   'anchor_public_id': extras[0][2], 'extra_public_ids': [r[0] for r in extras],
                   'extra_ious': [r[1] for r in extras], 'excess_box_count': len(extras)}
            if emit:
                emit(row)
            count += len(extras)
            duplicate_frames += 1
            if identity not in active:
                active[identity] = {'sequence': sequence, 'method': method, 'gt_id': identity,
                                    'start_frame': frame, 'end_frame': frame, 'duration_frames': 0,
                                    'participating_public_ids': set(), 'excess_box_count': 0}
            event = active[identity]
            event['end_frame'] = frame
            event['duration_frames'] += 1
            event['participating_public_ids'].update(public_ids)
            event['excess_box_count'] += len(extras)
    events.extend(active.values())
    for event in events:
        event['participating_public_ids'] = sorted(event['participating_public_ids'])
    return {'duplicate_excess_boxes': count, 'duplicate_gt_frames': duplicate_frames,
            'duplicate_gt_frame_percent': ratio(duplicate_frames, sum(len(rows) for rows in gt.values()), 100),
            'duplicate_events': len(events)}, events


def summarize_confirmation(rows):
    complete = [row for row in rows if not row['no_detection'] and not row['censored']]
    successes = [row for row in complete if row['success']]
    return {'confirmation_opportunities': len(complete), 'confirmation_censored': sum(row['censored'] for row in rows),
            'no_detection_opportunities': sum(row['no_detection'] for row in rows),
            'stable_confirmation_count': len(successes), 'unconfirmed_count': len(complete)-len(successes),
            'stable_confirmation_rate': ratio(len(successes), len(complete)),
            'confirmation_latency_median': float(np.median([row['latency_frames'] for row in successes])) if successes else None,
            'confirmation_penalized_latency_mean': float(np.mean([row['penalized_latency_frames'] for row in complete])) if complete else None,
            'preconfirmation_fn': sum(row['preconfirmation_fn'] for row in complete)}


def summarize_recovery(rows):
    complete = [row for row in rows if complete_recovery(row)]
    pretracked = [row for row in complete if row['pretracked']]
    success = [row for row in complete if row['old_id_success']]
    return {'all_opportunities': len(complete), 'recovery_censored': sum(row['censored'] for row in rows),
            'recovery_gt_gap_excluded': sum(row.get('gap_has_gt_missing', False) for row in rows),
            'recovery_sequence_tail_censored': sum(row['censored'] and row.get('censor_reason') == 'sequence_tail' for row in rows),
            'recovery_gt_window_censored': sum(row['censored'] and row.get('censor_reason') == 'gt_gap' for row in rows),
            'method_pretracked_count': len(pretracked), 'old_id_success_count': len(success),
            'old_id_success_rate': ratio(len(success), len(complete)),
            'method_pretracked_success_rate': ratio(len(success), len(pretracked)),
            'recovery_latency_median': float(np.median([row['old_id_latency_frames'] for row in success])) if success else None,
            'recovery_penalized_latency_mean': float(np.mean([row['penalized_latency_frames'] for row in complete])) if complete else None}


def common_recovery_comparison(method_events, comparison_id):
    """Input maps method -> complete/censored events for one sequence."""
    maps = {method: {row['event_id']: row for row in rows} for method, rows in method_events.items()}
    common = set.intersection(*(set(rows) for rows in maps.values())) if maps else set()
    common = {event for event in common if all(complete_recovery(rows[event]) and rows[event]['pretracked'] for rows in maps.values())}
    summaries, event_rows = [], []
    for method, rows in maps.items():
        selected = [rows[event] for event in sorted(common)]
        joint = Counter(f'{int(row["old_id_success"])}{int(row["new_public_id_stable"])}' for row in selected)
        summary = {**summarize_recovery(list(rows.values())), 'comparison_id': comparison_id, 'method': method,
                   'common_pretracked_count': len(common), 'common_old_id_success_count': sum(row['old_id_success'] for row in selected),
                   'common_old_id_success_rate': ratio(sum(row['old_id_success'] for row in selected), len(common)),
                   'new_id_event_count': sum(row['new_public_id_stable'] for row in selected),
                   'new_id_event_rate': ratio(sum(row['new_public_id_stable'] for row in selected), len(common)),
                   'existing_other_id_event_count': sum(row['existing_other_id_stable'] for row in selected),
                   **{'old_new_joint_' + key: joint[key] for key in ('00', '01', '10', '11')}}
        assert sum(joint.values()) == len(common)
        summaries.append(summary)
        event_rows.extend({**row, 'method': method, 'comparison_id': comparison_id,
                           'in_common_pretracked_set': row['event_id'] in common} for row in rows.values())
    return summaries, event_rows


def extract_metrics(row):
    result = {}
    for metric in METRICS:
        if metric in ('HOTA', 'DetA', 'AssA', 'LocA'):
            values = [float(value) for key, value in row.items() if key.startswith(metric+'___') and key != metric+'___AUC']
            if not values:
                raise ValueError(f'Missing native threshold arrays for {metric}')
            result[metric] = float(np.mean(values))*100
        else:
            key = {'TP': 'CLR_TP', 'FP': 'CLR_FP', 'FN': 'CLR_FN'}.get(metric, metric)
            result[metric] = int(float(row[key])) if metric in COUNTS else float(row[key])*100
    return result


def bootstrap_difference(first, second, iterations=10000, seed=20261030):
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != second.shape or first.ndim != 1 or not first.size:
        raise ValueError('Paired bootstrap requires equally sized nonempty sequence vectors')
    rng = np.random.default_rng(seed)
    differences = first-second
    values = differences[rng.integers(0, len(differences), size=(iterations, len(differences)))].mean(axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    return {'mean_difference': float(differences.mean()), 'ci_low': float(low), 'ci_high': float(high),
            'paired_sequences': len(first), 'iterations': iterations, 'seed': seed}


def frame_log(path, length):
    """Stream the mandatory per-frame records envelope without truncation."""
    path = Path(path)
    with path.open(encoding='utf-8') as stream:
        expected = 1
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            if int(value['frame']) != expected or not isinstance(value.get('records'), list):
                raise ValueError(f'Invalid paper frame log {path}: expected frame {expected}')
            yield value['records']
            expected += 1
        if expected != length+1:
            raise ValueError(f'Incomplete paper frame log {path}: {expected-1} != {length}')


def stable_owner_history(matches, stable=3):
    rows = defaultdict(list)
    for (frame, gt_id), public_id in sorted(matches.items()):
        rows[public_id].append((frame, gt_id))
    owners = {}
    for public_id, assignments in rows.items():
        run = 0
        previous = None
        history = []
        for frame, gt_id in assignments:
            run = run+1 if previous == (frame-1, gt_id) else 1
            if run >= stable and (not history or history[-1][1] != gt_id):
                history.append((frame, gt_id))
            previous = frame, gt_id
        owners[public_id] = history
    return owners


def owner_before(histories, public_id, frame):
    records = histories.get(public_id, [])
    index = bisect.bisect_left([row[0] for row in records], frame)-1
    return records[index][1] if index >= 0 else None


def internal_statistics(log_root, gt, matches, births, length, sequence, method, center_distance=80,
                        first_active_lower_bounds=None):
    """Read observation/action envelopes in lockstep, retaining only GT history."""
    histories = defaultdict(lambda: deque(maxlen=3))
    stable_owners = stable_owner_history(matches)
    action_rows, confirmation_rows, recovery_rows = [], [], []
    first_active = {}
    confirmed = set()
    suppressions = supported_suppressions = 0
    suppressed_gt_frames = set()
    streams = [frame_log(Path(log_root)/(kind+'.jsonl'), length) for kind in ('observations', 'associations', 'events')]
    # The other two mandatory streams are checked one frame at a time too.
    extra_streams = [frame_log(Path(log_root)/(kind+'.jsonl'), length) for kind in ('states', 'outputs')]
    for frame, packed in enumerate(zip(*streams, *extra_streams, strict=True), 1):
        observations, associations, events, states, outputs = packed
        group_boxes = [box_record(row, row['group_id']) for row in observations]
        group_gt = {group_boxes[pi][0]: gt[frame][gi][0] for gi, pi, _ in geometric_matching(gt.get(frame, []), group_boxes)}
        groups = {row['group_id']: row for row in observations}
        updates = {}
        for edge in associations:
            if edge.get('outcome') != 'matched':
                continue
            gid = edge['gid']
            group_id = edge.get('group_id')
            current_gt = group_gt.get(group_id)
            prior = [value for value in histories[gid] if value is not None]
            counts = Counter(prior)
            historical = counts.most_common(1)[0][0] if counts and counts.most_common(1)[0][1] >= 2 else None
            if edge.get('phase') == 'ordinary':
                overlap = float(edge.get('IoU', edge.get('iou', 0)))
                distance = float(edge.get('distance', math.inf))
                independent = overlap < .2 and distance <= center_distance
                if independent:
                    outcome = 'unknown' if historical is None else 'correct' if historical == current_gt else 'wrong'
                    action_rows.append({'sequence': sequence, 'method': method, 'frame': frame, 'gid': gid,
                                        'public_id': edge.get('public_id'), 'group_id': group_id, 'IoU': overlap,
                                        'distance': distance, 'prior_gt': historical, 'current_gt': current_gt, 'outcome': outcome})
            updates[gid] = current_gt
        new_confirmations = []
        for event in events:
            kind = str(event.get('event_type', ''))
            gid = event.get('gid')
            public_id = event.get('public_id')
            if public_id is None:
                public_id = event.get('new_public_id')
            group_id = event.get('group_id')
            if kind in ('NEW_CANDIDATE', 'NEW_LOW_CANDIDATE') and gid is not None:
                updates.setdefault(gid, group_gt.get(group_id))
            if kind in ('CONFIRM_ACTIVE', 'CONFIRM_LOW_ACTIVE') and public_id is not None:
                if public_id not in confirmed and event.get('id_origin', 'new') != 'reused':
                    new_confirmations.append(event)
                confirmed.add(public_id)
            is_recovery = kind in ('LOST_REACQUIRED', 'REMOVED_ID_RECOVERY_CANDIDATE')
            if is_recovery and event.get('accepted', True):
                old_id = event.get('old_public_id')
                if old_id is None:
                    # An accepted Removed restore may come from a signature
                    # outside the current live pool. Its resulting public_id
                    # is the reused identity even without a before snapshot.
                    old_id = public_id
                published = old_id in births and births[old_id] < frame
                old_gt = owner_before(stable_owners, old_id, frame)
                current_gt = group_gt.get(group_id)
                binding = 'correct' if old_gt is not None and old_gt == current_gt else 'cross_gt' if old_gt is not None and current_gt is not None else 'background' if current_gt is None else 'unknown_history'
                recovery_rows.append({'sequence': sequence, 'method': method, 'frame': frame,
                                      'branch': event.get('branch', 'lost' if kind == 'LOST_REACQUIRED' else 'removed'),
                                      'old_public_id': old_id, 'old_gt': old_gt, 'current_gt': current_gt,
                                      'accepted_published_identity': published, 'binding': binding})
            if kind in ('LOST_SPAWN_SUPPRESSED', 'SPAWN_SUPPRESSED_LOST') or (event.get('branch') == 'lost' and 'SUPPRESS' in kind):
                suppressions += 1
                current_gt = group_gt.get(group_id)
                if current_gt is not None:
                    supported_suppressions += 1
                    suppressed_gt_frames.add((frame, current_gt))
        if new_confirmations:
            confirmation_boxes = []
            for event in new_confirmations:
                try:
                    confirmation_boxes.append(box_record(event, event.get('public_id', event.get('new_public_id'))))
                except ValueError:
                    group = groups.get(event.get('group_id'))
                    if group is None:
                        raise ValueError(f'Missing confirmation geometry/class {sequence}/{method}/{frame}')
                    confirmation_boxes.append(box_record(group, event.get('public_id', event.get('new_public_id'))))
            assigned = {pi: gt[frame][gi][0] for gi, pi, _ in geometric_matching(gt.get(frame, []), confirmation_boxes)}
            for index, event in enumerate(new_confirmations):
                public = confirmation_boxes[index][0]
                confirmation_rows.append({'sequence': sequence, 'method': method, 'frame': frame,
                                          'gid': event.get('gid'), 'public_id': public,
                                          'matched_gt': assigned.get(index), 'false_confirmation': index not in assigned,
                                          'ever_published': public in births,
                                          'published_this_frame': any(row.get('public_id') == public and row.get('emitted') for row in outputs)})
        active_records = {record['gid']: record for record in states
                          if str(record.get('state_after', record.get('state', ''))).split('.')[-1].lower() == 'active'}
        for event in events:
            if event.get('event_type') in ('CONFIRM_ACTIVE', 'CONFIRM_LOW_ACTIVE', 'LOST_REACQUIRED'):
                active_records.setdefault(event['gid'], event)
        active_boxes = [box_record(record, gid) for gid, record in active_records.items()]
        for gi, _, _ in geometric_matching(gt.get(frame, []), active_boxes):
            identity = gt[frame][gi][0]
            if frame >= (first_active_lower_bounds or {}).get(identity, 1):
                first_active.setdefault(identity, frame)
        for gid, assigned_gt in updates.items():
            histories[gid].append(assigned_gt)
    a_counts = Counter(row['outcome'] for row in action_rows)
    accepted = [row for row in recovery_rows if row['accepted_published_identity']]
    r_counts = Counter(row['binding'] for row in accepted)
    summary = {'distance_only_pairs': len(action_rows), 'distance_correct': a_counts['correct'],
               'distance_wrong': a_counts['wrong'], 'distance_unknown': a_counts['unknown'],
               'distance_precision': ratio(a_counts['correct'], a_counts['correct']+a_counts['wrong']),
               'new_identity_confirmations': len(confirmation_rows),
               'false_confirmations': sum(row['false_confirmation'] for row in confirmation_rows),
               'false_confirmation_rate': ratio(sum(row['false_confirmation'] for row in confirmation_rows), len(confirmation_rows)),
               'confirmed_never_published': sum(not row['ever_published'] for row in confirmation_rows),
               'accepted_published_reuse': len(accepted), 'correct_reuse_actions': r_counts['correct'],
               'cross_gt_reuse_actions': r_counts['cross_gt'], 'background_reuse_actions': r_counts['background'],
               'unknown_history_reuse_actions': r_counts['unknown_history'],
               'unpublished_reuse_actions': len(recovery_rows)-len(accepted),
               'reuse_precision': ratio(r_counts['correct'], len(accepted)),
               'cross_gt_reuse_rate': ratio(r_counts['cross_gt'], len(accepted)),
               'suppressed_groups': suppressions, 'supported_suppressed_groups': supported_suppressions,
               'suppressed_gt_frames': len(suppressed_gt_frames)}
    for branch in ('lost', 'removed'):
        rows = [row for row in accepted if row['branch'] == branch]
        summary[branch+'_reuse_actions'] = len(rows)
        summary[branch+'_reuse_precision'] = ratio(sum(row['binding'] == 'correct' for row in rows), len(rows))
    return summary, action_rows, confirmation_rows, recovery_rows, first_active


def export_run(run_root, spec_path, phase='core'):
    """Public entry point; import filesystem orchestration only when requested."""
    from tools.evaluation.paper_statistics_run import export_run as run
    return run(run_root, spec_path, phase)
