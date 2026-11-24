"""Post-process existing MOT results; report longest continuous same-ID coverage.

This is an explicitly defined proxy, not a standardized maritime metric.
Only positive-marked GT frames contribute to observable duration. Matching uses
the run's CLEAR convention (IoU >= 0.5, prior-ID preference), with class
equality when recorded in evaluation_protocol.json. Older runs retain class-agnostic matching.
Every missing GT, missing match, or changed predicted ID breaks a segment.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def protocol_requires_class_matching(eval_root):
    """Use the saved run protocol; absent metadata keeps historical semantics."""
    path = Path(eval_root) / "evaluation_protocol.json"
    if not path.exists():
        return False
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = protocol.get("class_consistency_required", False)
    if not isinstance(required, bool):
        raise ValueError(f"Invalid class matching protocol: {path}")
    return required


def validate_class_matching_inputs(gt_path, pred_path):
    """Class-aware runs require the same validated classes as main evaluation."""
    from tools.evaluation.class_aware_mot import validate_gt_labels, validate_mot_class_file
    validate_gt_labels(gt_path)
    validate_mot_class_file(gt_path)
    validate_mot_class_file(pred_path)


def read_mot(path, gt=False):
    frames = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            if not row or (gt and float(row[6]) == 0):
                continue
            frame, identity = int(row[0]), int(row[1])
            if any(r[0] == identity for r in frames[frame]):
                raise ValueError(f"Duplicate frame/ID in {path}: {frame}/{identity}")
            frames[frame].append((identity, np.array(row[2:6], dtype=float),
                                  (int(row[7]) if gt else int(float(row[7]))) if len(row) >= 8 else -1))
    return frames


def match_frames(gt, pred, length, threshold=.5, require_class_match=False):
    """Mirror vendored CLEAR matching, retaining original ID numbers."""
    previous = {}
    matches = {}
    for frame in range(1, length + 1):
        gs, ps = gt[frame], pred[frame]
        if not gs or not ps:
            continue
        g = np.array([r[1] for r in gs])
        p = np.array([r[1] for r in ps])
        intersection = np.maximum(0, np.minimum(g[:, None, :2] + g[:, None, 2:],
                                                p[None, :, :2] + p[None, :, 2:])
                                  - np.maximum(g[:, None, :2], p[None, :, :2])).prod(2)
        union = g[:, 2:].prod(1)[:, None] + p[:, 2:].prod(1)[None, :] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        if require_class_match:
            same_class = np.array([[a[2] == b[2] for b in ps] for a in gs])
            iou = np.where(same_class, iou, 0.0)
        bonus = np.array([[previous.get(a[0]) == b[0] for b in ps] for a in gs])
        score = iou + 1000 * bonus
        score[iou < threshold - np.finfo(float).eps] = 0
        rows, cols = linear_sum_assignment(-score)
        previous = {}
        for a, b in zip(rows, cols):
            if score[a, b] > np.finfo(float).eps:
                matches[frame, gs[a][0]] = ps[b][0]
                previous[gs[a][0]] = ps[b][0]
    return matches


def continuous_segments(frames, matches, identity):
    segments = []
    for frame in sorted(frames):
        predicted = matches.get((frame, identity))
        if predicted is None:
            continue
        if segments and segments[-1]['end_frame'] == frame - 1 and segments[-1]['track_id'] == predicted:
            segments[-1]['end_frame'] = frame
            segments[-1]['frames'] += 1
        else:
            segments.append(dict(start_frame=frame, end_frame=frame, track_id=predicted, frames=1))
    return segments


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_root, output):
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads((run_root / 'inventory.json').read_text())
    metadata = json.loads((run_root / 'run_metadata.json').read_text())
    tracker = metadata['tracker_names']['msdc_elt']
    require_class_match = protocol_requires_class_matching(run_root / "eval")
    protocol_path = run_root / "eval/evaluation_protocol.json"
    with (run_root / 'eval' / tracker / 'pedestrian_detailed.csv').open() as stream:
        reference = {r['seq']: r for r in csv.DictReader(stream)}
    datasets, targets, segments, provenance = [], [], [], []
    for item in inventory:
        sequence = item['sequence']
        source_gt = Path(item['gt'])
        gt_path = run_root / 'mot_gt' / sequence / 'gt/gt.txt'
        pred_path = run_root / 'trackers' / tracker / 'data' / f'{sequence}.txt'
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest(source_gt) == digest(gt_path) == item['gt_sha256'], 'GT changed'
        info = configparser.ConfigParser()
        info.read(run_root / 'mot_gt' / sequence / 'seqinfo.ini')
        fps = float(info['Sequence']['frameRate'])
        length = int(info['Sequence']['seqLength'])
        if require_class_match:
            validate_class_matching_inputs(gt_path, pred_path)
        gt, pred = read_mot(gt_path, gt=True), read_mot(pred_path)
        assert fps > 0 and all(1 <= f <= length for f in set(gt) | set(pred))
        matches = match_frames(gt, pred, length, require_class_match=require_class_match)
        observable = defaultdict(list)
        classes = defaultdict(set)
        for frame, rows in gt.items():
            for identity, _, cls in rows:
                observable[identity].append(frame)
                classes[identity].add(cls)
        labels = (source_gt.parent / 'labels.txt').read_text(encoding='utf-8-sig').splitlines()
        seq_targets = []
        for identity, frames in sorted(observable.items()):
            runs = continuous_segments(frames, matches, identity)
            longest = max(runs, key=lambda s: s['frames']) if runs else None
            stable = longest['frames'] if longest else 0
            count = len(frames)
            row = dict(sequence=sequence, gt_id=identity,
                       gt_class='|'.join(labels[c - 1] for c in sorted(classes[identity])),
                       observable_frames=count, observable_seconds=count/fps,
                       matched_frames=sum(s['frames'] for s in runs),
                       longest_frames=stable, longest_seconds=stable/fps,
                       longest_percent=100*stable/count,
                       longest_start=longest['start_frame'] if longest else '',
                       longest_end=longest['end_frame'] if longest else '',
                       longest_track_id=longest['track_id'] if longest else '',
                       segment_count=len(runs))
            targets.append(row)
            seq_targets.append(row)
            segments.extend(dict(sequence=sequence, gt_id=identity, **s, seconds=s['frames']/fps) for s in runs)
        tp = len(matches)
        n_gt = sum(len(rows) for rows in gt.values())
        n_pred = sum(len(rows) for rows in pred.values())
        last_id, switches = {}, 0
        for (frame, identity), predicted in sorted(matches.items()):
            switches += identity in last_id and last_id[identity] != predicted
            last_id[identity] = predicted
        check = dict(CLR_TP=tp, CLR_FN=n_gt-tp, CLR_FP=n_pred-tp, IDSW=switches)
        assert all(v == int(float(reference[sequence][k])) for k, v in check.items()), check
        stable = sum(r['longest_frames'] for r in seq_targets)
        datasets.append(dict(sequence=sequence, fps=fps, video_frames=length, gt_targets=len(seq_targets),
                             class_consistency_required=require_class_match,
                             observable_frames=n_gt, observable_seconds=n_gt/fps,
                             longest_frames=stable, longest_seconds=stable/fps,
                             longest_percent=100*stable/n_gt, matched_percent=100*tp/n_gt, **check))
        provenance.append(dict(sequence=sequence, gt=str(gt_path), predictions=str(pred_path),
                               gt_sha256=digest(gt_path), predictions_sha256=digest(pred_path),
                               class_consistency_required=require_class_match,
                               evaluation_protocol=str(protocol_path) if protocol_path.exists() else None,
                               evaluation_protocol_sha256=digest(protocol_path) if protocol_path.exists() else None,
                               clear_counts_verified=True))
    total_observable = sum(r['observable_seconds'] for r in datasets)
    total_stable = sum(r['longest_seconds'] for r in datasets)
    report = dict(analysis_type='postprocess_existing_run_not_new_benchmark', source_run=str(run_root),
                  variant=metadata['msdc_variant'],
                  matching=('class-consistent CLEAR, IoU >= 0.5' if require_class_match else 'historical class-agnostic CLEAR, IoU >= 0.5'),
                  class_consistency_required=require_class_match,
                  definition='Sum of each GT ID longest consecutive-frame same-predicted-ID matched segment / positive GT target-time',
                  denominator_limit='Unannotated frames excluded; visibility and annotation completeness not verified',
                  class_limit=('Same-class predictions required by saved evaluation protocol' if require_class_match else 'Historical matching ignores prediction classes'),
                  missing_gt_breaks_segment=True, minimum_stable_duration_seconds=None,
                  observable_seconds=total_observable, longest_seconds=total_stable,
                  pooled_longest_percent=100*total_stable/total_observable,
                  macro_longest_percent=sum(r['longest_percent'] for r in datasets)/len(datasets),
                  datasets=datasets, targets=targets, provenance=provenance)
    write_csv(output / 'datasets.csv', datasets)
    write_csv(output / 'targets.csv', targets)
    write_csv(output / 'segments.csv', segments)
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', required=True, help='New analysis directory; must not exist')
    args = parser.parse_args()
    analyze(args.run_root, args.output)
