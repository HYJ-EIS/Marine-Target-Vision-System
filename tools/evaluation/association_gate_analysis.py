"""Audit ordinary MSDC association on cached detections (not a benchmark).

Count track/predicted-box--merged-observation pairs, not raw detection boxes.
No detector, video decoder, rendering or accuracy evaluation is performed.
The current lifecycle implementation does not read frame pixels during replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import configparser
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from target_module.image_detect_module.config import Config, MSDC_V3_CONFIG
from target_module.image_detect_module.utils.evidence_state import (
    _iou_batch, _center_distance_batch,
)
from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker
from target_module.image_detect_module.utils.msdc_types import TrackState
from target_module.image_detect_module.utils.tracking_update import update_tracking_for_frame


def gate_masks(iou, distance, iou_threshold, center_threshold):
    a, b = iou >= iou_threshold, distance <= center_threshold
    return {'both': a & b, 'iou_only': a & ~b, 'center_only': ~a & b,
            'neither': ~a & ~b}


class AssociationAudit:
    def __init__(self, updater, writer):
        self.updater, self.writer = updater, writer
        self.original = updater.associate_tracks_to_observations
        self.frame = 0
        self.counts = Counter()
        self.max_iou_pass_distance = None

    def associate(self, tracks, observation_groups, track_indices=None):
        if track_indices is None:
            track_indices = [i for i, t in enumerate(tracks)
                             if TrackState(t.state) in {TrackState.CANDIDATE, TrackState.ACTIVE}]
        self.counts['calls'] += 1
        self.counts['groups_seen'] += len(observation_groups)
        result = self.original(tracks, observation_groups, track_indices)
        matches = set(result[0])
        if not track_indices or not observation_groups:
            return result
        boxes = np.vstack([self.updater._predict_box(tracks[i]) for i in track_indices])
        observations = np.vstack([g.box for g in observation_groups])
        iou, distance = _iou_batch(boxes, observations), _center_distance_batch(boxes, observations)
        it = float(self.updater.config.MSDC_ASSOC_IOU_THRESH)
        ct = float(self.updater.config.MSDC_ASSOC_CENTER_DIST)
        masks = gate_masks(iou, distance, it, ct)
        valid = ~masks['neither']
        self.counts['pairs_tested'] += int(valid.size)
        self.counts['eligible_pairs'] += int(valid.sum())
        # Per-frame observation groups are deduplicated across tracks here.
        iou_group = (masks['both'] | masks['iou_only']).any(axis=0)
        center_group = (masks['both'] | masks['center_only']).any(axis=0)
        self.counts['eligible_groups'] += int((iou_group | center_group).sum())
        self.counts['groups_both'] += int((iou_group & center_group).sum())
        self.counts['groups_iou_only'] += int((iou_group & ~center_group).sum())
        self.counts['groups_center_only'] += int((~iou_group & center_group).sum())
        self.counts['ordinary_matches'] += len(matches)
        for label, mask in masks.items():
            self.counts[label] += int(mask.sum())
        passed_distance = distance[masks['both'] | masks['iou_only']]
        if passed_distance.size:
            largest = float(passed_distance.max())
            self.max_iou_pass_distance = max(self.max_iou_pass_distance or 0, largest)
        bonus = np.maximum(0., 1. - distance / max(ct, 1e-6))
        scores = iou + .5 * bonus
        method = self.updater.config.MSDC_ASSOC_METHOD
        assignment = (self.updater._greedy_gated_assignment if method == 'greedy'
                      else self.updater._hungarian_gated_assignment)
        # Guard against code drift in the scoring expression used for diagnostics.
        assert set(assignment(scores, valid, track_indices)) == matches
        center_gate_matches = set(assignment(scores, distance <= ct, track_indices))
        center_score_matches = set(assignment(.5 * bonus, valid, track_indices))
        self.counts['frames_changed_without_iou_gate'] += center_gate_matches != matches
        self.counts['frames_changed_without_iou_score'] += center_score_matches != matches
        self.counts['pairs_changed_without_iou_score'] += len(matches ^ center_score_matches)
        self.counts['frames_with_competing_pairs'] += bool(
            (valid.sum(axis=0) > 1).any() or (valid.sum(axis=1) > 1).any())
        for a, b in zip(*np.where(valid)):
            track, group = tracks[track_indices[a]], observation_groups[b]
            label = next(k for k in ('both', 'iou_only', 'center_only') if masks[k][a, b])
            selected = (track_indices[a], int(b)) in matches
            self.counts['selected_' + label] += selected
            self.writer.writerow(dict(
                frame_id=self.frame, gid=track.gid, public_id=track.public_id,
                track_state=TrackState(track.state).value, group_index=int(b),
                group_class=group.class_name, group_observation_count=len(group.observations),
                iou=float(iou[a, b]), center_distance=float(distance[a, b]),
                gate=label, selected=int(selected), score=float(scores[a, b])))
        return result


PAIR_FIELDS = ['frame_id', 'gid', 'public_id', 'track_state', 'group_index',
               'group_class', 'group_observation_count', 'iou', 'center_distance',
               'gate', 'selected', 'score']
COUNT_FIELDS = ['calls', 'groups_seen', 'pairs_tested', 'eligible_pairs',
                'both', 'iou_only', 'center_only', 'neither', 'eligible_groups',
                'groups_both', 'groups_iou_only', 'groups_center_only',
                'ordinary_matches', 'selected_both', 'selected_iou_only', 'selected_center_only',
                'frames_with_competing_pairs', 'frames_changed_without_iou_gate',
                'frames_changed_without_iou_score', 'pairs_changed_without_iou_score']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def analyze(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'status.json').write_text('{"status":"partial_diagnostic_analysis"}\n')
    inventory = json.loads((run / 'inventory.json').read_text())
    base = {k: getattr(Config, k) for k in dir(Config)
            if not k.startswith('_') and not callable(getattr(Config, k))}
    base.update(MSDC_V3_CONFIG)
    # Disable disk debug only. The ordinary association and lifecycle are unchanged.
    base['MSDC_DEBUG_EVENTS'] = False
    config = SimpleNamespace(**base)
    (output / 'effective_config.json').write_text(json.dumps(base, indent=2, ensure_ascii=False))
    summaries, provenance = [], []
    for item in inventory:
        sequence = item['sequence']
        cache = run / 'detections' / f'{sequence}_high_low_detections.jsonl'
        history = run / 'trackers/msdc_v3_replay/diagnostics' / sequence / 'msdc_tracks.jsonl'
        assert sha256(cache) == item['cache_sha256'], f'Cache changed: {sequence}'
        assert sha256(item['gt']) == item['gt_sha256'], f'GT changed: {sequence}'
        info = configparser.ConfigParser()
        info.read(run / 'mot_gt' / sequence / 'seqinfo.ini')
        shape = (int(info['Sequence']['imHeight']), int(info['Sequence']['imWidth']), 3)
        frame = np.zeros(shape, dtype=np.uint8)
        tracker = MSDCLifecycleTracker(config=config, debug_dir=output / 'unused_debug')
        started = time.perf_counter()
        mismatches, first_mismatches, frames = 0, [], 0
        with (output / f'{sequence}_pairs.csv').open('w', newline='') as pair_file, \
                cache.open() as cached, history.open() as historic:
            writer = csv.DictWriter(pair_file, fieldnames=PAIR_FIELDS)
            writer.writeheader()
            audit = AssociationAudit(tracker.evidence_updater, writer)
            tracker.evidence_updater.associate_tracks_to_observations = audit.associate
            for line in cached:
                row = json.loads(line)
                frames += 1
                assert row['frame_id'] == frames
                audit.frame = frames
                update_tracking_for_frame('msdc_elt', None, tracker, None, frame,
                                          frames - 1, row['file_type'], row['high_boxes'], row['low_boxes'])
                old = json.loads(next(historic))
                assert old['frame_idx'] == frames - 1
                # Complete non-removed snapshots include IDs, states, velocities,
                # boxes, scores, detection ages and source histories.
                if tracker._debug_track_snapshot() != old['tracks']:
                    mismatches += 1
                    if len(first_mismatches) < 5:
                        first_mismatches.append(frames)
            assert next(historic, None) is None
        assert frames == item['frames'], sequence
        summary = dict(sequence=sequence, frames=frames,
                       **{k: audit.counts[k] for k in COUNT_FIELDS},
                       max_distance_when_iou_passes=audit.max_iou_pass_distance,
                       historical_snapshot_mismatch_frames=mismatches,
                       diagnostic_seconds=time.perf_counter()-started)
        summaries.append(summary)
        provenance.append(dict(sequence=sequence, cache=str(cache), cache_sha256=sha256(cache),
                               historical_tracks=str(history), historical_tracks_sha256=sha256(history),
                               first_mismatch_frames=first_mismatches))
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    total = dict(sequence='TOTAL', frames=sum(r['frames'] for r in summaries),
                 **{k: sum(r[k] for r in summaries) for k in COUNT_FIELDS},
                 max_distance_when_iou_passes=max(r['max_distance_when_iou_passes'] or 0 for r in summaries),
                 historical_snapshot_mismatch_frames=sum(r['historical_snapshot_mismatch_frames'] for r in summaries),
                 diagnostic_seconds=sum(r['diagnostic_seconds'] for r in summaries))
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(total))
        writer.writeheader()
        writer.writerows([*summaries, total])
    details = dict(analysis_type='cached ordinary-association diagnostic, not a benchmark',
                   source_run=str(run), unit='predicted track box--merged observation group pair per frame',
                   iou_threshold=config.MSDC_ASSOC_IOU_THRESH,
                   center_distance_threshold_pixels=config.MSDC_ASSOC_CENTER_DIST,
                   note='Local alternative assignments hold states fixed; no claim of alternative MOT accuracy.',
                   dummy_frame='Only array validity checked by current lifecycle; no pixel-dependent work executed.',
                   totals=total, sequences=summaries, provenance=provenance,
                   source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [
                       Path(__file__), ROOT/'target_module/image_detect_module/config.py',
                       ROOT/'target_module/image_detect_module/utils/evidence_state.py',
                       ROOT/'target_module/image_detect_module/utils/lifecycle_tracker.py']})
    (output / 'analysis.json').write_text(json.dumps(details, ensure_ascii=False, indent=2)+'\n')
    (output / 'status.json').write_text('{"status":"complete_diagnostic_analysis_not_benchmark"}\n')
    print(json.dumps(total, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.run_root, args.output)
