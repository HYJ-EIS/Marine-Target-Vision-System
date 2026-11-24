from io import StringIO
from types import SimpleNamespace
import csv

import numpy as np

from tools.evaluation.association_gate_analysis import AssociationAudit, PAIR_FIELDS, gate_masks
from target_module.image_detect_module.utils.evidence_state import EvidenceStateUpdater, _ObservationGroup
from target_module.image_detect_module.utils.msdc_types import EvidenceTrack, TrackState


def test_gate_partition_includes_boundaries_and_iou_only():
    masks = gate_masks(np.array([[.2, .19, .21, 0.]]),
                       np.array([[80., 80., 81., 81.]]), .2, 80.)
    assert {k: int(v.sum()) for k, v in masks.items()} == dict(both=1, iou_only=1, center_only=1, neither=1)
    assert np.all(sum(masks.values()) == 1)


def test_pair_counts_and_deduplicated_groups_differ_without_mutating_matches():
    updater = EvidenceStateUpdater(SimpleNamespace(MSDC_ASSOC_IOU_THRESH=.2,
                                                   MSDC_ASSOC_CENTER_DIST=80., MSDC_ASSOC_METHOD='greedy'))
    tracks = [EvidenceTrack(gid=i+1, state=TrackState.ACTIVE, box=[0, 0, 10, 10]) for i in range(2)]
    groups = [_ObservationGroup(boxes=[np.array([0, 0, 10, 10])], weights=[1.])]
    original = updater.associate_tracks_to_observations(tracks, groups, [0, 1])
    stream = StringIO()
    audit = AssociationAudit(updater, csv.DictWriter(stream, fieldnames=PAIR_FIELDS))
    assert audit.associate(tracks, groups, [0, 1]) == original
    assert audit.counts['eligible_pairs'] == 2
    assert audit.counts['eligible_groups'] == 1
    assert audit.counts['selected_both'] == 1


def test_empty_candidates_and_large_box_iou_only():
    updater = EvidenceStateUpdater(SimpleNamespace(MSDC_ASSOC_IOU_THRESH=.2,
                                                   MSDC_ASSOC_CENTER_DIST=80., MSDC_ASSOC_METHOD='greedy'))
    audit = AssociationAudit(updater, csv.DictWriter(StringIO(), fieldnames=PAIR_FIELDS))
    assert audit.associate([], [], []) == ([], [], [])
    tracks = [EvidenceTrack(gid=1, state=TrackState.ACTIVE, box=[0, 0, 400, 400])]
    groups = [_ObservationGroup(boxes=[np.array([200, 0, 600, 400])], weights=[1.])]
    assert audit.associate(tracks, groups, [0])[0] == [(0, 0)]
    assert audit.counts['iou_only'] == 1
    assert audit.counts['frames_changed_without_iou_gate'] == 1


def test_iou_score_can_change_assignment_even_when_gate_is_redundant():
    updater = EvidenceStateUpdater(SimpleNamespace(MSDC_ASSOC_IOU_THRESH=.2,
                                                   MSDC_ASSOC_CENTER_DIST=80., MSDC_ASSOC_METHOD='greedy'))
    audit = AssociationAudit(updater, csv.DictWriter(StringIO(), fieldnames=PAIR_FIELDS))
    tracks = [EvidenceTrack(gid=1, state=TrackState.ACTIVE, box=[0, 0, 100, 100])]
    # The small centered box is closest, but the shifted large box has much higher IoU.
    groups = [_ObservationGroup(boxes=[np.array(b)], weights=[1.])
              for b in ([45, 45, 55, 55], [10, 0, 110, 100])]
    assert audit.associate(tracks, groups, [0])[0] == [(0, 1)]
    assert audit.counts['frames_changed_without_iou_gate'] == 0
    assert audit.counts['frames_changed_without_iou_score'] == 1
