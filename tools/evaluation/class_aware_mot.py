"""Class-consistent matching for the project's aggregate MOT metrics.

Both prediction and GT MOT column 8 contain Config.CLASSES indices plus one.
Track IDs remain global within each sequence, including when a prediction's
class changes. Class mismatches are forbidden before metric assignment.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

from third_party.trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
from tools.evaluation.class_mapping import class_names


def validate_gt_labels(gt_file: str | Path, labels_file: str | Path | None = None) -> None:
    """Reject a supplied local codebook that differs from the model order."""
    labels_file = Path(labels_file) if labels_file is not None else Path(gt_file).with_name('labels.txt')
    if labels_file.exists():
        labels = tuple(line.strip() for line in labels_file.read_text(encoding='utf-8-sig').splitlines() if line.strip())
        if labels != class_names():
            raise ValueError(f'{labels_file}: labels must follow Config.CLASSES {class_names()}, got {labels}; normalize GT class IDs first')


def validate_mot_class_file(path: str | Path) -> None:
    """Validate before TrackEval's integer cast can truncate invalid classes."""
    path = Path(path)
    count = len(class_names())
    with path.open(encoding='utf-8-sig', newline='') as fh:
        for line_number, row in enumerate(csv.reader(fh), 1):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) < 8:
                raise ValueError(f'{path}:{line_number}: missing class in MOT column 8; re-export predictions with class labels')
            try:
                value = Decimal(row[7].strip())
            except InvalidOperation:
                value = Decimal('NaN')
            if not value.is_finite() or value != value.to_integral_value() or not 1 <= value <= count:
                raise ValueError(f'{path}:{line_number}: class in column 8 must be an integer in 1..{count}, got {row[7]!r}; old classless predictions must be re-exported')


class ClassAwareMotChallenge2DBox(MotChallenge2DBox):
    """Keep the aggregate TrackEval output layout and require class equality."""

    @classmethod
    def get_class_name(cls):
        # Existing report readers use this dataset key and its aggregate slot.
        return 'MotChallenge2DBox'

    def __init__(self, config=None):
        super().__init__(config)
        if self.do_preproc:
            raise ValueError('Class-aware project evaluation requires DO_PREPROC=False; MOTChallenge street-scene class rules are not this model\'s classes')
        if self.data_is_zipped:
            raise ValueError('Class-aware project evaluation requires unzipped MOT files')

    def _load_raw_file(self, tracker, seq, is_gt):
        if is_gt:
            path = Path(self.config['GT_LOC_FORMAT'].format(gt_folder=self.gt_fol, seq=seq))
            validate_gt_labels(path)
        else:
            path = Path(self.tracker_fol) / tracker / self.tracker_sub_fol / (seq + '.txt')
        validate_mot_class_file(path)
        return super()._load_raw_file(tracker, seq, is_gt)

    def get_preprocessed_seq_data(self, raw_data, cls):
        gated = dict(raw_data)
        gated['similarity_scores'] = []
        for gt_classes, predicted_classes, similarity in zip(raw_data['gt_classes'], raw_data['tracker_classes'], raw_data['similarity_scores']):
            same_class = gt_classes[:, None] == predicted_classes[None, :]
            gated['similarity_scores'].append(np.where(same_class, similarity, 0.0))
        # Parent preprocessing only accepts its single MOTChallenge class slot.
        # Actual classes have already gated the similarities; keep every box
        # and every original track ID so wrong classes count toward FP/FN.
        gated['tracker_classes'] = [np.ones_like(values) for values in raw_data['tracker_classes']]
        return super().get_preprocessed_seq_data(gated, cls)
