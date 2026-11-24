"""Class correctness is required before every MOT metric assignment."""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config
from third_party.trackeval.utils import TrackEvalException
from tools.evaluation.motchallenge_eval import run_motchallenge_eval


def _evaluate(tmp_path, gt_rows, prediction_rows, frames=2, labels=None):
    sequence = "class_check"
    gt = tmp_path / "gt" / sequence
    predictions = tmp_path / "trackers" / "candidate" / "data"
    (gt / "gt").mkdir(parents=True)
    predictions.mkdir(parents=True)
    (gt / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={sequence}\nimDir=img1\nframeRate=25\nseqLength={frames}\n"
        "imWidth=200\nimHeight=100\nimExt=.jpg\n",
        encoding="utf-8",
    )
    (gt / "gt" / "gt.txt").write_text("".join(row + "\n" for row in gt_rows), encoding="utf-8")
    (gt / "gt" / "labels.txt").write_text(
        "\n".join(Config.CLASSES if labels is None else labels) + "\n", encoding="utf-8"
    )
    (predictions / f"{sequence}.txt").write_text(
        "".join(row + "\n" for row in prediction_rows), encoding="utf-8"
    )
    result = run_motchallenge_eval(
        tmp_path / "gt", tmp_path / "trackers", tmp_path / "metrics", print_results=False
    )
    return result["candidate"]


def _gt(frame, identity, class_id, x=10, valid=1):
    return f"{frame},{identity},{x},10,20,20,{valid},{class_id},1"


def _prediction(frame, identity, class_id, x=10):
    return f"{frame},{identity},{x},10,20,20,0.95,{class_id},-1,-1"


def test_all_three_model_classes_match_their_one_based_gt_ids(tmp_path):
    gt = []
    predictions = []
    for frame in (1, 2):
        for class_id in (1, 2, 3):
            x = 10 + 50 * (class_id - 1)
            gt.append(_gt(frame, class_id, class_id, x))
            predictions.append(_prediction(frame, class_id + 10, class_id, x))
    result = _evaluate(tmp_path, gt, predictions)
    assert result["TP"] == result["IDTP"] == 6
    assert result["FN"] == result["FP"] == result["IDSW"] == 0
    assert result["HOTA"] == result["MOTA"] == result["IDF1"] == 100.0


def test_wrong_class_at_perfect_location_counts_false_positive_and_false_negative(tmp_path):
    result = _evaluate(
        tmp_path,
        [_gt(1, 1, 1), _gt(2, 1, 1)],
        [_prediction(1, 10, 3), _prediction(2, 10, 3)],
    )
    assert result["TP"] == result["IDTP"] == 0
    assert result["FP"] == result["FN"] == result["IDFP"] == result["IDFN"] == 2
    assert result["HOTA"] == result["IDF1"] == 0.0
    assert result["MOTA"] == -100.0


def test_class_gate_is_applied_before_assignment_of_overlapping_objects(tmp_path):
    # Every IoU is one. Unmasked Hungarian ordering would pair different classes.
    result = _evaluate(
        tmp_path,
        [_gt(1, 1, 1), _gt(1, 2, 3)],
        [_prediction(1, 10, 3), _prediction(1, 20, 1)],
        frames=1,
    )
    assert result["TP"] == result["IDTP"] == 2
    assert result["FP"] == result["FN"] == 0
    assert result["HOTA"] == result["IDF1"] == 100.0


def test_misclassification_in_one_frame_does_not_match_that_frame(tmp_path):
    result = _evaluate(
        tmp_path,
        [_gt(1, 1, 1), _gt(2, 1, 1)],
        [_prediction(1, 10, 1), _prediction(2, 10, 3)],
    )
    assert result["TP"] == result["IDTP"] == 1
    assert result["FP"] == result["FN"] == result["IDFP"] == result["IDFN"] == 1
    assert result["IDF1"] == 50.0
    assert result["IDSW"] == 0


def test_same_prediction_identity_is_not_split_when_class_changes(tmp_path):
    # One prediction ID serves two GT IDs in different classes. Evaluating each
    # class independently and averaging would incorrectly erase this ID conflict.
    result = _evaluate(
        tmp_path,
        [_gt(1, 1, 1), _gt(2, 2, 3)],
        [_prediction(1, 10, 1), _prediction(2, 10, 3)],
    )
    assert result["TP"] == 2
    assert result["FP"] == result["FN"] == 0
    assert result["IDTP"] == result["IDFP"] == result["IDFN"] == 1
    assert result["IDF1"] == 50.0


@pytest.mark.parametrize(
    "gt,predictions,fp,fn",
    [
        ([], [_prediction(1, 10, 1)], 1, 0),
        ([_gt(1, 1, 1)], [], 0, 1),
        ([], [], 0, 0),
    ],
)
def test_empty_gt_and_prediction_files_preserve_unmatched_counts(tmp_path, gt, predictions, fp, fn):
    result = _evaluate(tmp_path, gt, predictions, frames=1)
    assert result["TP"] == 0
    assert result["FP"] == fp
    assert result["FN"] == fn


@pytest.mark.parametrize("class_id", [-1, 0, 4, 1.5, "nan", "unknown"])
def test_invalid_prediction_classes_are_rejected(tmp_path, class_id):
    with pytest.raises((ValueError, TrackEvalException)):
        _evaluate(tmp_path, [_gt(1, 1, 1)], [_prediction(1, 10, class_id)], frames=1)


def test_missing_prediction_class_column_is_rejected(tmp_path):
    with pytest.raises((ValueError, TrackEvalException)):
        _evaluate(tmp_path, [_gt(1, 1, 1)], ["1,10,10,10,20,20,0.95"], frames=1)


@pytest.mark.parametrize("class_id", [0, 4, 1.5, "nan", "unknown"])
def test_invalid_gt_classes_are_rejected(tmp_path, class_id):
    with pytest.raises((ValueError, TrackEvalException)):
        _evaluate(tmp_path, [_gt(1, 1, class_id)], [_prediction(1, 10, 1)], frames=1)


def test_old_sequence_local_label_order_is_rejected(tmp_path):
    with pytest.raises((ValueError, TrackEvalException)):
        _evaluate(
            tmp_path,
            [_gt(1, 1, 1)],
            [_prediction(1, 10, 1)],
            frames=1,
            labels=list(reversed(Config.CLASSES)),
        )


def test_gt_ignore_flag_is_still_respected(tmp_path):
    result = _evaluate(
        tmp_path,
        [_gt(1, 1, 1, valid=0), _gt(1, 2, 3, x=80)],
        [_prediction(1, 20, 3, x=80)],
        frames=1,
    )
    assert result["TP"] == result["IDTP"] == 1
    assert result["FP"] == result["FN"] == 0
    assert result["HOTA"] == result["IDF1"] == 100.0
