import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.evaluate_detection_results import evaluate_detection_results


def test_evaluate_detection_results_counts_tp_fp_fn_and_small_recall(tmp_path):
    gt = tmp_path / "gt.txt"
    det = tmp_path / "det.txt"
    gt.write_text(
        "\n".join([
            "1,1,0,0,10,10,1,1,1",
            "1,2,100,100,10,10,1,1,1",
            "2,1,0,0,40,40,1,1,1",
        ]),
        encoding="utf-8",
    )
    det.write_text(
        "\n".join([
            "1,-1,0,0,10,10,0.9,-1,-1,-1",
            "1,-1,200,200,10,10,0.8,-1,-1,-1",
            "2,-1,0,0,40,40,0.7,-1,-1,-1",
        ]),
        encoding="utf-8",
    )

    summary = evaluate_detection_results(gt, det, iou_thresh=0.5, small_area_thresh=200)

    assert summary["tp"] == 2
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["precision"] == 0.666667
    assert summary["recall"] == 0.666667
    assert summary["fp_per_frame"] == 0.5
    assert summary["small_gt"] == 2
    assert summary["small_tp"] == 1
    assert summary["small_recall"] == 0.5
