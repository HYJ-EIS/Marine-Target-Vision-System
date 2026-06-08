import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.render_mot_video import _assign_classes, load_mot_results


def test_load_mot_results_groups_rows_by_frame(tmp_path):
    mot_file = tmp_path / "seq.txt"
    mot_file.write_text(
        "3,1,10,20,30,40,0.9,-1,-1,-1\n"
        "3,2,50,60,20,10,0.8,-1,-1,-1\n"
        "4,1,11,21,30,40,0.7,-1,-1,-1\n",
        encoding="utf-8",
    )

    rows = load_mot_results(mot_file)

    assert sorted(rows) == [3, 4]
    assert rows[3][0]["track_id"] == 1
    assert rows[3][1]["w"] == 20.0
    assert rows[4][0]["confidence"] == 0.7


def test_assign_classes_matches_detection_and_uses_cache():
    cache = {}
    tracks = [{"track_id": 7, "x": 10, "y": 10, "w": 20, "h": 20, "confidence": 1.0}]
    detections = [{"x": 11, "y": 11, "w": 20, "h": 20, "confidence": 0.8, "class": "UAV"}]

    first = _assign_classes(tracks, detections, cache, iou_threshold=0.1)
    second = _assign_classes(tracks, [], cache, iou_threshold=0.1)

    assert first[0]["class"] == "UAV"
    assert first[0]["class_confidence"] == 0.8
    assert second[0]["class"] == "UAV"
    assert second[0]["id"] == 7
