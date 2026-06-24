import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.render_mot_video import _assign_classes, load_detection_class_cache, load_mot_results


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


def test_load_detection_class_cache_reads_high_and_low_boxes(tmp_path):
    cache_file = tmp_path / "detections.jsonl"
    cache_file.write_text(
        '{"frame_id": 1, "high_boxes": [{"x": 10, "y": 20, "w": 30, "h": 40, "confidence": 0.9, "class": "USV"}], '
        '"low_boxes": [{"x": 100, "y": 120, "w": 10, "h": 12, "confidence": 0.3, "class": "UAV"}]}\n'
        '{"frame_id": 2, "low_boxes": [{"x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.4, "class": "UAV"}]}\n',
        encoding="utf-8",
    )

    cache = load_detection_class_cache(cache_file)

    assert sorted(cache) == [1, 2]
    assert [box["class"] for box in cache[1]] == ["USV", "UAV"]
    assert cache[1][0]["class_confidence"] == 0.9
    assert cache[2][0]["class"] == "UAV"
