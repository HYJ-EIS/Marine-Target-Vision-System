import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from target_module.image_detect_module.config import Config
from tools.evaluation.class_mapping import (
    class_names,
    class_name_from_gt_id,
    model_index_to_gt_id,
    normalize_prediction_class,
)
from tools.evaluation.export_mot_results import format_mot_result_line
from tools.evaluation.render_mot_video import _assign_classes, load_mot_results


@pytest.mark.parametrize("index,name", [(0, "USV"), (1, "fishship"), (2, "UAV")])
def test_model_to_mot_class_and_export_roundtrip(tmp_path, index, name):
    box = {"track_id": 9, "x": 1, "y": 2, "w": 3, "h": 4,
           "confidence": 0.8, "class": name, "class_id": index}
    line = format_mot_result_line(5, box)
    fields = line.split(",")
    assert len(fields) == 10
    assert fields[7:] == [str(index + 1), "-1", "-1"]
    path = tmp_path / "seq.txt"
    path.write_text(line + "\n", encoding="utf-8")
    restored = load_mot_results(path)[5][0]
    assert restored["class"] == name
    assert restored["class_id"] == index
    assert restored["mot_class_id"] == index + 1


def test_mapping_follows_config_without_a_second_class_list(monkeypatch):
    monkeypatch.setattr(Config, "CLASSES", ["beta", "alpha"])
    assert class_names() == ("beta", "alpha")
    assert normalize_prediction_class({"class": "alpha"}) == 2
    assert model_index_to_gt_id(0) == 1
    assert class_name_from_gt_id(1) == "beta"


@pytest.mark.parametrize("box", [{}, {"class": "unknown"}, {"class_id": -1},
                                      {"class_id": 3}, {"class_id": 0.5},
                                      {"class_id": float("nan")}, {"class_id": True},
                                      {"class": "USV", "class_id": 2},
                                      {"class": "UAV", "mot_class_id": 1}])
def test_missing_invalid_and_conflicting_prediction_classes_fail(box):
    with pytest.raises(ValueError):
        normalize_prediction_class(box)


def test_formatter_rejects_classless_predictions():
    with pytest.raises(ValueError, match="no class"):
        format_mot_result_line(1, {"track_id": 1, "x": 0, "y": 0, "w": 5, "h": 5})


def test_renderer_preserves_evaluated_class_against_overlapping_wrong_detection(tmp_path):
    path = tmp_path / "seq.txt"
    path.write_text("1,8,10,10,20,20,0.9,1,-1,-1\n", encoding="utf-8")
    tracks = load_mot_results(path)[1]
    wrong_detections = [{"x": 10, "y": 10, "w": 20, "h": 20, "class": "UAV", "confidence": 1.0}]
    cache = {8: ("UAV", 1.0)}
    annotated = _assign_classes(tracks, wrong_detections, cache, 0.1)
    assert annotated[0]["class"] == "USV"
    assert annotated[0]["class_id"] == 0
    assert cache[8][0] == "USV"


def test_renderer_rejects_invalid_explicit_mot_class(tmp_path):
    path = tmp_path / "seq.txt"
    path.write_text("1,1,0,0,10,10,0.8,0,-1,-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="MOT class ID"):
        load_mot_results(path)
