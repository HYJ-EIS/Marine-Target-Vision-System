import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from target_module.image_detect_module import image_processor


def test_paper_contract_initializes_only_visible_model(monkeypatch):
    calls = []
    monkeypatch.setenv("PAPER_EXPERIMENT_SPEC", "visible-paper-spec.json")
    monkeypatch.setattr(image_processor, "OnnxDetector", lambda **kwargs: calls.append(kwargs) or object())
    processor = image_processor.ImageProcessor()
    assert len(calls) == 1
    assert calls[0]["onnx_path"] == image_processor.Config.ONNX_VISIBLE_MODEL_PATH
    assert processor.infrared_detector is None


def test_regular_runtime_keeps_both_existing_models(monkeypatch):
    calls = []
    monkeypatch.delenv("PAPER_EXPERIMENT_SPEC", raising=False)
    monkeypatch.setattr(image_processor, "OnnxDetector", lambda **kwargs: calls.append(kwargs) or object())
    processor = image_processor.ImageProcessor()
    assert len(calls) == 2 and processor.infrared_detector is not None
