import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.config import Config


def test_default_onnx_model_paths_exist_on_current_platform():
    assert Path(Config.ONNX_VISIBLE_MODEL_PATH).is_file()
    assert Path(Config.ONNX_INFRARED_MODEL_PATH).is_file()


def test_msdc_v2_formal_defaults_avoid_template_and_duplicate_full_frame_detection():
    assert Config.MSDC_USE_TEMPLATE is False
    assert Config.MSDC_TEMPLATE_ENABLE is False
    assert Config.MSDC_EXPORT_SHARE_LOW_HIGH_DET is True
