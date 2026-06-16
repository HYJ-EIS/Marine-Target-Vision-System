"""MS-DC-ELT detector helpers shared by runtime and evaluation paths."""

from __future__ import annotations

from target_module.image_detect_module.config import Config


def _get_msdc_low_conf_thresh(file_type: str) -> float:
    if file_type == "infrared":
        return float(Config.MSDC_LOW_CONF_INFRARED)
    return float(Config.MSDC_LOW_CONF_VISIBLE)


def _get_default_conf_thresh(file_type: str) -> float:
    if file_type == "infrared":
        return float(Config.INFRARED_CONF_THRESH)
    return float(Config.VISIBLE_CONF_THRESH)


def _get_processor(detector):
    processor = getattr(detector, "processor", None)
    if processor is None or not hasattr(processor, "process_frame"):
        raise RuntimeError("detector.processor.process_frame(frame, file_type, conf_override=...) is unavailable")
    return processor


def run_msdc_low_threshold_detection(detector, frame, file_type: str) -> list[dict]:
    if not bool(getattr(Config, "MSDC_USE_LOW_DET", True)):
        return []
    processor = _get_processor(detector)
    low_conf = _get_msdc_low_conf_thresh(file_type)
    low_stats = processor.process_frame(frame, file_type, conf_override=low_conf)
    if low_stats is None:
        raise RuntimeError(f"Low-threshold detection failed: file_type={file_type}")
    return list(low_stats.get("boxes", []) or [])


def run_msdc_high_threshold_detection(detector, frame, file_type: str) -> list[dict]:
    processor = _get_processor(detector)
    high_stats = processor.process_frame(frame, file_type)
    if high_stats is None:
        raise RuntimeError(f"High-threshold detection failed: file_type={file_type}")
    return list(high_stats.get("boxes", []) or [])


def split_msdc_high_from_low_boxes(low_boxes: list[dict], file_type: str) -> list[dict]:
    threshold = _get_default_conf_thresh(file_type)
    return [
        dict(box)
        for box in list(low_boxes or [])
        if float(box.get("confidence", box.get("score", 0.0))) >= threshold
    ]


def resolve_msdc_high_low_boxes(detector, frame, file_type: str) -> tuple[list[dict], list[dict]]:
    """Return high and low boxes using the configured MS-DC detector strategy."""
    if bool(getattr(Config, "MSDC_EXPORT_SHARE_LOW_HIGH_DET", False)) and bool(
        getattr(Config, "MSDC_USE_LOW_DET", True)
    ):
        low_boxes = run_msdc_low_threshold_detection(detector, frame, file_type)
        return split_msdc_high_from_low_boxes(low_boxes, file_type), low_boxes
    high_boxes = run_msdc_high_threshold_detection(detector, frame, file_type)
    low_boxes = run_msdc_low_threshold_detection(detector, frame, file_type)
    return high_boxes, low_boxes
