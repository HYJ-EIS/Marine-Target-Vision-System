"""Shared per-frame tracking update helpers."""

from __future__ import annotations

from target_module.image_detect_module.utils.msdc_detection import run_msdc_low_threshold_detection


def is_msdc_tracker(tracker_type: str | None) -> bool:
    return tracker_type == "msdc_elt"


def update_tracking_for_frame(
    tracker_type: str,
    tracker,
    lifecycle_tracker,
    detector,
    frame,
    frame_idx: int,
    file_type: str,
    boxes: list[dict],
    low_boxes: list[dict] | None = None,
) -> list[dict]:
    if is_msdc_tracker(tracker_type):
        if lifecycle_tracker is None:
            raise RuntimeError("MS-DC-ELT lifecycle tracker is not initialized")
        if low_boxes is None:
            low_boxes = run_msdc_low_threshold_detection(detector, frame, file_type)
        return lifecycle_tracker.update(
            frame=frame,
            frame_idx=frame_idx,
            file_type=file_type,
            high_boxes=boxes,
            low_boxes=low_boxes,
        )

    if tracker is None:
        raise RuntimeError("Baseline tracker is not initialized")
    return tracker.update(boxes, frame.shape, frame=frame)
