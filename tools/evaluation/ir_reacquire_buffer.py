"""IR-only candidate buffer for lost-track reacquisition experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from tools.evaluation.proben_fusion import compute_center_distance, compute_iou


@dataclass
class IRCandidate:
    candidate_id: int
    box: dict
    first_frame: int
    last_frame: int
    hits: int
    confidence_sum: float


def _diag(box: dict) -> float:
    return math.hypot(float(box["w"]), float(box["h"]))


def _average_box(old_box: dict, new_box: dict, hits: int) -> dict:
    updated = dict(new_box)
    if hits <= 1:
        return updated
    for key in ("x", "y", "w", "h"):
        updated[key] = (
            float(old_box[key]) * float(hits - 1) + float(new_box[key])
        ) / float(hits)
    return updated


class IRReacquireBuffer:
    def __init__(self):
        self._next_id = 1
        self._candidates: list[IRCandidate] = []

    def update(self, frame_id: int, ir_only_boxes: list[dict],
               lost_track_boxes: list[dict], args) -> tuple[list[dict], list[dict]]:
        self._prune(frame_id, int(args.ir_reacquire_max_gap))
        diagnostics: list[dict] = []
        detections: list[dict] = []
        emitted = 0

        for ir_box in ir_only_boxes:
            if float(ir_box.get("confidence", 0.0)) < float(args.ir_reacquire_min_conf):
                diagnostics.append(self._diag_row(
                    frame_id, None, ir_box, None, "rejected", "low_conf"
                ))
                continue

            candidate = self._match_or_create_candidate(frame_id, ir_box, args)
            lost_track, iou, distance = self._match_lost_track(candidate.box, lost_track_boxes, args)
            if candidate.hits < int(args.ir_reacquire_confirm_frames):
                diagnostics.append(self._diag_row(
                    frame_id, candidate, ir_box, lost_track, "buffered", "not_confirmed",
                    iou=iou, center_distance=distance,
                ))
                continue
            if lost_track is None:
                reason = "no_lost_track" if not lost_track_boxes else "gate_failed"
                diagnostics.append(self._diag_row(
                    frame_id, candidate, ir_box, None, "rejected", reason,
                    iou=iou, center_distance=distance,
                ))
                continue
            if emitted >= int(args.ir_reacquire_max_per_frame):
                diagnostics.append(self._diag_row(
                    frame_id, candidate, ir_box, lost_track, "rejected", "max_per_frame",
                    iou=iou, center_distance=distance,
                ))
                continue

            detections.append(self._to_detection(candidate.box, lost_track))
            emitted += 1
            diagnostics.append(self._diag_row(
                frame_id, candidate, ir_box, lost_track, "emitted", "",
                iou=iou, center_distance=distance,
            ))
        return detections, diagnostics

    def _prune(self, frame_id: int, max_gap: int) -> None:
        self._candidates = [
            candidate for candidate in self._candidates
            if frame_id - candidate.last_frame <= max_gap
        ]

    def _match_or_create_candidate(self, frame_id: int, ir_box: dict, args) -> IRCandidate:
        best = None
        best_distance = float("inf")
        for candidate in self._candidates:
            iou = compute_iou(candidate.box, ir_box)
            distance = compute_center_distance(candidate.box, ir_box)
            if (
                iou >= float(args.ir_reacquire_candidate_iou)
                or distance <= float(args.ir_reacquire_candidate_dist_factor) * _diag(candidate.box)
            ) and distance < best_distance:
                best = candidate
                best_distance = distance
        if best is not None:
            best.hits += 1
            best.last_frame = frame_id
            best.confidence_sum += float(ir_box.get("confidence", 0.0))
            best.box = _average_box(best.box, ir_box, best.hits)
            best.box["confidence"] = max(float(best.box.get("confidence", 0.0)), float(ir_box.get("confidence", 0.0)))
            return best

        candidate = IRCandidate(
            candidate_id=self._next_id,
            box=dict(ir_box),
            first_frame=frame_id,
            last_frame=frame_id,
            hits=1,
            confidence_sum=float(ir_box.get("confidence", 0.0)),
        )
        self._next_id += 1
        self._candidates.append(candidate)
        return candidate

    def _match_lost_track(self, box: dict, lost_track_boxes: list[dict], args) -> tuple[dict | None, float, float]:
        best_track = None
        best_iou = 0.0
        best_distance = float("inf")
        for track in lost_track_boxes:
            if int(track.get("time_since_update", 0)) > int(args.ir_reacquire_max_age):
                continue
            iou = compute_iou(box, track)
            distance = compute_center_distance(box, track)
            passes = (
                iou >= float(args.ir_reacquire_iou)
                or distance <= float(args.ir_reacquire_dist_factor) * _diag(track)
            )
            if passes and distance < best_distance:
                best_track = track
                best_iou = iou
                best_distance = distance
        if best_track is None:
            return None, best_iou, best_distance if best_distance < float("inf") else 0.0
        return best_track, best_iou, best_distance

    @staticmethod
    def _to_detection(ir_box: dict, lost_track: dict) -> dict:
        return {
            "x": ir_box["x"],
            "y": ir_box["y"],
            "w": ir_box["w"],
            "h": ir_box["h"],
            "confidence": min(0.99, float(ir_box.get("confidence", 0.0))),
            "class": lost_track.get("class", ir_box.get("class", "target")),
            "class_confidence": min(0.99, float(ir_box.get("confidence", 0.0))),
            "allow_new_track": False,
            "support_type": "ir_lost_reacquire",
            "linked_track_id": int(lost_track["track_id"]),
        }

    @staticmethod
    def _diag_row(
        frame_id: int,
        candidate: IRCandidate | None,
        ir_box: dict,
        lost_track: dict | None,
        decision: str,
        reject_reason: str,
        iou: float = 0.0,
        center_distance: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "frame_id": frame_id,
            "candidate_id": candidate.candidate_id if candidate is not None else None,
            "ir_box_mapped": ir_box,
            "matched_lost_track": lost_track,
            "hits": candidate.hits if candidate is not None else 0,
            "decision": decision,
            "reject_reason": reject_reason,
            "center_distance": center_distance,
            "iou": iou,
        }
