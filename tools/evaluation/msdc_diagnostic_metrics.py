"""Summarize MS-DC lifecycle diagnostics for formal and replay runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

DIAGNOSTIC_FIELDS = [
    "seq_name",
    "tracker",
    "low_candidate_confirmed",
    "low_candidate_precision",
    "low_candidate_recall",
    "low_candidate_avg_confirm_delay",
    "inherit_correct",
    "inherit_wrong",
    "inherit_ambiguous",
    "reacquire_success",
    "fragmentation_count",
    "track_break_count",
    "idsw_before_reacquire_inherit",
    "idsw_after_reacquire_inherit",
]

_LOW_CONFIRM_EVENTS = {
    "LOW_CANDIDATE_CONFIRMED",
    "LOW_CANDIDATE_INHERITED",
    "CONFIRM_LOW_ACTIVE",
    "LOW_CANDIDATE_INHERITED_LOST",
    "LOW_CANDIDATE_MERGED",
}
_LOW_INHERIT_EVENTS = {"LOW_CANDIDATE_INHERITED", "LOW_CANDIDATE_INHERITED_LOST"}
_REACQUIRE_EVENTS = {"REACQUIRED", "LOST_REACQUIRED", "REACQUIRE_LOST"}


def _read_jsonl(path: str | Path) -> list[dict]:
    jsonl_path = Path(path)
    if not jsonl_path.is_file():
        return []
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _read_csv(path: str | Path) -> list[dict]:
    csv_path = Path(path)
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def compute_break_count(matched_segments: str) -> int:
    segments = [segment.strip() for segment in str(matched_segments or "").split(";") if segment.strip()]
    return max(0, len(segments) - 1)


def summarize_msdc_diagnostics(
    events_path: str | Path,
    stage_observations_path: str | Path,
    per_gt_diagnostics_path: str | Path,
) -> dict:
    events = _read_jsonl(events_path)
    _read_jsonl(stage_observations_path)
    per_gt_rows = _read_csv(per_gt_diagnostics_path)

    created_frames: dict[int, int] = {}
    confirmed_gids: set[int] = set()
    confirm_delays: list[int] = []
    inherit_correct = 0
    inherit_wrong = 0
    inherit_ambiguous = 0
    reacquire_success = 0

    for event in events:
        event_type = str(event.get("event_type", ""))
        gid = _coerce_int(event.get("gid"))
        frame_idx = _coerce_int(event.get("frame_idx"))
        extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}

        if event_type == "NEW_LOW_CANDIDATE" and gid is not None and frame_idx is not None:
            created_frames.setdefault(gid, frame_idx)
            continue

        if event_type in _LOW_CONFIRM_EVENTS and frame_idx is not None:
            low_gid = _event_low_candidate_gid(event)
            if low_gid is not None and low_gid in created_frames and low_gid not in confirmed_gids:
                confirmed_gids.add(low_gid)
                confirm_delays.append(max(0, frame_idx - created_frames[low_gid]))

        if event_type in _LOW_INHERIT_EVENTS:
            inherit_result = _inherit_result(extra.get("inherit_result", event.get("inherit_result")))
            if inherit_result == "correct":
                inherit_correct += 1
            elif inherit_result == "wrong":
                inherit_wrong += 1
            else:
                inherit_ambiguous += 1

        if event_type in _REACQUIRE_EVENTS:
            reacquire_success += 1

    fragmentation_count = 0
    track_break_count = 0
    for row in per_gt_rows:
        predicted_id_count = _coerce_int(row.get("predicted_id_count")) or 0
        fragmentation_count += max(0, predicted_id_count - 1)
        track_break_count += compute_break_count(str(row.get("matched_segments", "")))

    created_count = len(created_frames)
    confirmed_count = len(confirm_delays)
    avg_delay = round(sum(confirm_delays) / confirmed_count, 4) if confirmed_count else 0.0
    return {
        "seq_name": "",
        "tracker": "",
        "low_candidate_confirmed": int(confirmed_count),
        "low_candidate_precision": round(confirmed_count / created_count, 4) if created_count else "N/A",
        "low_candidate_recall": "N/A",
        "low_candidate_avg_confirm_delay": avg_delay,
        "inherit_correct": int(inherit_correct),
        "inherit_wrong": int(inherit_wrong),
        "inherit_ambiguous": int(inherit_ambiguous),
        "reacquire_success": int(reacquire_success),
        "fragmentation_count": int(fragmentation_count),
        "track_break_count": int(track_break_count),
        "idsw_before_reacquire_inherit": "N/A",
        "idsw_after_reacquire_inherit": "N/A",
    }


def write_msdc_diagnostic_summary(output_path: str | Path, rows: Iterable[dict]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        for values in rows:
            row = {field: "" for field in DIAGNOSTIC_FIELDS}
            row.update({key: value for key, value in values.items() if key in DIAGNOSTIC_FIELDS})
            writer.writerow(row)
    return output_path


def _event_low_candidate_gid(event: dict) -> int | None:
    extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}
    for key in ("low_candidate_gid", "merged_low_candidate_gid"):
        value = _coerce_int(extra.get(key))
        if value is not None:
            return value
    return _coerce_int(event.get("gid"))


def _inherit_result(value: object) -> str:
    result = str(value or "").strip().lower()
    if result in {"correct", "right", "true", "1", "yes"}:
        return "correct"
    if result in {"wrong", "incorrect", "false", "0", "no"}:
        return "wrong"
    return "ambiguous"


def _coerce_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
