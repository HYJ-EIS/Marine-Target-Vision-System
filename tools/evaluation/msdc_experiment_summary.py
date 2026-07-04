"""Summarize MS-DC-ELT paper experiment outputs without inventing metrics."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.constants import (  # noqa: E402
    FORMAL_MSDC_VARIANT,
    MAIN_MSDC_TRACKER,
    METHOD_LABELS,
    METRIC_FIELDS,
    SPEED_FIELDS,
    SPEED_OUTPUT_FIELDS,
    SUMMARY_MAIN_TRACKERS,
)
from tools.experiments.run_msdc_ablation import (  # noqa: E402
    ABLATION_VARIANTS as FORMAL_ABLATION_VARIANTS,
    FORMAL_V3_ENV,
)


LEGACY_FORMAL_MSDC_VARIANT = "v3_candidate_topk_no_roi_no_motion"
MAIN_TRACKERS = set(SUMMARY_MAIN_TRACKERS) | {LEGACY_FORMAL_MSDC_VARIANT}

LEGACY_ABLATION_FIELDS = [
    "MSDC_LOW_CANDIDATE_ENABLE",
    "MSDC_USE_REACQUIRE",
    "MSDC_REUSE_GUARD_ENABLE",
    "MSDC_LOW_INHERIT_ENABLE",
    "MSDC_OUTPUT_NMS_ENABLE",
    "MSDC_OUTPUT_MAX_REAL_DET_AGE",
    "MSDC_OUTPUT_MIN_BOX_SIZE",
    "MSDC_LOW_CONFIRM_MIN_HITS",
    "MSDC_LOW_CONFIRM_WINDOW",
    "MSDC_CONFIRM_MIN_REAL_DET_HITS",
    "MSDC_CANDIDATE_MAX_AGE",
    "MSDC_LOW_OBS_TOPK",
    "MSDC_LOW_OBS_GLOBAL_TOPK",
    "MSDC_LOW_OBS_PER_TRACK_NEAREST",
    "MSDC_LOW_OBS_MAX_PER_FRAME",
    "MSDC_LOW_OBS_MIN_CONF",
    "MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY",
    "MSDC_EVIDENCE_MODE",
    "MSDC_REACQUIRE_SCORE",
    "MSDC_LOW_INHERIT_SCORE",
    "MSDC_EVIDENCE_ALPHA",
    "MSDC_CONFIRM_SCORE",
    "MSDC_DEBUG_EVENTS",
    "MSDC_MAX_ACTIVE_TRACKS",
    "MSDC_MAX_LOST_TRACKS",
    "MSDC_MAX_CANDIDATES",
    "MSDC_MAX_LOW_CANDIDATES",
    "MSDC_MAX_TOTAL_TRACKS",
    "MSDC_REACQUIRE_INTERVAL",
    "MSDC_REACQUIRE_CENTER_DIST",
    "MSDC_REACQUIRE_MAX_CENTER_DIST",
    "MSDC_PENDING_RECOVERY_ENABLE",
    "MSDC_PENDING_RECOVERY_FRAMES",
    "MSDC_PENDING_RECOVERY_REQUIRE_CLASS_MATCH",
    "MSDC_PENDING_RECOVERY_MAX_LOST_AGE",
    "MSDC_PENDING_RECOVERY_REMOVED_GUARD_FRAMES",
]
FORMAL_ABLATION_FIELDS = list(dict.fromkeys(
    field
    for env in [FORMAL_V3_ENV, *FORMAL_ABLATION_VARIANTS.values()]
    for field in env
))
ABLATION_FIELDS = list(dict.fromkeys([*LEGACY_ABLATION_FIELDS, *FORMAL_ABLATION_FIELDS]))

ABLATION_SWITCH_DEFAULTS = {
    "MSDC_LOW_CANDIDATE_ENABLE": "True",
    "MSDC_USE_REACQUIRE": "True",
    "MSDC_REUSE_GUARD_ENABLE": "True",
    "MSDC_LOW_INHERIT_ENABLE": "True",
    "MSDC_OUTPUT_NMS_ENABLE": "True",
    "MSDC_OUTPUT_MAX_REAL_DET_AGE": "3",
    "MSDC_OUTPUT_MIN_BOX_SIZE": "12",
    "MSDC_LOW_CONFIRM_MIN_HITS": "5",
    "MSDC_LOW_CONFIRM_WINDOW": "8",
    "MSDC_CONFIRM_MIN_REAL_DET_HITS": "4",
    "MSDC_CANDIDATE_MAX_AGE": "5",
    "MSDC_LOW_OBS_TOPK": "0",
    "MSDC_LOW_OBS_GLOBAL_TOPK": "0",
    "MSDC_LOW_OBS_PER_TRACK_NEAREST": "1",
    "MSDC_LOW_OBS_MAX_PER_FRAME": "64",
    "MSDC_LOW_OBS_MIN_CONF": "0.0",
    "MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY": "True",
    "MSDC_EVIDENCE_MODE": "score",
    "MSDC_REACQUIRE_SCORE": "0.8",
    "MSDC_LOW_INHERIT_SCORE": "0.40",
    "MSDC_EVIDENCE_ALPHA": "0.85",
    "MSDC_CONFIRM_SCORE": "2.5",
    "MSDC_DEBUG_EVENTS": "True",
    "MSDC_MAX_ACTIVE_TRACKS": "128",
    "MSDC_MAX_LOST_TRACKS": "64",
    "MSDC_MAX_CANDIDATES": "64",
    "MSDC_MAX_LOW_CANDIDATES": "48",
    "MSDC_MAX_TOTAL_TRACKS": "256",
    "MSDC_REACQUIRE_INTERVAL": "1",
    "MSDC_REACQUIRE_CENTER_DIST": "160",
    "MSDC_REACQUIRE_MAX_CENTER_DIST": "240",
    "MSDC_PENDING_RECOVERY_ENABLE": "False",
    "MSDC_PENDING_RECOVERY_FRAMES": "1",
    "MSDC_PENDING_RECOVERY_REQUIRE_CLASS_MATCH": "True",
    "MSDC_PENDING_RECOVERY_MAX_LOST_AGE": "40",
    "MSDC_PENDING_RECOVERY_REMOVED_GUARD_FRAMES": "80",
}
ABLATION_SWITCH_OVERRIDES = {
    "Ours-no-reacquire": {"MSDC_USE_REACQUIRE": "False"},
    "Ours-no-removed-guard": {"MSDC_REUSE_GUARD_ENABLE": "False"},
    "v2_output_age5_size8": {
        "MSDC_OUTPUT_MAX_REAL_DET_AGE": "5",
        "MSDC_OUTPUT_MIN_BOX_SIZE": "8",
    },
    "v2_output_age8_size8": {
        "MSDC_OUTPUT_MAX_REAL_DET_AGE": "8",
        "MSDC_OUTPUT_MIN_BOX_SIZE": "8",
    },
    "v2_output_age12_size8": {
        "MSDC_OUTPUT_MAX_REAL_DET_AGE": "12",
        "MSDC_OUTPUT_MIN_BOX_SIZE": "8",
    },
    "v2_candidate_low3_window6": {
        "MSDC_LOW_CONFIRM_MIN_HITS": "3",
        "MSDC_LOW_CONFIRM_WINDOW": "6",
    },
    "v2_candidate_low4_window8": {
        "MSDC_LOW_CONFIRM_MIN_HITS": "4",
        "MSDC_LOW_CONFIRM_WINDOW": "8",
    },
    "v2_candidate_real2_age8": {
        "MSDC_CONFIRM_MIN_REAL_DET_HITS": "2",
        "MSDC_CANDIDATE_MAX_AGE": "8",
    },
    "v2_candidate_topk": {
        "MSDC_LOW_OBS_TOPK": "32",
        "MSDC_LOW_OBS_GLOBAL_TOPK": "32",
        "MSDC_LOW_OBS_MIN_CONF": "0.25",
    },
    LEGACY_FORMAL_MSDC_VARIANT: {
        "MSDC_USE_REACQUIRE": "True",
        "MSDC_REUSE_GUARD_ENABLE": "True",
        "MSDC_LOW_OBS_TOPK": "32",
        "MSDC_LOW_OBS_GLOBAL_TOPK": "32",
        "MSDC_LOW_OBS_PER_TRACK_NEAREST": "1",
        "MSDC_LOW_OBS_MAX_PER_FRAME": "64",
        "MSDC_LOW_OBS_MIN_CONF": "0.25",
        "MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY": "True",
        "MSDC_DEBUG_EVENTS": "False",
        "MSDC_MAX_ACTIVE_TRACKS": "128",
        "MSDC_MAX_LOST_TRACKS": "64",
        "MSDC_MAX_CANDIDATES": "64",
        "MSDC_MAX_LOW_CANDIDATES": "48",
        "MSDC_MAX_TOTAL_TRACKS": "256",
    },
    "v2_speed_diag_off": {
        "MSDC_DEBUG_EVENTS": "False",
    },
    "v2_reacquire_interval1": {
        "MSDC_REACQUIRE_INTERVAL": "1",
    },
    "v2_reacquire_interval2": {
        "MSDC_REACQUIRE_INTERVAL": "2",
    },
    "v2_reacquire_interval5_center240": {
        "MSDC_REACQUIRE_INTERVAL": "5",
        "MSDC_REACQUIRE_CENTER_DIST": "240",
        "MSDC_REACQUIRE_MAX_CENTER_DIST": "320",
    },
}

PATH_PATTERNS = [
    ("mot_result", "**/trackers/*/data/*.txt"),
    ("trackeval_summary", "**/eval/motchallenge_summary.csv"),
    ("summary_csv", "**/main_results.csv"),
    ("summary_csv", "**/ablation_results.csv"),
    ("summary_csv", "**/sensitivity_results.csv"),
    ("summary_csv", "**/slice_metrics.csv"),
    ("summary_csv", "**/diagnostic_results.csv"),
    ("sensitivity_metrics_csv", "**/sensitivity_metrics/eval/motchallenge_summary.csv"),
    ("slice_metrics_csv", "**/slice_metrics/slice_metrics.csv"),
    ("diagnostic_csv", "**/diagnostics/**/*.csv"),
    ("diagnostic_jsonl", "**/trackers/*/diagnostics/**/*.jsonl"),
    ("effective_config_json", "**/effective_config/*.json"),
    ("visualization_mp4", "**/visualizations/**/*.mp4"),
    ("speed_csv", "**/speed_results.csv"),
    ("metadata_json", "**/run_metadata.json"),
    ("commands_jsonl", "**/commands.jsonl"),
    ("failure_csv", "**/failures.csv"),
    ("analysis_markdown", "**/analysis_*.md"),
    ("report_markdown", "**/MSDC_EXPERIMENT_REPORT.md"),
    ("docs_markdown", "**/MSDC_EXPERIMENT_RESULT.md"),
]


def _read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: str | Path, rows: list[dict], fields: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_metric_rows(summary_csv: str | Path) -> list[dict]:
    rows = _read_csv(summary_csv)
    if not rows:
        raise ValueError(f"No metric rows found: {summary_csv}")
    return rows


def _preflight_metric_rows(main_summary_csv: Path, ablation_summary_csv: Path) -> tuple[list[dict], list[dict]]:
    loaded = []
    for label, path in [("main", main_summary_csv), ("ablation", ablation_summary_csv)]:
        if not path.is_file():
            raise ValueError(f"Missing required summary CSV for {label}: {path}")
        try:
            rows = load_metric_rows(path)
        except ValueError as exc:
            raise ValueError(f"Empty required summary CSV for {label}: {path}") from exc
        loaded.append(rows)
    return loaded[0], loaded[1]


def _metric_value(row: dict, field: str) -> str:
    value = row.get(field, "")
    if value is None or value == "":
        return "N/A"
    return str(value)


def _artifact_path(root: str | Path, tracker: str, suffix: str) -> str:
    root = Path(root)
    matches = sorted(root.glob(f"**/{tracker}/{suffix}"))
    return str(matches[0]) if matches else "N/A"


def _main_tracker_key(tracker: str) -> str:
    if tracker.endswith("_replay"):
        return tracker[:-len("_replay")]
    return tracker


def _variant_key_from_tracker(tracker: str) -> str:
    if tracker.endswith("_replay"):
        return tracker[:-len("_replay")]
    return tracker


def write_main_results(
    metric_rows: list[dict],
    output_csv: str | Path,
    run_name: str,
    benchmark_root: str | Path,
) -> list[dict]:
    benchmark_root = Path(benchmark_root)
    rows = []
    for metric in metric_rows:
        tracker = metric.get("tracker", "")
        tracker_key = _main_tracker_key(tracker)
        if tracker_key not in MAIN_TRACKERS:
            continue
        rows.append({
            "run_name": run_name,
            "method": METHOD_LABELS.get(tracker_key, METHOD_LABELS.get(tracker, tracker)),
            "tracker": tracker,
            **{field: _metric_value(metric, field) for field in METRIC_FIELDS},
            "mot_result_path": _artifact_path(benchmark_root, tracker, "data/*.txt"),
            "trackeval_output_root": str(benchmark_root / "eval"),
        })

    fields = ["run_name", "method", "tracker", *METRIC_FIELDS, "mot_result_path", "trackeval_output_root"]
    _write_csv(output_csv, rows, fields)
    return rows


def _ablation_switches(variant: str) -> dict[str, str]:
    variant_key = _variant_key_from_tracker(variant)
    switches = {field: "N/A" for field in ABLATION_FIELDS}
    if variant_key in FORMAL_ABLATION_VARIANTS:
        switches.update({
            "MSDC_REUSE_GUARD_ENABLE": "True",
            "MSDC_EVIDENCE_MODE": "score",
        })
        switches.update({key: str(value) for key, value in FORMAL_ABLATION_VARIANTS[variant_key].items()})
        return switches

    switches.update(ABLATION_SWITCH_DEFAULTS)
    switches.update(ABLATION_SWITCH_OVERRIDES.get(variant_key, {}))
    return switches


def write_ablation_results(
    metric_rows: list[dict],
    output_csv: str | Path,
    run_name: str,
    benchmark_root: str | Path,
) -> list[dict]:
    benchmark_root = Path(benchmark_root)
    rows = []
    for metric in metric_rows:
        variant = metric.get("tracker", "")
        variant_key = _variant_key_from_tracker(variant)
        rows.append({
            "run_name": run_name,
            "variant": variant,
            "variant_key": variant_key,
            **_ablation_switches(variant),
            **{field: _metric_value(metric, field) for field in METRIC_FIELDS},
            "mot_result_path": _artifact_path(benchmark_root, variant, "data/*.txt"),
            "trackeval_output_root": str(benchmark_root / "eval"),
        })

    fields = [
        "run_name",
        "variant",
        "variant_key",
        *ABLATION_FIELDS,
        *METRIC_FIELDS,
        "mot_result_path",
        "trackeval_output_root",
    ]
    _write_csv(output_csv, rows, fields)
    return rows


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _row_name(row: dict) -> str:
    return row.get("method") or row.get("variant") or row.get("tracker") or "N/A"


def _comparison_phrase(metric: str, focus_value: str, baseline_value: str) -> str:
    focus_float = _as_float(focus_value)
    baseline_float = _as_float(baseline_value)
    if focus_float is None or baseline_float is None:
        return "direction is N/A because at least one value is missing"
    if focus_float == baseline_float:
        return "unchanged"
    lower_is_better = metric in {"IDSW", "FP", "FN"}
    improved = focus_float < baseline_float if lower_is_better else focus_float > baseline_float
    return "better" if improved else "worse"


def build_analysis_text(rows: list[dict], focus_name: str, baseline_names: list[str]) -> str:
    focus = next((row for row in rows if _row_name(row) == focus_name), None)
    if focus is None:
        return f"{focus_name} is missing from the real metric table. No conclusion is reported."

    lines = [f"Focus method: {focus_name}."]
    for baseline_name in baseline_names:
        baseline = next((row for row in rows if _row_name(row) == baseline_name), None)
        if baseline is None:
            lines.append(f"{baseline_name}: missing; comparison is N/A.")
            continue
        for metric in ["IDF1", "IDSW", "HOTA", "AssA", "FP", "FN"]:
            focus_value = _metric_value(focus, metric)
            baseline_value = _metric_value(baseline, metric)
            direction = _comparison_phrase(metric, focus_value, baseline_value)
            lines.append(
                f"{metric} versus {baseline_name}: {focus_value} vs {baseline_value}; {direction}."
            )
    return "\n".join(lines)


def _collect_manifest_rows(roots: list[Path]) -> list[dict]:
    seen = set()
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for kind, pattern in PATH_PATTERNS:
            for path in sorted(root.glob(pattern)):
                key = (kind, str(path))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"kind": kind, "path": str(path)})
    return rows


def write_path_manifest(root: str | Path, output_csv: str | Path) -> list[dict]:
    rows = _collect_manifest_rows([Path(root)])
    _write_csv(output_csv, rows, ["kind", "path"])
    return rows


def _append_manifest_path(rows: list[dict], kind: str, path: Path, require_exists: bool = True) -> None:
    path_text = str(path)
    if (not require_exists or path.is_file()) and not any(
        row["kind"] == kind and row["path"] == path_text for row in rows
    ):
        rows.append({"kind": kind, "path": path_text})


def _read_optional_csv(path: Path) -> list[dict]:
    if path.is_file():
        return _read_csv(path)
    return []


def _write_missing_speed_csv(path: Path, source: str) -> list[dict]:
    row = {field: "N/A" for field in SPEED_FIELDS}
    row.update({
        "status": "missing",
        "failure": f"Speed CSV not found: {source}",
    })
    rows = [row]
    _write_csv(path, rows, SPEED_OUTPUT_FIELDS)
    return rows


def _write_malformed_speed_csv(path: Path, source: str, detail: str) -> list[dict]:
    row = {field: "N/A" for field in SPEED_FIELDS}
    row.update({
        "status": "malformed",
        "failure": f"Malformed speed CSV: {source}; {detail}",
    })
    rows = [row]
    _write_csv(path, rows, SPEED_OUTPUT_FIELDS)
    return rows


def _normalize_speed_rows(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        output_row = {field: _metric_value(row, field) for field in SPEED_FIELDS}
        output_row["status"] = _metric_value(row, "status") if row.get("status") else "ok"
        output_row["failure"] = _metric_value(row, "failure") if row.get("failure") else ""
        normalized.append(output_row)
    return normalized


def _materialize_speed_csv(speed_csv: str | Path | None, output_csv: Path) -> list[dict]:
    if speed_csv is None:
        return _write_missing_speed_csv(output_csv, "N/A")

    source = Path(speed_csv)
    if source.is_file():
        try:
            rows = _read_csv(source)
        except csv.Error as exc:
            return _write_malformed_speed_csv(output_csv, str(source), str(exc))
        if not rows:
            return _write_malformed_speed_csv(output_csv, str(source), "no data rows")
        rows = _normalize_speed_rows(rows)
        _write_csv(output_csv, rows, SPEED_OUTPUT_FIELDS)
        return rows
    return _write_missing_speed_csv(output_csv, str(source))


def _materialize_optional_csv(source_csv: str | Path, output_csv: Path) -> list[dict]:
    source = Path(source_csv)
    if not source.is_file():
        rows = [{"status": "missing", "failure": str(source)}]
        _write_csv(output_csv, rows, ["status", "failure"])
        return rows

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output_csv)
    return _read_csv(output_csv)


def _first_existing_path(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return fallback


def _resolve_sensitivity_metrics_csv(args: argparse.Namespace, run_root: Path) -> Path:
    if args.sensitivity_root:
        source = Path(args.sensitivity_root)
        if source.is_file():
            return source
        return source / "sensitivity_metrics" / "eval" / "motchallenge_summary.csv"
    return run_root / "sensitivity" / "sensitivity_metrics" / "eval" / "motchallenge_summary.csv"


def _resolve_slice_metrics_csv(args: argparse.Namespace, run_root: Path) -> Path:
    if args.slice_metrics_csv:
        return Path(args.slice_metrics_csv)
    return run_root / "slice" / "slice_metrics" / "slice_metrics.csv"


def _resolve_diagnostic_csv(args: argparse.Namespace, main_root: Path) -> Path:
    if args.diagnostic_csv:
        return Path(args.diagnostic_csv)
    fallback = main_root / "diagnostics" / "msdc_diagnostic_summary.csv"
    return _first_existing_path(
        [fallback, *sorted(main_root.glob("**/diagnostics/msdc_diagnostic_summary.csv"))],
        fallback,
    )


def _markdown_table(rows: list[dict], fields: list[str]) -> str:
    if not rows:
        rows = [{field: "N/A" for field in fields}]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "N/A")) or "N/A" for field in fields) + " |")
    return "\n".join(lines)


def _load_metadata(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("**/run_metadata.json")) if root.exists() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            data = {"error": str(exc)}
        rows.append({"path": str(path), "metadata": json.dumps(data, ensure_ascii=False, sort_keys=True)})
    return rows


def _missing_metric_lines(rows: list[dict], name_field: str) -> list[str]:
    missing = []
    for row in rows:
        name = row.get(name_field, "N/A")
        fields = [field for field in METRIC_FIELDS if _metric_value(row, field) == "N/A"]
        if fields:
            missing.append(f"- {name}: missing {', '.join(fields)}")
    return missing


def _failure_lines(roots: list[Path]) -> list[str]:
    lines = []
    for root in roots:
        if not root.exists():
            lines.append(f"- Missing root: {root}")
            continue
        for path in sorted(root.glob("**/failures.csv")):
            failures = _read_optional_csv(path)
            if failures:
                lines.append(f"- {path}: {len(failures)} failure row(s)")
            else:
                lines.append(f"- {path}: present but empty")
    return lines


def _speed_failure_lines(speed_rows: list[dict]) -> list[str]:
    lines = []
    for row in speed_rows:
        failure = row.get("failure", "")
        if failure:
            lines.append(f"- Speed result failure: {failure}")
    return lines


def _optional_failure_lines(label: str, rows: list[dict]) -> list[str]:
    lines = []
    for row in rows:
        if row.get("status") == "missing" and row.get("failure"):
            lines.append(f"- {label} result failure: {row['failure']}")
    return lines


def _effectiveness_conclusion(main_rows: list[dict]) -> str:
    focus_name = METHOD_LABELS[MAIN_MSDC_TRACKER]
    focus = next((row for row in main_rows if row.get("method") == focus_name), None)
    if focus is None:
        return "MS-DC-ELT is absent from the main metric table, so no effectiveness conclusion is reported."

    baselines = [row for row in main_rows if row.get("method") != focus_name]
    if not baselines:
        return "No baseline rows are available, so no comparative effectiveness conclusion is reported."

    comparable = []
    for metric in ["IDF1", "HOTA", "AssA", "IDSW", "FP", "FN"]:
        focus_value = _metric_value(focus, metric)
        for baseline in baselines:
            baseline_value = _metric_value(baseline, metric)
            direction = _comparison_phrase(metric, focus_value, baseline_value)
            comparable.append(f"{metric} vs {_row_name(baseline)}: {focus_value} vs {baseline_value}; {direction}")
    return "Comparative evidence from available rows: " + "; ".join(comparable) + "."


def _write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _run_name(root: Path, fallback: str) -> str:
    return root.name if str(root) not in {"", "."} else fallback


def build_report(
    main_rows: list[dict],
    ablation_rows: list[dict],
    speed_rows: list[dict],
    sensitivity_rows: list[dict],
    slice_rows: list[dict],
    diagnostic_rows: list[dict],
    manifest_rows: list[dict],
    metadata_rows: list[dict],
    main_root: Path,
    ablation_root: Path,
    output_root: Path,
) -> str:
    main_fields = ["run_name", "method", "tracker", *METRIC_FIELDS]
    ablation_fields = ["run_name", "variant", "variant_key", *ABLATION_FIELDS, *METRIC_FIELDS]
    speed_fields = list(speed_rows[0]) if speed_rows else ["status", "failure"]
    sensitivity_fields = list(sensitivity_rows[0]) if sensitivity_rows else ["status", "failure"]
    slice_fields = list(slice_rows[0]) if slice_rows else ["status", "failure"]
    diagnostic_fields = list(diagnostic_rows[0]) if diagnostic_rows else ["status", "failure"]
    manifest_fields = ["kind", "path"]
    metadata_fields = ["path", "metadata"]

    missing_lines = (
        _missing_metric_lines(main_rows, "method")
        + _missing_metric_lines(ablation_rows, "variant")
        + _speed_failure_lines(speed_rows)
        + _optional_failure_lines("Sensitivity", sensitivity_rows)
        + _optional_failure_lines("Slice metrics", slice_rows)
        + _optional_failure_lines("Diagnostic", diagnostic_rows)
        + _failure_lines([main_root, ablation_root, output_root])
    )
    if not missing_lines:
        missing_lines = ["- No missing metric fields or failure CSVs were found in the summarized rows."]

    return "\n\n".join([
        "# MS-DC-ELT Experiment Report",
        "## Metadata and Paths\n\n"
        + f"- Main root: `{main_root}`\n"
        + f"- Ablation root: `{ablation_root}`\n"
        + f"- Summary output root: `{output_root}`",
        "## Run Metadata\n\n" + _markdown_table(metadata_rows, metadata_fields),
        "## Main Results\n\n" + _markdown_table(main_rows, main_fields),
        "## Ablation Results\n\n" + _markdown_table(ablation_rows, ablation_fields),
        "## Speed Results\n\n" + _markdown_table(speed_rows, speed_fields),
        "## Sensitivity Results\n\n" + _markdown_table(sensitivity_rows, sensitivity_fields),
        "## Slice Metrics\n\n" + _markdown_table(slice_rows, slice_fields),
        "## Diagnostic Results\n\n" + _markdown_table(diagnostic_rows, diagnostic_fields),
        "## Output Paths\n\n" + _markdown_table(manifest_rows, manifest_fields),
        "## Missing Metrics and Failures\n\n" + "\n".join(missing_lines),
        "## Cautious Effectiveness Conclusion\n\n" + _effectiveness_conclusion(main_rows),
    ]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MS-DC-ELT experiment outputs")
    parser.add_argument("--main-root", required=True, help="Formal main run root")
    parser.add_argument("--ablation-root", required=True, help="Formal ablation run root")
    parser.add_argument("--speed-csv", default="", help="Optional speed_results.csv path")
    parser.add_argument(
        "--sensitivity-root",
        default="",
        help="Optional sensitivity root or sensitivity motchallenge_summary.csv path",
    )
    parser.add_argument("--slice-metrics-csv", default="", help="Optional slice_metrics.csv path")
    parser.add_argument("--diagnostic-csv", default="", help="Optional msdc_diagnostic_summary.csv path")
    parser.add_argument("--output-root", required=True, help="Directory for summary CSV and analysis files")
    parser.add_argument("--report-output", required=True, help="Markdown report output path")
    parser.add_argument("--docs-output", required=True, help="Docs markdown result output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_root = Path(args.main_root)
    ablation_root = Path(args.ablation_root)
    output_root = Path(args.output_root)
    run_root = output_root.parent

    try:
        main_metric_rows, ablation_metric_rows = _preflight_metric_rows(
            main_root / "eval" / "motchallenge_summary.csv",
            ablation_root / "eval" / "motchallenge_summary.csv",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    output_root.mkdir(parents=True, exist_ok=True)

    main_rows = write_main_results(
        metric_rows=main_metric_rows,
        output_csv=output_root / "main_results.csv",
        run_name=_run_name(main_root, "main"),
        benchmark_root=main_root,
    )
    ablation_rows = write_ablation_results(
        metric_rows=ablation_metric_rows,
        output_csv=output_root / "ablation_results.csv",
        run_name=_run_name(ablation_root, "ablation"),
        benchmark_root=ablation_root,
    )
    speed_source = args.speed_csv or None
    speed_rows = _materialize_speed_csv(speed_source, output_root / "speed_results.csv")
    sensitivity_source = _resolve_sensitivity_metrics_csv(args, run_root)
    slice_source = _resolve_slice_metrics_csv(args, run_root)
    diagnostic_source = _resolve_diagnostic_csv(args, main_root)
    sensitivity_rows = _materialize_optional_csv(
        sensitivity_source,
        output_root / "sensitivity_results.csv",
    )
    slice_rows = _materialize_optional_csv(slice_source, output_root / "slice_metrics.csv")
    diagnostic_rows = _materialize_optional_csv(
        diagnostic_source,
        output_root / "diagnostic_results.csv",
    )

    main_baselines = [METHOD_LABELS["ocsort"], METHOD_LABELS["botsort"]]
    analysis_main = build_analysis_text(main_rows, METHOD_LABELS[MAIN_MSDC_TRACKER], main_baselines)
    analysis_ablation = build_analysis_text(ablation_rows, MAIN_MSDC_TRACKER, [row["variant"] for row in ablation_rows if row["variant"] != MAIN_MSDC_TRACKER])
    analysis_speed = "Speed rows are copied from the provided speed CSV when available; missing speed input is reported as N/A."
    if speed_rows and speed_rows[0].get("status") in {"missing", "malformed"}:
        analysis_speed = speed_rows[0].get("failure", analysis_speed)

    _write_text(output_root / "analysis_main.md", analysis_main + "\n")
    _write_text(output_root / "analysis_ablation.md", analysis_ablation + "\n")
    _write_text(output_root / "analysis_speed.md", analysis_speed + "\n")

    metadata_rows = _load_metadata(main_root) + _load_metadata(ablation_root)
    report_output = Path(args.report_output)
    docs_output = Path(args.docs_output)
    manifest_roots = [main_root, ablation_root, output_root, report_output.parent, docs_output.parent]
    if speed_source:
        manifest_roots.append(Path(speed_source).parent)
    manifest_roots.extend([sensitivity_source.parent, slice_source.parent, diagnostic_source.parent])
    report_manifest_rows = _collect_manifest_rows(manifest_roots)
    _append_manifest_path(report_manifest_rows, "report_markdown", report_output, require_exists=False)
    _append_manifest_path(report_manifest_rows, "docs_markdown", docs_output, require_exists=False)
    report = build_report(
        main_rows=main_rows,
        ablation_rows=ablation_rows,
        speed_rows=speed_rows,
        sensitivity_rows=sensitivity_rows,
        slice_rows=slice_rows,
        diagnostic_rows=diagnostic_rows,
        manifest_rows=report_manifest_rows,
        metadata_rows=metadata_rows,
        main_root=main_root,
        ablation_root=ablation_root,
        output_root=output_root,
    )
    _write_text(report_output, report)
    _write_text(docs_output, report)

    manifest_rows = _collect_manifest_rows(manifest_roots)
    _append_manifest_path(manifest_rows, "report_markdown", report_output)
    _append_manifest_path(manifest_rows, "docs_markdown", docs_output)
    _write_csv(output_root / "path_manifest.csv", manifest_rows, ["kind", "path"])


if __name__ == "__main__":
    main()
