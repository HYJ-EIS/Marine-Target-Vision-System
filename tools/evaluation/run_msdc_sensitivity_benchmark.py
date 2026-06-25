"""Replay MS-DC-ELT sensitivity variants from cached detections."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.constants import FORMAL_FRAME_LIMIT  # noqa: E402
from tools.evaluation.detection_replay_benchmark import run_dynamic_msdc_replay_benchmark  # noqa: E402


def sensitivity_tracker_name(variant: str) -> str:
    return f"{variant}_replay"


def read_matrix(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def build_dynamic_variant_envs(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    variant_envs: dict[str, dict[str, str]] = {}
    for row in rows:
        variant = str(row.get("variant", "")).strip()
        if not variant:
            raise ValueError("Sensitivity matrix row is missing variant")
        env_json = str(row.get("env_json", "") or "").strip()
        if not env_json:
            variant_envs[variant] = {}
            continue
        parsed = json.loads(env_json)
        if not isinstance(parsed, dict):
            raise ValueError(f"env_json for {variant} must decode to an object")
        variant_envs[variant] = {str(key): str(value) for key, value in parsed.items()}
    return variant_envs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MS-DC-ELT sensitivity metrics from cached detections")
    parser.add_argument("--dataset-root", nargs="+", required=True, help="Single-sequence annotation roots")
    parser.add_argument("--matrix", required=True, help="Sensitivity matrix CSV")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", default="sensitivity_metrics")
    parser.add_argument("--formal-frame-limit", type=int, default=FORMAL_FRAME_LIMIT)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--render-class-source", choices=["detector", "cache", "none"], default="cache")
    parser.add_argument(
        "--source-detection-cache-root",
        required=True,
        help="Directory containing <seq>_high_low_detections.jsonl caches to reuse",
    )
    args = parser.parse_args(argv)
    if int(args.formal_frame_limit) < 0:
        parser.error("--formal-frame-limit must be >= 0")
    if int(args.progress_interval) < 0:
        parser.error("--progress-interval must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = read_matrix(args.matrix)
    variant_envs = build_dynamic_variant_envs(rows)
    summary_path = run_dynamic_msdc_replay_benchmark(
        dataset_root=list(args.dataset_root),
        output_root=args.output_root,
        run_id=args.run_id,
        dynamic_variant_envs=variant_envs,
        source_detection_cache_root=args.source_detection_cache_root,
        formal_frame_limit=int(args.formal_frame_limit),
        progress_interval=int(args.progress_interval),
        render_class_source=str(args.render_class_source),
    )
    print(f"[OK] Sensitivity replay summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
