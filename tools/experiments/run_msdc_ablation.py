"""
Build or run MS-DC-ELT ablation export commands.

The script does not hard-code dataset paths. By default it prints commands only;
use --run to execute a smoke/export run with the selected variants.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.constants import (  # noqa: E402
    DATASET_EXPORT_TRACKER_CHOICES,
    FORMAL_MSDC_VARIANT,
)


V2_BASELINE_ENV = {
    "MSDC_USE_REACQUIRE": "1",
    "MSDC_REUSE_GUARD_ENABLE": "1",
    "MSDC_OUTPUT_MAX_REAL_DET_AGE": "3",
    "MSDC_OUTPUT_MIN_BOX_SIZE": "12",
    "MSDC_LOW_CONFIRM_MIN_HITS": "5",
    "MSDC_LOW_CONFIRM_WINDOW": "8",
    "MSDC_CONFIRM_MIN_REAL_DET_HITS": "4",
    "MSDC_CANDIDATE_MAX_AGE": "5",
    "MSDC_LOW_OBS_TOPK": "0",
    "MSDC_LOW_OBS_MIN_CONF": "0.0",
    "MSDC_REACQUIRE_INTERVAL": "5",
    "MSDC_REACQUIRE_CENTER_DIST": "160",
    "MSDC_REACQUIRE_MAX_CENTER_DIST": "240",
    "MSDC_DEBUG_EVENTS": "1",
}

V3_CANDIDATE_TOPK_NO_ROI_NO_MOTION_ENV = {
    **V2_BASELINE_ENV,
    "MSDC_LOW_OBS_TOPK": "32",
    "MSDC_LOW_OBS_GLOBAL_TOPK": "32",
    "MSDC_LOW_OBS_PER_TRACK_NEAREST": "1",
    "MSDC_LOW_OBS_MAX_PER_FRAME": "64",
    "MSDC_LOW_OBS_MIN_CONF": "0.25",
    "MSDC_LOW_OBS_REQUIRE_TRACK_PROXIMITY": "1",
    "MSDC_DEBUG_EVENTS": "0",
    "MSDC_MAX_ACTIVE_TRACKS": "128",
    "MSDC_MAX_LOST_TRACKS": "64",
    "MSDC_MAX_CANDIDATES": "64",
    "MSDC_MAX_LOW_CANDIDATES": "48",
    "MSDC_MAX_TOTAL_TRACKS": "256",
}


ABLATION_VARIANTS = {
    "Ours-no-reacquire": {
        "MSDC_USE_REACQUIRE": "0",
        "MSDC_REUSE_GUARD_ENABLE": "1",
    },
    "Ours-no-removed-guard": {
        "MSDC_USE_REACQUIRE": "1",
        "MSDC_REUSE_GUARD_ENABLE": "0",
    },
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
        "MSDC_LOW_OBS_MIN_CONF": "0.25",
    },
    FORMAL_MSDC_VARIANT: dict(V3_CANDIDATE_TOPK_NO_ROI_NO_MOTION_ENV),
    "v2_speed_diag_off": {
        "MSDC_DEBUG_EVENTS": "0",
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


def _quote_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in argv)


def _variant_env(variant: str) -> dict[str, str]:
    if variant.startswith("v2_"):
        env = dict(V2_BASELINE_ENV)
        env.update(ABLATION_VARIANTS[variant])
        return env
    return dict(ABLATION_VARIANTS[variant])


def _execution_env(label: str, env_delta: dict[str, str]) -> dict[str, str]:
    if label.startswith("v2_") or label == FORMAL_MSDC_VARIANT:
        env = {key: value for key, value in os.environ.items() if not key.startswith("MSDC_")}
    else:
        env = os.environ.copy()
    env.update(env_delta)
    return env


def build_export_command(
    input_path: str,
    output_root: Path,
    tracker: str,
    seq_name: str,
    max_frames: int = 0,
) -> list[str]:
    cmd = [
        sys.executable,
        str(_ROOT / "tools" / "evaluation" / "export_mot_results.py"),
        "--input",
        input_path,
        "--output-root",
        str(output_root),
        "--tracker",
        tracker,
        "--seq-name",
        seq_name,
    ]
    if max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print or run MS-DC-ELT ablation MOT export commands")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output-root", default="results/msdc_ablation", help="Root directory for this ablation run")
    parser.add_argument("--seq-name", default="", help="MOT sequence name; default is input video stem")
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=list(DATASET_EXPORT_TRACKER_CHOICES),
        choices=DATASET_EXPORT_TRACKER_CHOICES,
        help="Trackers to include",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(ABLATION_VARIANTS),
        choices=list(ABLATION_VARIANTS),
        help="MS-DC-ELT ablation variants to include when tracker msdc_elt is selected",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Smoke/debug only; 0 means full video")
    parser.add_argument("--run", action="store_true", help="Actually execute commands; default only prints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    seq_name = args.seq_name or input_path.stem
    run_id = f"{seq_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_root) / run_id

    planned: list[tuple[str, dict[str, str], list[str]]] = []
    for tracker in args.trackers:
        if tracker == "msdc_elt":
            for variant in args.variants:
                variant_root = output_root / variant
                cmd = build_export_command(str(input_path), variant_root, tracker, seq_name, args.max_frames)
                planned.append((variant, _variant_env(variant), cmd))
        else:
            tracker_root = output_root / tracker
            cmd = build_export_command(str(input_path), tracker_root, tracker, seq_name, args.max_frames)
            planned.append((tracker, {}, cmd))

    for label, env_delta, cmd in planned:
        env_prefix = " ".join(f"{key}={value}" for key, value in sorted(env_delta.items()))
        printable = _quote_command(cmd)
        if env_prefix:
            printable = f"{env_prefix} {printable}"
        print(f"[{label}] {printable}")

        if args.run:
            env = _execution_env(label, env_delta)
            subprocess.run(cmd, cwd=str(_ROOT), env=env, check=True)

    if not args.run:
        print("[INFO] print-only mode; add --run to execute these commands.")
    if args.max_frames > 0:
        print("[WARN] --max-frames is for smoke/debug runs, not formal evaluation.")
    print(f"[INFO] planned output root: {output_root}")


if __name__ == "__main__":
    main()
