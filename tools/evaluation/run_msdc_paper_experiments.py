"""
Safe orchestration entry point for MS-DC-ELT paper phase-1 experiments.

This module plans or runs existing benchmark scripts. It does not modify
detector, tracker, or MS-DC-ELT algorithm behavior.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]

MAIN_TRACKERS = ["ocsort", "botsort", "msdc_elt"]
ABLATION_VARIANTS = [
    "Ours-full",
    "Ours-lite-no-motion",
    "Ours-lite-no-low-det",
    "Ours-no-template",
    "Ours-no-reacquire",
    "Ours-no-removed-guard",
]
DEFAULT_DATASET_ROOTS = [
    "/home/hyj/Anti_Drone_Project/UAV_USV_MOT标注数据集",
    "/home/hyj/Anti_Drone_Project/USV_MOT标注数据集",
]
DEFAULT_OUTPUT_ROOT = Path("results/msdc_paper_phase1")
LATEST_RUN_FILENAME = "latest_run.json"


def _dataset_benchmark_script() -> str:
    return str(_ROOT / "tools" / "evaluation" / "msdc_dataset_benchmark.py")


def _speed_benchmark_script() -> str:
    return str(_ROOT / "tools" / "evaluation" / "msdc_speed_benchmark.py")


def _summary_script() -> str:
    return str(_ROOT / "tools" / "evaluation" / "msdc_experiment_summary.py")


def build_main_command(
    dataset_roots: list[str],
    output_root: Path,
    run_id: str,
    commit_hash: str,
    progress_interval: int,
    run: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        _dataset_benchmark_script(),
        "--dataset-root",
        *[str(path) for path in dataset_roots],
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--mode",
        "main",
        "--commit-hash",
        commit_hash,
        "--trackers",
        *MAIN_TRACKERS,
        "--variants",
        "Ours-full",
        "--progress-interval",
        str(progress_interval),
        "--render",
    ]
    if run:
        cmd.append("--run")
    return cmd


def build_ablation_command(
    dataset_roots: list[str],
    output_root: Path,
    run_id: str,
    commit_hash: str,
    progress_interval: int,
    run: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        _dataset_benchmark_script(),
        "--dataset-root",
        *[str(path) for path in dataset_roots],
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--mode",
        "ablation",
        "--commit-hash",
        commit_hash,
        "--trackers",
        "msdc_elt",
        "--variants",
        *ABLATION_VARIANTS,
        "--progress-interval",
        str(progress_interval),
        "--render",
    ]
    if run:
        cmd.append("--run")
    return cmd


def build_speed_command(
    dataset_root: str,
    output_root: Path,
    run_id: str,
    commit_hash: str,
    frames: int,
    progress_interval: int,
) -> list[str]:
    return [
        sys.executable,
        _speed_benchmark_script(),
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--commit-hash",
        commit_hash,
        "--frames",
        str(frames),
        "--trackers",
        *MAIN_TRACKERS,
        "--progress-interval",
        str(progress_interval),
    ]


def build_smoke_command(
    dataset_roots: list[str],
    output_root: Path,
    run_id: str,
    commit_hash: str,
    progress_interval: int,
) -> list[str]:
    return [
        sys.executable,
        _dataset_benchmark_script(),
        "--dataset-root",
        *[str(path) for path in dataset_roots],
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--mode",
        "smoke",
        "--commit-hash",
        commit_hash,
        "--trackers",
        *MAIN_TRACKERS,
        "--variants",
        "Ours-full",
        "--max-frames",
        "100",
        "--progress-interval",
        str(progress_interval),
        "--render",
        "--run",
    ]


def build_summary_command(
    main_root: Path,
    ablation_root: Path,
    speed_csv: Path,
    output_root: Path,
    report_output: Path,
    docs_output: Path,
) -> list[str]:
    return [
        sys.executable,
        _summary_script(),
        "--main-root",
        str(main_root),
        "--ablation-root",
        str(ablation_root),
        "--speed-csv",
        str(speed_csv),
        "--output-root",
        str(output_root),
        "--report-output",
        str(report_output),
        "--docs-output",
        str(docs_output),
    ]


def quote_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(label: str, cmd: list[str]) -> None:
    print(f"[{label}] {quote_command(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def get_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "N/A"
    return result.stdout.strip() or "N/A"


def latest_run_path(output_root: Path) -> Path:
    return output_root / LATEST_RUN_FILENAME


def write_latest_run(path: Path, payload: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_latest_run(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Latest run file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Latest run file must contain a JSON object: {path}")
    return data


def print_latest(output_root: Path) -> None:
    payload = load_latest_run(latest_run_path(output_root))
    for key in sorted(payload):
        print(f"{key}: {payload[key]}")


def check_latest(output_root: Path) -> None:
    payload = load_latest_run(latest_run_path(output_root))
    missing: list[str] = []
    for key, value in payload.items():
        if key.endswith("_path") or key.endswith("_csv") or key.endswith("_root") or key == "run_root":
            if not Path(str(value)).exists():
                missing.append(f"{key}: {value}")
    if missing:
        raise FileNotFoundError("Missing latest run paths:\n" + "\n".join(missing))
    print(f"[OK] Latest run paths exist: {latest_run_path(output_root)}")


def build_latest_payload(
    run_root: Path,
    main_output_root: Path,
    ablation_output_root: Path,
    speed_output_root: Path,
    summary_output_root: Path,
    frames: int,
) -> dict[str, str]:
    main_root = main_output_root / "main_full"
    ablation_root = ablation_output_root / "ablation_full"
    speed_root = speed_output_root / f"speed_{frames}"
    return {
        "run_root": str(run_root),
        "main_results_csv": str(summary_output_root / "main_results.csv"),
        "ablation_results_csv": str(summary_output_root / "ablation_results.csv"),
        "speed_results_csv": str(summary_output_root / "speed_results.csv"),
        "report_path": str(run_root / "MSDC_EXPERIMENT_REPORT.md"),
        "docs_result_path": str(_ROOT / "docs" / "MSDC_EXPERIMENT_RESULT.md"),
        "main_root": str(main_root),
        "ablation_root": str(ablation_root),
        "speed_root": str(speed_root),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run MS-DC-ELT paper phase-1 experiments")
    parser.add_argument("--dataset-root", nargs="+", default=DEFAULT_DATASET_ROOTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--speed-frames", type=int, default=1000)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--run-id", default="", help="Top-level run id; default is timestamp")
    parser.add_argument("--smoke", action="store_true", help="Run a 100-frame smoke benchmark only")
    parser.add_argument("--run-formal", action="store_true", help="Execute full formal main/ablation/speed runs")
    parser.add_argument("--print-latest", action="store_true", help="Print paths from latest_run.json and exit")
    parser.add_argument("--check-latest", action="store_true", help="Verify paths recorded in latest_run.json and exit")
    args = parser.parse_args(argv)
    if int(args.speed_frames) < 0:
        parser.error("--speed-frames must be >= 0")
    if int(args.progress_interval) < 0:
        parser.error("--progress-interval must be >= 0")
    return args


def _print_plan(commands: list[tuple[str, list[str]]]) -> None:
    for label, cmd in commands:
        print(f"[{label}] {quote_command(cmd)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root)

    if args.print_latest:
        print_latest(output_root)
        return
    if args.check_latest:
        check_latest(output_root)
        return

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = output_root / run_id
    commit_hash = get_commit_hash()

    if args.smoke:
        smoke_output_root = run_root / "smoke"
        smoke_cmd = build_smoke_command(
            dataset_roots=list(args.dataset_root),
            output_root=smoke_output_root,
            run_id="smoke_100",
            commit_hash=commit_hash,
            progress_interval=int(args.progress_interval),
        )
        run_command("smoke", smoke_cmd)
        print("[OK] Smoke run finished without updating latest formal run metadata.")
        return

    main_output_root = run_root / "main"
    ablation_output_root = run_root / "ablation"
    speed_output_root = run_root / "speed"
    summary_output_root = run_root / "summary"
    speed_run_id = f"speed_{int(args.speed_frames)}"

    main_cmd = build_main_command(
        dataset_roots=list(args.dataset_root),
        output_root=main_output_root,
        run_id="main_full",
        commit_hash=commit_hash,
        progress_interval=int(args.progress_interval),
        run=bool(args.run_formal),
    )
    ablation_cmd = build_ablation_command(
        dataset_roots=list(args.dataset_root),
        output_root=ablation_output_root,
        run_id="ablation_full",
        commit_hash=commit_hash,
        progress_interval=int(args.progress_interval),
        run=bool(args.run_formal),
    )
    speed_cmd = build_speed_command(
        dataset_root=str(args.dataset_root[0]),
        output_root=speed_output_root,
        run_id=speed_run_id,
        commit_hash=commit_hash,
        frames=int(args.speed_frames),
        progress_interval=int(args.progress_interval),
    )

    summary_cmd = build_summary_command(
        main_root=main_output_root / "main_full",
        ablation_root=ablation_output_root / "ablation_full",
        speed_csv=speed_output_root / speed_run_id / "speed_results.csv",
        output_root=summary_output_root,
        report_output=run_root / "MSDC_EXPERIMENT_REPORT.md",
        docs_output=_ROOT / "docs" / "MSDC_EXPERIMENT_RESULT.md",
    )

    formal_commands = [
        ("main", main_cmd),
        ("ablation", ablation_cmd),
        ("speed", speed_cmd),
        ("summary", summary_cmd),
    ]
    if not args.run_formal:
        print("[PLAN] Formal commands are not executed without --run-formal.")
        _print_plan(formal_commands)
        return

    for label, cmd in formal_commands:
        run_command(label, cmd)

    payload = build_latest_payload(
        run_root=run_root,
        main_output_root=main_output_root,
        ablation_output_root=ablation_output_root,
        speed_output_root=speed_output_root,
        summary_output_root=summary_output_root,
        frames=int(args.speed_frames),
    )
    write_latest_run(latest_run_path(output_root), payload)
    print(f"[OK] Latest formal run metadata: {latest_run_path(output_root)}")


if __name__ == "__main__":
    main()
