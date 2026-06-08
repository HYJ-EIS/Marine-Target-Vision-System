import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.run_msdc_paper_experiments import (
    build_ablation_command,
    build_main_command,
    build_speed_command,
)


def test_main_command_uses_full_video_and_render():
    cmd = build_main_command(
        dataset_roots=["/data/a", "/data/b"],
        output_root=Path("results/main"),
        run_id="main_full",
        commit_hash="abcdef0",
        progress_interval=500,
        run=True,
    )
    assert "--max-frames" not in cmd
    assert "--duration-seconds" not in cmd
    assert "--render" in cmd
    assert "--run" in cmd
    assert cmd[cmd.index("--trackers") + 1:cmd.index("--variants")] == ["ocsort", "botsort", "msdc_elt"]


def test_ablation_command_includes_required_variants():
    cmd = build_ablation_command(
        dataset_roots=["/data/a"],
        output_root=Path("results/ablation"),
        run_id="ablation_full",
        commit_hash="abcdef0",
        progress_interval=500,
        run=True,
    )
    assert "Ours-full" in cmd
    assert "Ours-lite-no-motion" in cmd
    assert "Ours-lite-no-low-det" in cmd
    assert "Ours-no-template" in cmd
    assert "Ours-no-reacquire" in cmd
    assert "Ours-no-removed-guard" in cmd


def test_speed_command_records_fixed_frame_count():
    cmd = build_speed_command(
        dataset_root="/data/a",
        output_root=Path("results/speed"),
        run_id="speed_1000",
        commit_hash="abcdef0",
        frames=1000,
        progress_interval=100,
    )
    assert cmd[cmd.index("--frames") + 1] == "1000"
    assert "ocsort" in cmd
    assert "botsort" in cmd
    assert "msdc_elt" in cmd
