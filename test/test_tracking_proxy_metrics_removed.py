from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tracking_proxy_metric_scripts_are_removed():
    removed = [
        ROOT / "tools" / "experiments" / "tracking_stats.py",
        ROOT / "tools" / "experiments" / "run_compare_trackers.py",
        ROOT / "tools" / "experiments" / "run_full_experiment.py",
        ROOT / "tools" / "validation" / "test_discrete_tracking.py",
    ]

    assert [path for path in removed if path.exists()] == []


def test_tracking_proxy_metric_references_are_removed_from_public_tools():
    checked_files = [
        ROOT / "README.md",
        ROOT / "tools" / "README.md",
        ROOT / "tools" / "validation" / "video_test_tracking.py",
    ]
    forbidden_terms = [
        "tracking_stats",
        "run_compare_trackers",
        "run_full_experiment",
        "test_discrete_tracking",
        "id_switch_rate_per_min",
        "id_switches",
        "total_id_switches",
        "avg_displacement_px",
        "max_displacement_px",
        "p90_displacement_px",
        "ID切换估计",
        "ID切换",
        "FPS 梯度统计",
        "proxy 指标",
    ]

    offenders = []
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            if term in text:
                offenders.append(f"{path.relative_to(ROOT)}: {term}")

    assert offenders == []
