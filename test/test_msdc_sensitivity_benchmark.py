import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation import run_msdc_sensitivity_benchmark as sensitivity


def test_sensitivity_tracker_name_is_replay_safe():
    assert sensitivity.sensitivity_tracker_name("msdc_v3") == "msdc_v3_replay"
    assert sensitivity.sensitivity_tracker_name("msdc_confirm_score_2p0") == "msdc_confirm_score_2p0_replay"


def test_build_dynamic_variant_envs_parses_matrix_rows(tmp_path):
    matrix_path = tmp_path / "sensitivity_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["variant", "parameter", "value", "env_json"])
        writer.writeheader()
        writer.writerow({"variant": "msdc_v3", "parameter": "baseline", "value": "baseline", "env_json": ""})
        writer.writerow(
            {
                "variant": "msdc_confirm_score_2p0",
                "parameter": "MSDC_CONFIRM_SCORE",
                "value": "2.0",
                "env_json": json.dumps({"MSDC_CONFIRM_SCORE": 2.0, "MSDC_LOW_CANDIDATE_ENABLE": True}),
            }
        )

    rows = sensitivity.read_matrix(matrix_path)
    envs = sensitivity.build_dynamic_variant_envs(rows)

    assert [row["variant"] for row in rows] == ["msdc_v3", "msdc_confirm_score_2p0"]
    assert envs == {
        "msdc_v3": {},
        "msdc_confirm_score_2p0": {
            "MSDC_CONFIRM_SCORE": "2.0",
            "MSDC_LOW_CANDIDATE_ENABLE": "True",
        },
    }
