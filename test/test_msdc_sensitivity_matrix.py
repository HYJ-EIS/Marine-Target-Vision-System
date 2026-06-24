import csv
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.constants import FORMAL_MSDC_VARIANT
from tools.evaluation import msdc_sensitivity_matrix as matrix
from tools.experiments.run_msdc_ablation import FORMAL_V3_ENV


EXPECTED_POINTS = {
    "MSDC_EVIDENCE_ALPHA": ["0.68", "0.85", "1.02"],
    "MSDC_CONFIRM_SCORE": ["2.0", "2.5", "3.0"],
    "MSDC_LOW_CONFIRM_MIN_HITS": ["4", "5", "6"],
    "MSDC_LOW_INHERIT_SCORE": ["0.32", "0.40", "0.48"],
    "MSDC_REACQUIRE_SCORE": ["1.2", "1.5", "1.8"],
}


def test_sensitivity_points_cover_required_parameters_and_values():
    assert set(matrix.SENSITIVITY_POINTS) == set(EXPECTED_POINTS)
    assert matrix.SENSITIVITY_POINTS == EXPECTED_POINTS


def test_build_sensitivity_variants_starts_with_baseline_then_exact_points():
    rows = matrix.build_sensitivity_variants()

    assert rows[0] == {
        "variant": FORMAL_MSDC_VARIANT,
        "parameter": "baseline",
        "value": "baseline",
        "env_json": "",
    }
    assert len(rows) == 1 + sum(len(values) for values in EXPECTED_POINTS.values())

    by_parameter = {
        parameter: [row for row in rows[1:] if row["parameter"] == parameter]
        for parameter in EXPECTED_POINTS
    }
    for parameter, expected_values in EXPECTED_POINTS.items():
        assert [row["value"] for row in by_parameter[parameter]] == expected_values
        for row in by_parameter[parameter]:
            env = json.loads(row["env_json"])
            assert env == {**FORMAL_V3_ENV, parameter: row["value"]}
            assert list(json.loads(row["env_json"]).keys()) == sorted(env)


def test_write_sensitivity_matrix_writes_expected_csv(tmp_path):
    output = tmp_path / "nested" / "sensitivity_matrix.csv"

    matrix.write_sensitivity_matrix(output)

    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert rows == matrix.build_sensitivity_variants()


def test_cli_runs_directly_as_script(tmp_path):
    output = tmp_path / "sensitivity_matrix.csv"
    script = _ROOT / "tools" / "evaluation" / "msdc_sensitivity_matrix.py"

    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert rows
