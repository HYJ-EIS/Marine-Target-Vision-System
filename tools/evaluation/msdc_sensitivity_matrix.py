"""Build MS-DC-ELT v3 hyperparameter sensitivity variant rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from target_module.image_detect_module.constants import FORMAL_MSDC_VARIANT  # noqa: E402
from tools.experiments.run_msdc_ablation import FORMAL_V3_ENV  # noqa: E402


SENSITIVITY_POINTS = {
    "MSDC_EVIDENCE_ALPHA": ["0.68", "0.85", "1.02"],
    "MSDC_CONFIRM_SCORE": ["2.0", "2.5", "3.0"],
    "MSDC_LOW_CONFIRM_MIN_HITS": ["4", "5", "6"],
    "MSDC_LOW_INHERIT_SCORE": ["0.32", "0.40", "0.48"],
    "MSDC_REACQUIRE_SCORE": ["1.2", "1.5", "1.8"],
}

CSV_FIELDS = ["variant", "parameter", "value", "env_json"]


def _variant_name(parameter: str, value: str) -> str:
    safe_value = value.replace(".", "p")
    return f"{parameter.lower()}_{safe_value}"


def build_sensitivity_variants() -> list[dict[str, str]]:
    rows = [
        {
            "variant": FORMAL_MSDC_VARIANT,
            "parameter": "baseline",
            "value": "baseline",
            "env_json": "",
        }
    ]
    for parameter, values in SENSITIVITY_POINTS.items():
        for value in values:
            env = dict(FORMAL_V3_ENV)
            env[parameter] = value
            rows.append(
                {
                    "variant": _variant_name(parameter, value),
                    "parameter": parameter,
                    "value": value,
                    "env_json": json.dumps(env, ensure_ascii=False, sort_keys=True),
                }
            )
    return rows


def write_sensitivity_matrix(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(build_sensitivity_variants())
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write MS-DC-ELT v3 sensitivity matrix")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = write_sensitivity_matrix(args.output)
    print(f"[OK] sensitivity matrix: {output_path.resolve()}")


if __name__ == "__main__":
    main()
