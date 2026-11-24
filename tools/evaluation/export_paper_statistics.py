"""Export the fixed paper experiment's existing results; never run inference."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluation.paper_statistics_run import export_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--phase', choices=('core', 'all'), required=True)
    args = parser.parse_args()
    print(json.dumps(export_run(args.run_root, args.spec, args.phase), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
