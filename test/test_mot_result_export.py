import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.export_mot_results import format_mot_result_line


def test_format_mot_result_line_uses_motchallenge_result_columns():
    box = {
        "track_id": 7,
        "x": 10,
        "y": 20,
        "w": 30,
        "h": 40,
        "confidence": 0.87654,
    }

    line = format_mot_result_line(frame_id=3, box=box)

    assert line == "3,7,10.00,20.00,30.00,40.00,0.8765,-1,-1,-1"
