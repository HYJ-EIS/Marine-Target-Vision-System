import sys
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.motchallenge_eval import run_motchallenge_eval


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_motchallenge_eval_reports_formal_metrics_for_perfect_track():
    tmp_path = _ROOT / "results" / "_pytest_motchallenge_eval"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    gt_root = tmp_path / "gt"
    trackers_root = tmp_path / "trackers"
    output_root = tmp_path / "eval"
    seq_name = "mini_seq"

    _write_text(
        gt_root / seq_name / "seqinfo.ini",
        "\n".join([
            "[Sequence]",
            f"name={seq_name}",
            "imDir=img1",
            "frameRate=25",
            "seqLength=2",
            "imWidth=100",
            "imHeight=100",
            "imExt=.jpg",
            "",
        ]),
    )
    _write_text(
        gt_root / seq_name / "gt" / "gt.txt",
        "\n".join([
            "1,1,10,10,20,20,1,1,1",
            "2,1,12,10,20,20,1,1,1",
            "",
        ]),
    )
    _write_text(
        trackers_root / "perfect" / "data" / f"{seq_name}.txt",
        "\n".join([
            "1,1,10,10,20,20,0.99,-1,-1,-1",
            "2,1,12,10,20,20,0.98,-1,-1,-1",
            "",
        ]),
    )

    metrics = run_motchallenge_eval(
        gt_root=gt_root,
        trackers_root=trackers_root,
        output_root=output_root,
        trackers=["perfect"],
        sequences=[seq_name],
        print_results=False,
    )

    perfect = metrics["perfect"]
    assert perfect["MOTA"] == 100.0
    assert perfect["IDF1"] == 100.0
    assert perfect["HOTA"] == 100.0
