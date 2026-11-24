import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluation.normalize_comparison_csv import normalize


def test_percent_export_preserves_raw_counts_and_negative_mota(tmp_path):
    src=tmp_path/'eval'/'method'/'pedestrian_detailed.csv'
    src.parent.mkdir(parents=True)
    original='seq,HOTA___5,MOTA,IDF1,CLR_TP,FP_per_frame,MOTP_sum\nCOMBINED,0.75,-0.5,0.6,30,2.5,29.5\n'
    src.write_text(original)
    (tmp_path/'summary').mkdir()
    (tmp_path/'REPORT.md').write_text('test')
    normalize(tmp_path)
    assert src.read_text()==original
    with (tmp_path/'summary/method_detailed_percent.csv').open(encoding='utf-8-sig') as f:
        row=next(csv.DictReader(f))
    assert float(row['HOTA___5'])==75
    assert float(row['MOTA'])==-50
    assert float(row['IDF1'])==60
    assert row['CLR_TP']=='30'
    assert row['FP_per_frame']=='2.5'
    assert row['MOTP_sum']=='29.5'
