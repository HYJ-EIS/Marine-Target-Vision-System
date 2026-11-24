"""Preserve native TrackEval CSV and export explicitly percentage-scaled detailed CSV."""
import argparse
import csv
import json
from pathlib import Path

PERCENT = {'HOTA','DetA','AssA','DetRe','DetPr','AssRe','AssPr','LocA','OWTA','HOTA(0)','LocA(0)','HOTALocA(0)','MOTA','MOTP','MODA','CLR_Re','CLR_Pr','MTR','PTR','MLR','sMOTA','CLR_F1','MOTAL','IDF1','IDR','IDP'}

def normalize(root):
    manifest=[]
    for src in sorted((root/'eval').glob('*/pedestrian_detailed.csv')):
        dst=root/'summary'/f'{src.parent.name}_detailed_percent.csv'
        with src.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
        scaled=[k for k in fields if k.split('___')[0] in PERCENT]
        for row in rows:
            for k in scaled: row[k]=float(row[k])*100
        with dst.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
        manifest.append({'raw':str(src.relative_to(root)),'percent_csv':str(dst.relative_to(root)),'scaled_columns':scaled,'raw_scale':'0–1 for proportions','reported_scale':'0–100 for proportions; counts unchanged'})
    (root/'summary/csv_scales.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    report=root/'REPORT.md'
    text=report.read_text()
    text=text.replace('原始 detailed CSV 沿用 TrackEval 的 0–100 百分比标度。','原始 detailed CSV 保留 TrackEval 原生比例标度（0–1）以供追溯，不作为展示表；`summary/*_detailed_percent.csv` 将逐序列、COMBINED 和各 HOTA 阈值下的全部比例指标统一转换为 0–100，计数不变，列清单见 `summary/csv_scales.json`。')
    report.write_text(text)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run_root',type=Path); normalize(p.parse_args().run_root)
