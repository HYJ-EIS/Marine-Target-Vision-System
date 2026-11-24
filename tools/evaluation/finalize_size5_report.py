"""Package size-5 formal metrics, duration, timing and controlled comparisons."""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import importlib.metadata
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.evaluation.run_unified_comparison import read,save,csvout,sha
from tools.evaluation.motchallenge_eval import run_motchallenge_eval
from tools.evaluation.unified_dataset_report import extract_metrics
from tools.evaluation.tracking_duration_report import discover,analyze_record


def load_csv(path):
    rows=list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
    for r in rows:
        for k,v in r.items():
            if v=='':r[k]=None
            else:
                try:r[k]=float(v) if any(c in v for c in '.eE') else int(v)
                except ValueError:pass
    return rows


def table(rows,keys):
    def fmt(v):return f'{v:.2f}' if isinstance(v,float) else str(v if v is not None else 'N/A')
    return '\n'.join(['| '+' | '.join(keys)+' |','|'+'---|'*len(keys)]+
                     ['| '+' | '.join(fmt(r.get(k)) for k in keys)+' |' for r in rows])


def finalize(root):
    validation=read(root/'validation.json')
    assert validation['passed'] and validation['annotated_videos']==24
    assert read(root/'effective_config/msdc_v3_replay.json')['MSDC_OUTPUT_MIN_BOX_SIZE']==5
    inventory=read(root/'inventory.json')
    source_manifest=read(root/'source_manifest.json')
    relative=Path(__file__).resolve().relative_to(ROOT)
    shutil.copy2(__file__,root/'source_snapshot'/relative)
    source_manifest[str(relative)]=sha(Path(__file__))
    for source in (ROOT/'third_party/official_trackers').rglob('*.py'):
        relative=source.relative_to(ROOT)
        dest=root/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);source_manifest[str(relative)]=sha(source)
    save(root/'source_manifest.json',source_manifest)
    packages={}
    for name in ['numpy','scipy','supervision','opencv-python','onnxruntime','onnxruntime-gpu','torch','openpyxl']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')),'unknown')
    save(root/'packages_and_hardware.json',dict(packages=packages,cpu=cpu_model,
         logical_cpus=os.cpu_count(),compute='CPU replay and render; no GPU inference in this run'))
    names=list(read(root/'run_metadata.json')['tracker_names'].values())
    seqs=[i['sequence'] for i in inventory]
    shaking=seqs[:4]
    pooled_shaking=run_motchallenge_eval(root/'mot_gt',root/'trackers',root/'eval_shaking_four',
                    trackers=names,sequences=shaking,print_results=False)
    metrics=load_csv(root/'summary/mot_metrics.csv')
    for r in metrics:
        r['Recall']=100*r['TP']/(r['TP']+r['FN'])
        r['Precision']=100*r['TP']/(r['TP']+r['FP']) if r['TP']+r['FP'] else 0
        r['GT_IDs']=r['MT']+r['PT']+r['ML']
        for k in ('MT','PT','ML'):r[k+'R']=100*r[k]/r['GT_IDs'] if r['GT_IDs'] else 0
    csvout(root/'summary/mot_metrics.csv',metrics)
    duration=load_csv(root/'duration/all_runs.csv')
    targets=load_csv(root/'duration/targets.csv')
    segments=load_csv(root/'duration/segments.csv')
    labels_by_sequence={}
    for item in inventory:
        old=Path(item['source_run'])
        if (old/'inventory.json').exists():
            previous=next(i for i in read(old/'inventory.json') if i['sequence']==item['sequence'])
            labels_file=Path(previous['gt']).parent/'labels.txt'
        else:
            labels_file=Path(item['video']).parent/'labels.txt'
        if labels_file.exists():labels_by_sequence[item['sequence']]=labels_file.read_text(encoding='utf-8-sig').splitlines()
    for row in targets:
        labels=labels_by_sequence.get(row['sequence'],[])
        ids=[int(v) for v in str(row['gt_class_ids']).split('|')]
        row['gt_class']='|'.join(labels[c-1] if 1<=c<=len(labels) else f'class_{c}' for c in ids)
    csvout(root/'summary/target_durations_with_classes.csv',targets)
    old_records={}
    for old in {i['source_run'] for i in inventory}:
        records,_=discover(Path(old))
        for record in records:
            if record[1].parent.name=='msdc_v3_replay':
                row,_,_,_=analyze_record(record,1.0)
                old_records[row['sequence']]=row
    before_after=[]
    for item in inventory:
        seq=item['sequence'];old=old_records[seq]
        new=next(r for r in duration if r['sequence']==seq and r['tracker']=='msdc_v3_replay')
        row=dict(sequence=seq,observable_seconds=new['observable_seconds'],
                 effective_seconds_12=old['effective_seconds'],effective_seconds_5=new['effective_seconds'],
                 effective_percent_12=old['effective_percent'],effective_percent_5=new['effective_percent'],
                 longest_seconds_12=old['longest_seconds'],longest_seconds_5=new['longest_seconds'])
        for key in ['HOTA','MOTA','IDF1','IDSW','CLR_TP','CLR_FP','CLR_FN','Frag','IDTP','IDFP','IDFN']:
            row[key+'_12']=old[key];row[key+'_5']=new[key];row[key+'_delta']=new[key]-old[key]
        before_after.append(row)
    csvout(root/'summary/v3_size12_vs_size5.csv',before_after)
    # Cache reuse and complete frame coverage are rechecked after the entire run.
    for item in inventory:
        seq=item['sequence']
        assert sha(root/'detections'/f'{seq}_high_low_detections.jsonl')==item['cache_sha256']
        assert sha(root/'mot_gt'/seq/'gt/gt.txt')==item['gt_sha256']
    speeds=load_csv(root/'summary/speed.csv')
    pooled_speeds=load_csv(root/'summary/speed_pooled.csv')
    stages=load_csv(root/'summary/stages.csv')
    for s in speeds:
        parts=[r['seconds'] for r in stages if r['sequence']==s['sequence'] and r['tracker']==s['tracker']]
        assert abs(sum(parts)-s['total_seconds'])<1e-6
    combined=[r for r in metrics if r['sequence']=='COMBINED']
    v3=[r for r in metrics if r['tracker']=='msdc_v3_replay' and r['sequence']!='COMBINED']
    v3_duration=[r for r in duration if r['tracker']=='msdc_v3_replay']
    duration_pooled=[]
    for name in names:
        ds=[r for r in duration if r['tracker']==name]
        observed=sum(r['observable_seconds'] for r in ds)
        effective=sum(r['effective_seconds'] for r in ds)
        longest=sum(r['longest_seconds'] for r in ds)
        duration_pooled.append(dict(tracker=name,observable_seconds=observed,effective_seconds=effective,
            effective_percent=100*effective/observed,longest_seconds=longest,longest_percent=100*longest/observed,
            missed_seconds=observed-effective,IDSW=sum(r['IDSW'] for r in ds)))
    csvout(root/'summary/duration_pooled.csv',duration_pooled)
    fields=['sequence','HOTA','DetA','AssA','MOTA','IDF1','IDSW','FP','FN','Frag','IDTP','IDFP','IDFN']
    report=['# V3最小输出尺寸5：六序列正式重测','',
        f"已完成六个历史观测区间，每方法 {validation['frames_per_method']:,} 帧；四方法共24个标注视频，全部重新解码核验。",'',
        '正式V3最小输出尺寸为5；与历史有效配置相比，唯一变化为12→5。每段视频均先使用尺寸12复现历史MOT，六段输出逐字节一致，然后运行尺寸5和三种baseline。三种baseline按当前项目参数执行，接收同一份完整低阈值检测输入；V3接收对应高、低阈值输入。','',
        '## V3逐视频MOT指标','',table(v3,fields),'',
        '比例指标单位为%，负MOTA不截断。MOTP为匹配框平均IoU。MT/PT/ML、IDP/IDR、Precision/Recall及完整指标另见Excel和summary/mot_metrics.csv。','',
        '## V3有效跟踪与连续时长','',
        table(v3_duration,['sequence','observable_seconds','effective_seconds','effective_percent','longest_seconds',
                           'longest_percent','missed_then_recovered_count','IDSW']),'',
        '时间为目标秒：正确匹配帧数/FPS，多个目标各自累加。有效跟踪率允许中断或换号后继续累计；longest_seconds为每个GT目标最长连续同ID片段之和。它们都不直接等同于海试稳定跟踪率。','',
        '## 六序列时长汇总','',table(duration_pooled,['tracker','observable_seconds','effective_seconds','effective_percent','longest_seconds','longest_percent','IDSW']),'',
        '时长比例按秒加权；MOT Recall按目标帧计数汇总。由于视频帧率不同，两种汇总比例可以不同。','',
        '## 12→5的受控比较','',
        table(before_after,['sequence','effective_seconds_12','effective_seconds_5','longest_seconds_12','longest_seconds_5',
                            'IDF1_12','IDF1_5','CLR_FN_12','CLR_FN_5']),'',
        '## 六序列总体指标','',table(combined,['method',*fields[1:]]),'',
        '此表由TrackEval按各指标定义跨序列聚合，不是六段视频的算术平均。跨方法比较仅代表本项目实现与固定参数。','',
        '## 四段摇晃视频总体指标','',table([dict(tracker=k,**v) for k,v in pooled_shaking.items()],
          ['tracker','HOTA','DetA','AssA','MOTA','IDF1','IDSW','FP','FN','Frag']),'',
        '## 速度','',table(pooled_speeds,['tracker','frames','total_seconds','mean_frame_ms','mean_fps','tracker_ms','tracker_fps']),'',
        '总耗时为本次回放和可视化的实测总和；包含读取/解码、跟踪、诊断生成/写盘和视频渲染/编码。复现尺寸12、TrackEval、统计、独立完整性检查不计入。检测复用缓存，因此低阈值检测与高阈值筛选阶段均为0，FPS不能称为包含检测的在线系统速度。全部六阶段及other见summary/stages.csv。单次测量，未给出重复运行方差。','',
        '## 数据与解释范围','',
        '- 四段摇晃视频采用全帧；USV（20250711）为4772帧，UAV_USV（20250916）为此前评测的前5400帧，未外推为完整原始长视频。',
        '- 两段旧视频缓存原链接失效，输入从stage_observations.jsonl原始high_det/low_det字段恢复；不采用过滤后的low_only、跟踪output或GT。历史MOT完全复现是恢复有效性的核验。',
        '- GT按历史文件原样复制，缺少标注的时刻不计可观测目标时长；没有独立审核理论可观测时间。检测与GT坐标可能重合，定位指标不能单独证明关联能力。',
        '- 预测MOT不包含类别。TrackEval沿用单类空间匹配，视频类别由缓存检测和同ID历史补充，unknown表示该帧缺少足够的类别证据。',
        '- 指标使用原始图像坐标；4K视频可视化缩放到1920宽，其他视频不放大。',
        '- 保留原TrackEval Frag，并另列逐帧漏跟后恢复次数；两者处理整帧无预测和GT间断的方式不同。',
        '- 第七段RGB/IR视频历史上没有V3结果，本次范围为已有V3的六段视频。','',
        '## 输出路径','',
        '- [完整Excel统计](V3尺寸5_跟踪时长与MOT指标.xlsx)',
        '- summary/mot_metrics.csv：逐视频及总体MOT指标；summary/v3_size12_vs_size5.csv：参数变化前后比较。',
        '- duration/：逐视频/逐目标时长、全部连续片段和计算口径。',
        '- trackers/*/data/*.txt：新MOT；trackers/msdc_v3_replay/diagnostics/：生命周期逐帧JSONL。',
        '- eval/：六序列TrackEval summary/detailed；eval_shaking_four/：四段摇晃视频聚合。',
        '- visualizations/：24个带框、类别和track ID的MP4及对应渲染计时。',
        '- summary/speed.csv、speed_pooled.csv、stages.csv：逐视频/总体速度及分阶段耗时。',
        '- summary/artifacts.csv：全部MOT、TrackEval、诊断、视频和计时路径。',
        '- inventory.json、effective_config/、source_manifest.json、source_snapshot/、historical_reproduction/：输入、配置、源码及历史复现依据。','']
    (root/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    from openpyxl import Workbook
    from openpyxl.styles import Font,PatternFill,Alignment
    from openpyxl.utils import get_column_letter
    wb=Workbook();wb.remove(wb.active)
    label={'sequence':'视频','method':'方法','tracker':'跟踪器','observable_seconds':'标注目标时长(目标秒)',
           'effective_seconds':'有效跟踪时长(目标秒)','effective_percent':'有效跟踪率(%)',
           'longest_seconds':'每目标最长同ID连续段之和(目标秒)','longest_percent':'最长连续占比(%)',
           'missed_then_recovered_count':'漏跟后恢复次数','frames':'帧数','total_seconds':'总耗时(秒)',
           'mean_frame_ms':'平均单帧耗时(ms)','mean_fps':'平均FPS(不含检测)','tracker_ms':'跟踪器单帧耗时(ms)',
           'gt_id':'GT目标ID','gt_class':'GT类别','seconds':'秒数','first_match_delay_seconds':'首次匹配延迟(秒)',
           'identity_switches':'IDSW','start_frame':'开始帧','end_frame':'结束帧','track_id':'输出ID'}
    def sheet(name,rows):
        ws=wb.create_sheet(name)
        if not rows:return
        keys=list(rows[0]);ws.append([label.get(k,k) for k in keys])
        for row in rows:ws.append([row.get(k) for k in keys])
        ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
        for c in ws[1]:
            c.font=Font(name='宋体',bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='24476A')
            c.alignment=Alignment(wrap_text=True,vertical='center')
        ws.row_dimensions[1].height=45
        for i,k in enumerate(keys,1):ws.column_dimensions[get_column_letter(i)].width=38 if k in ['sequence','tracker','method','run','说明'] else 19
        for line in ws.iter_rows(min_row=2):
            for c in line:
                if isinstance(c.value,float):c.number_format='0.00'
    sheet('说明',[{'项目':'参数','说明':'正式V3最小输出尺寸5，12对照复现旧MOT完全一致。'},
                  {'项目':'单位','说明':'MOT比例指标为0–100百分数；时长按目标分别累计为目标秒。'},
                  {'项目':'有效时长','说明':'IoU≥0.5的CLEAR一对一匹配累计时长；可跨中断和换号累计。'},
                  {'项目':'连续时长','说明':'每目标逐帧正确匹配且ID不变的最长段；视频表是各目标最长段之和。'},
                  {'项目':'范围','说明':'六段历史观测区间；四段摇晃全帧、USV4772帧、UAV_USV前5400帧。'},
                  {'项目':'速度','说明':'本次缓存回放+渲染；未执行检测，检测和高阈值筛选阶段为0。'},
                  {'项目':'标注','说明':'仅统计有效GT目标帧，理论可观测时间未独立审核。'},
                  {'项目':'完整说明','说明':str(root/'REPORT.md')}])
    definitions=[
        ('HOTA','综合检测与关联准确性；采用TrackEval标准实现跨IoU阈值汇总。','越高越好'),
        ('DetA','检测准确性，综合匹配、漏检和误检。','越高越好'),
        ('AssA','关联准确性，评价GT轨迹与预测轨迹对应的保持情况。','越高越好'),
        ('LocA','HOTA下匹配框的定位准确性。','越高越好'),
        ('MOTA','综合FN、FP和IDSW的准确率；允许负值。','越高越好'),
        ('MOTP','本项目报告匹配框平均IoU，非像素距离误差。','越高越好'),
        ('IDF1','全序列身份匹配的F1分数。','越高越好'),
        ('IDP / IDR','身份精确率/身份召回率，与逐帧空间匹配Precision/Recall不同。','越高越好'),
        ('TP / FP / FN','正确匹配/误检/漏检的目标帧数。','TP高；FP/FN低'),
        ('IDTP / IDFP / IDFN','全序列身份分配下正确/错误/缺失的目标帧数。','IDTP高；IDFP/IDFN低'),
        ('IDSW','同一GT目标再次匹配的预测ID相对上次发生改变的次数。','越低越好'),
        ('Frag','保留本项目TrackEval CLEAR的轨迹中断计数；另列逐帧漏跟后恢复次数。','越低越好'),
        ('MT / PT / ML','按本地TrackEval实现：匹配占比>80%、20%至80%、<20%的GT轨迹数。','MT高；ML低'),
        ('MTR / PTR / MLR','MT/PT/ML占GT轨迹总数的百分比。','MTR高；MLR低'),
        ('Precision / Recall','基于CLEAR逐帧空间匹配的精确率与召回率。','越高越好'),
    ]
    sheet('指标定义',[{'指标':k,'含义':v,'方向':d} for k,v,d in definitions])
    wb['指标定义'].column_dimensions['B'].width=85
    for line in wb['指标定义'].iter_rows(min_row=2):
        line[1].alignment=Alignment(wrap_text=True,vertical='center')
        wb['指标定义'].row_dimensions[line[0].row].height=32
    sheet('V3逐视频MOT',v3)
    sheet('V3逐视频时长',v3_duration)
    sheet('V3逐目标时长',[r for r in targets if r['tracker']=='msdc_v3_replay'])
    sheet('12与5对比',before_after)
    sheet('六视频总体MOT',combined)
    sheet('六视频总体时长',duration_pooled)
    sheet('四摇晃视频总体MOT',[dict(tracker=k,**v) for k,v in pooled_shaking.items()])
    sheet('各视频各方法MOT',metrics)
    sheet('各视频各方法时长',duration)
    sheet('各目标各方法时长',targets)
    sheet('连续片段',segments)
    sheet('总体速度',pooled_speeds)
    sheet('逐视频速度',speeds)
    sheet('分阶段耗时',stages)
    sheet('文件清单',load_csv(root/'summary/artifacts.csv'))
    path=root/'V3尺寸5_跟踪时长与MOT指标.xlsx';wb.save(path)
    save(root/'delivery_validation.json',dict(passed=True,workbook=str(path),workbook_sha256=sha(path),
        formal_validation=validation,all_cache_hashes_rechecked=True,gt_hashes_rechecked=True,
        stage_totals_equal_measured_time=True,metrics_rows=len(metrics),per_sequence_methods=24,
        effective_output_size=5,before_after_sequences=len(before_after)))
    print(json.dumps(dict(report=str(root/'REPORT.md'),workbook=str(path)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_root',type=Path)
    finalize(p.parse_args().run_root.resolve())
