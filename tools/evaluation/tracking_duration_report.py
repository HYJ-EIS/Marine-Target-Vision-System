"""Audit durations in existing MOT exports, without rerunning any tracker.

Use positive GT annotations as the denominator. CLEAR matches measure spatial
coverage; same-ID runs additionally measure continuity. Neither is an audited
maritime stable-tracking metric. All timings are frame counts / seqinfo FPS.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.evaluation.stable_tracking_analysis import (  # noqa: E402
    continuous_segments, match_frames, read_mot, protocol_requires_class_matching,
    validate_class_matching_inputs,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percent(a, b):
    return 100 * a / b if b else None


def summarize_target(frames, matches, identity, fps, minimum_seconds=1.0):
    """Missing annotation breaks a run but is not itself a tracking miss."""
    frames = sorted(frames)
    runs = continuous_segments(frames, matches, identity)
    matched = sum(s['frames'] for s in runs)
    longest = max((s['frames'] for s in runs), default=0)
    threshold = math.ceil(minimum_seconds * fps - 1e-9)
    sustained = sum(s['frames'] for s in runs if s['frames'] >= threshold)
    previous_id = None
    switches = 0
    direct_switches = 0
    recovered_gaps = 0
    missed_run = 0
    longest_missed = 0
    episode_had_match = False
    previous_frame = None
    previous_match = None
    for frame in frames:
        if previous_frame != frame - 1:
            episode_had_match = False
            missed_run = 0
            previous_match = None
        current = matches.get((frame, identity))
        if current is None:
            missed_run += 1
            longest_missed = max(longest_missed, missed_run)
        else:
            switches += previous_id is not None and current != previous_id
            direct_switches += previous_match is not None and current != previous_match
            recovered_gaps += bool(episode_had_match and missed_run)
            previous_id = current
            episode_had_match = True
            missed_run = 0
        previous_frame, previous_match = frame, current
    first = runs[0]['start_frame'] if runs else None
    return dict(
        observable_frames=len(frames), observable_seconds=len(frames) / fps,
        effective_frames=matched, effective_seconds=matched / fps,
        effective_percent=percent(matched, len(frames)),
        missed_frames=len(frames)-matched, missed_seconds=(len(frames)-matched)/fps,
        longest_frames=longest, longest_seconds=longest/fps,
        longest_percent=percent(longest, len(frames)),
        sustained_frames=sustained, sustained_seconds=sustained/fps,
        sustained_percent=percent(sustained, len(frames)),
        segment_count=len(runs),
        mean_segment_seconds=matched/fps/len(runs) if runs else 0.0,
        longest_missed_seconds=longest_missed/fps,
        missed_then_recovered_count=recovered_gaps,
        identity_switches=switches, direct_identity_switches=direct_switches,
        distinct_matched_ids=len({s['track_id'] for s in runs}),
        first_observable_frame=frames[0], last_observable_frame=frames[-1],
        first_match_frame=first,
        first_match_delay_seconds=(first-frames[0])/fps if first is not None else None,
    ), runs


def discover(results_root):
    included, excluded = [], []
    for ref in sorted(results_root.glob('**/pedestrian_detailed.csv')):
        # All accepted exports have run/{eval,trackeval}/tracker/detailed.csv.
        if ref.parent.parent.name not in ('eval', 'trackeval'):
            continue
        run = ref.parents[2]
        reason = None
        if any(s in str(run) for s in ('_pytest', 'smoke', 'partial', 'replay_300')):
            reason = 'test, smoke, partial, or 300-frame diagnostic run'
        if reason:
            excluded.append(dict(path=str(ref), reason=reason))
            continue
        for row in csv.DictReader(ref.open(encoding='utf-8-sig')):
            seq = row['seq']
            if seq == 'COMBINED':
                continue
            gt_root = next((run / name / seq for name in ('mot_gt', 'gt')
                            if (run / name / seq / 'gt/gt.txt').is_file()), None)
            pred = run / 'trackers' / ref.parent.name / 'data' / f'{seq}.txt'
            if gt_root is None or not pred.is_file() or not (gt_root / 'seqinfo.ini').is_file():
                excluded.append(dict(path=str(ref), sequence=seq, reason='missing GT, seqinfo, or MOT export'))
                continue
            included.append((run, ref, row, gt_root, pred))
    for pred in sorted(results_root.glob('**/trackers/*/data/*.txt')):
        run, tracker = pred.parents[3], pred.parents[1].name
        if not any((run/d/tracker/'pedestrian_detailed.csv').exists() for d in ('eval','trackeval')):
            excluded.append(dict(path=str(pred), reason='MOT exists but no completed TrackEval detailed result'))
    return included, excluded


def analyze_record(record, minimum_seconds):
    run, ref, reference, gt_root, pred_path = record
    sequence = reference['seq']
    tracker = ref.parent.name
    cfg = configparser.ConfigParser()
    cfg.read(gt_root / 'seqinfo.ini')
    fps = float(cfg['Sequence']['frameRate'])
    length = int(cfg['Sequence']['seqLength'])
    if fps <= 0 or length <= 0:
        raise ValueError(f'Invalid FPS/length: {gt_root}')
    gt_path = gt_root / 'gt/gt.txt'
    protocol_path = ref.parent.parent / 'evaluation_protocol.json'
    require_class_match = protocol_requires_class_matching(ref.parent.parent)
    if require_class_match:
        validate_class_matching_inputs(gt_path, pred_path)
    gt, pred = read_mot(gt_path, gt=True), read_mot(pred_path)
    if not all(1 <= f <= length for f in set(gt) | set(pred)):
        raise ValueError(f'Frame outside evaluated scope: {pred_path}')
    matches = match_frames(gt, pred, length, require_class_match=require_class_match)
    per_id, classes = defaultdict(list), defaultdict(set)
    for frame, rows in gt.items():
        for identity, _, cls in rows:
            per_id[identity].append(frame)
            classes[identity].add(cls)
    common = dict(run=str(run), sequence=sequence, tracker=tracker, fps=fps,
                  class_consistency_required=require_class_match)
    targets, segments = [], []
    for identity, frames in sorted(per_id.items()):
        row, runs = summarize_target(frames, matches, identity, fps, minimum_seconds)
        targets.append(dict(**common, gt_id=identity,
                            gt_class_ids='|'.join(map(str, sorted(classes[identity]))), **row))
        segments.extend(dict(**common, gt_id=identity, **s, seconds=s['frames']/fps,
                             start_seconds=(s['start_frame']-1)/fps,
                             end_seconds_exclusive=s['end_frame']/fps) for s in runs)
    ng = sum(len(rows) for rows in gt.values())
    npred = sum(len(rows) for rows in pred.values())
    tp = len(matches)
    counts = dict(CLR_TP=tp, CLR_FN=ng-tp, CLR_FP=npred-tp,
                  IDSW=sum(t['identity_switches'] for t in targets), GT_Dets=ng, Dets=npred)
    for key, value in counts.items():
        if value != int(float(reference[key])):
            raise ValueError(f'{run}/{tracker}/{sequence}: {key}: recomputed {value} != {reference[key]}')
    total = lambda key: sum(t[key] for t in targets)
    observed_frames = {f for f, rows in gt.items() if rows}
    any_matched = {f for f, identity in matches}
    all_matched = {f for f in observed_frames if all((f, row[0]) in matches for row in gt[f])}
    config_path = run / 'effective_config' / f'{tracker}.json'
    effective_cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
    row = dict(**common, eval_frames=length, eval_timeline_seconds=length/fps,
               annotated_video_seconds=len(observed_frames)/fps,
               output_video_seconds=sum(bool(p) for p in pred.values())/fps,
               any_target_matched_video_seconds=len(any_matched)/fps,
               all_targets_matched_video_seconds=len(all_matched)/fps,
               gt_targets=len(targets), never_tracked_targets=sum(t['effective_frames']==0 for t in targets),
               observable_frames=ng, observable_seconds=ng/fps,
               effective_frames=tp, effective_seconds=tp/fps, effective_percent=percent(tp, ng),
               missed_seconds=(ng-tp)/fps, false_output_target_seconds=(npred-tp)/fps,
               longest_frames=total('longest_frames'), longest_seconds=total('longest_seconds'),
               longest_percent=percent(total('longest_frames'), ng),
               sustained_frames=total('sustained_frames'), sustained_seconds=total('sustained_seconds'),
               sustained_percent=percent(total('sustained_frames'), ng),
               minimum_segment_seconds=minimum_seconds, segment_count=total('segment_count'),
               mean_segment_seconds=tp/fps/total('segment_count') if total('segment_count') else 0,
               missed_then_recovered_count=total('missed_then_recovered_count'),
               direct_identity_switches=total('direct_identity_switches'),
               identity_consistent_seconds=int(float(reference['IDTP']))/fps,
               identity_consistent_percent=100*float(reference['IDR']),
               output_min_box_size=effective_cfg.get('MSDC_OUTPUT_MIN_BOX_SIZE'),
               clear_counts_verified=True, **counts)
    for key in ['MOTA', 'IDF1', 'IDP', 'IDR', 'MOTP']:
        row[key] = 100*float(reference[key])
    row['HOTA'] = 100*float(reference['HOTA___AUC'])
    for key in ['Frag', 'IDTP', 'IDFP', 'IDFN']:
        row[key] = int(float(reference[key]))
    provenance = dict(**common, gt=str(gt_path), prediction=str(pred_path), reference=str(ref),
                      gt_sha256=digest(gt_path), prediction_sha256=digest(pred_path),
                      reference_sha256=digest(ref), seqinfo_sha256=digest(gt_root/'seqinfo.ini'),
                      evaluation_protocol=str(protocol_path) if protocol_path.exists() else None,
                      evaluation_protocol_sha256=digest(protocol_path) if protocol_path.exists() else None,
                      effective_config=str(config_path) if config_path.exists() else None,
                      effective_config_sha256=digest(config_path) if config_path.exists() else None)
    return row, targets, segments, provenance


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def latest_v3(rows):
    selected = {}
    def stamp(row):
        dates = re.findall(r'20\d{6}(?:_\d{6})?', row['run'])
        return max(dates, default='')
    for row in rows:
        if row['tracker'] == 'msdc_v3_replay':
            old = selected.get(row['sequence'])
            if old is None or stamp(row) > stamp(old):
                selected[row['sequence']] = row
    return [selected[k] for k in sorted(selected)]


def write_report(output, rows, targets, segments, selected, comparisons, manifest):
    minimum_seconds = rows[0]['minimum_segment_seconds'] if rows else 1.0
    minimum_sizes = sorted({r['output_min_box_size'] for r in selected if r['output_min_box_size'] is not None})
    labels = {
        'sequence':'视频', 'tracker':'方法', 'run':'源结果目录', 'fps':'评测帧率',
        'gt_id':'标注目标ID', 'gt_class_ids':'原始标注类别ID', 'gt_targets':'标注目标数',
        'observable_seconds':'标注目标时长(目标秒)', 'effective_seconds':'有效跟踪时长(目标秒)',
        'effective_percent':'有效跟踪率(%)', 'missed_seconds':'漏跟时长(目标秒)',
        'longest_seconds':'各目标最长同ID连续时长之和(目标秒)', 'longest_percent':'最长连续占比(%)',
        'sustained_seconds':'至少1秒连续段累计时长(目标秒)', 'sustained_percent':'至少1秒连续段占比(%)',
        'identity_consistent_seconds':'IDTP对应时长(目标秒)', 'identity_consistent_percent':'IDR(%)',
        'missed_then_recovered_count':'漏跟后恢复次数', 'segment_count':'同ID连续段数',
        'mean_segment_seconds':'平均连续段时长(秒)', 'first_match_delay_seconds':'首次匹配延迟(秒)',
        'longest_missed_seconds':'最长连续漏跟(秒)', 'identity_switches':'IDSW',
        'first_observable_frame':'首次标注帧', 'last_observable_frame':'末次标注帧',
        'first_match_frame':'首次匹配帧', 'eval_frames':'评测时间轴帧数',
        'eval_timeline_seconds':'评测时间轴秒数', 'annotated_video_seconds':'有GT的视频秒数',
        'output_video_seconds':'有输出框的视频秒数', 'never_tracked_targets':'从未匹配目标数',
        'any_target_matched_video_seconds':'至少一个目标匹配的视频秒数',
        'all_targets_matched_video_seconds':'全部在场GT匹配的视频秒数',
        'false_output_target_seconds':'FP对应时长(目标秒)',
        'output_min_box_size':'历史最小输出尺寸(像素)', 'start_frame':'起始帧',
        'end_frame':'结束帧', 'track_id':'输出ID', 'frames':'帧数', 'seconds':'片段秒数',
        'CLR_TP':'TP', 'CLR_FN':'FN', 'CLR_FP':'FP', 'Frag':'TrackEval Frag',
    }
    labels['sustained_seconds'] = f'至少{minimum_seconds:g}秒连续段累计时长(目标秒)'
    labels['sustained_percent'] = f'至少{minimum_seconds:g}秒连续段占比(%)'
    main_keys = ['sequence','observable_seconds','effective_seconds','effective_percent',
                 'longest_seconds','longest_percent','missed_then_recovered_count','IDSW',
                 'IDF1','MOTA','HOTA','Frag','missed_seconds','identity_consistent_seconds',
                 'sustained_seconds','segment_count','output_min_box_size','run']
    notes = [
        ('统计范围', f"{manifest['counts']['unique_sequences']} 段不同视频、{len(rows)} 组既有评测记录；其中 {len(selected)} 段有 V3。主表每视频只取最新可用 V3；历史明细不可直接累加。"),
        ('运行性质', '本报告只分析历史 MOT 导出，不运行检测或跟踪，不属于一次新的正式测试。'),
        ('有效跟踪时长', '按已有 CLEAR 规则进行逐帧一对一匹配，IoU≥0.5；有效目标帧数÷该视频FPS。允许中断后继续累计，也允许ID改变，因此不等同于稳定跟踪时长。'),
        ('有效跟踪率', '有效跟踪时长÷有效标注目标时长×100%。多目标分别累计：两个目标各可观测10秒，分母为20目标秒。'),
        ('标注分母', '仅有效GT行计入目标可观测时间。GT缺帧不计漏跟但打断连续片段；未独立核验全部理论可观测时间。'),
        ('最长连续时长', '每个GT目标选取逐帧匹配且不换输出ID的最长一段；逐视频表将各目标最长段相加。它不累计恢复后的多个片段。'),
        (f'至少{minimum_seconds:g}秒连续段', f'补充口径：将持续至少{minimum_seconds:g}秒且逐帧匹配、ID不变的片段完整计入；不扣除片段起始时段。此为离线回顾统计，不宣称为海试统一标准。'),
        ('IDTP对应时长', '沿用原TrackEval Identity的全序列身份分配结果，IDTP÷FPS；用于观察长期身份一致性，不要求时间连续。'),
        ('漏跟后恢复次数', '同一连续GT区间内，匹配→至少一帧漏匹配→再次匹配计1次。首次检出前的等待、末尾未恢复和GT缺帧不计；直接换号另由IDSW表示。'),
        ('TrackEval Frag差异', '保留原评测Frag。当前vendored CLEAR在整帧无GT或无预测时直接跳过部分状态更新；它与逐帧漏跟后恢复次数不完全相同，不混用两列。'),
        ('编号及类别', '按各run的evaluation_protocol.json选择匹配：要求类别一致的新协议同时检查类别与IoU；无协议的历史结果保持忽略类别。CSV记录class_consistency_required，身份编号不按类别拆分。'),
        ('配置版本', f'主表V3实际最小输出尺寸为{minimum_sizes}像素，以各run的effective_config为准；不同配置结果分开保存。'),
        ('时间轴', '使用各run的seqinfo帧率换算；不是处理程序耗时。评测时间轴长度不一定等于源视频解码长度，尤其旧5400帧评测有填充范围。'),
        ('速度', '本次不重测运行速度；原run速度和阶段计时路径在source_artifacts.csv中。读取本报告的计算耗时不能当作跟踪FPS。'),
        ('比较范围', '差值仅在同一run、同一视频内计算；baseline相对变化=（V3−baseline）÷baseline。不同视频或历史配置不作公平性等价假设，也不将重复run当随机种子。'),
        ('验证', '全部记录的TP/FN/FP/IDSW、GT框数和预测框数已逐项与原TrackEval detailed核对；analysis.json记录输入哈希。'),
    ]
    def table(data, keys):
        def fmt(value):
            return f'{value:.2f}' if isinstance(value,float) else str(value if value is not None else 'N/A')
        return '\n'.join(['| '+' | '.join(labels.get(k,k) for k in keys)+' |',
                          '| '+' | '.join('---' for _ in keys)+' |']+
                         ['| '+' | '.join(fmt(r.get(k)) for k in keys)+' |' for r in data])
    report = ['# 已有跟踪视频的有效时长统计', '',
              f'统计日期 {date.today().isoformat()}。所有时间单位为秒；多目标累计时长使用“目标秒”。', '',
              '## 最新可用 V3 结果', '',
              table(selected, main_keys[:11]), '',
              '主表按视频去重，包含不同日期的历史配置；不将六段数据合成一次同配置正式评测。', '',
              '## 四段摇晃视频的同次运行对比', '']
    shaking_sequences = {'08_drone_boat_continuous_sway', '09_KM2_drone_boat_sway',
                         'KM2无人机视频数据-3_V', 'on_sea_collection_2_7_11_V'}
    complete_shaking_runs = [run for run in {r['run'] for r in rows}
                             if shaking_sequences <= {r['sequence'] for r in rows if r['run']==run}]
    shaking_run = max(complete_shaking_runs, key=lambda run: max(re.findall(r'20\d{6}(?:_\d{6})?',run),default=''),default=None)
    shaking = [r for r in rows if r['run']==shaking_run and r['sequence'] in shaking_sequences]
    report.append(table(shaking,['sequence','tracker','effective_seconds','effective_percent','longest_seconds','IDSW','IDF1','MOTA','HOTA']))
    v3_shaking = [r for r in shaking if r['tracker']=='msdc_v3_replay']
    if v3_shaking:
        total_observed = sum(r['observable_seconds'] for r in v3_shaking)
        total_effective = sum(r['effective_seconds'] for r in v3_shaking)
        total_longest = sum(r['longest_seconds'] for r in v3_shaking)
        report += ['', f'四段视频V3：有效跟踪 {total_effective:.2f}/{total_observed:.2f} 目标秒'
                   f'（{percent(total_effective,total_observed):.2f}%）；各目标最长连续段合计 '
                   f'{total_longest:.2f} 目标秒（{percent(total_longest,total_observed):.2f}%）。']
    selected_keys={(r['run'],r['sequence'],r['tracker']) for r in selected}
    selected_targets=[t for t in targets if (t['run'],t['sequence'],t['tracker']) in selected_keys]
    report += ['', '## V3逐目标结果', '',table(selected_targets,
        ['sequence','gt_id','observable_seconds','effective_seconds','effective_percent','longest_seconds',
         'missed_then_recovered_count','identity_switches']), '',
        '## 主要发现', '']
    for row in selected:
        report.append(f"- {row['sequence']}：累计正确匹配率 {row['effective_percent']:.2f}%，"
                      f"最长连续占比 {row['longest_percent']:.2f}%；"
                      f"漏跟后恢复 {row['missed_then_recovered_count']} 次，IDSW {row['IDSW']} 次。")
    non_v3 = sorted({r['sequence'] for r in rows} - {r['sequence'] for r in selected})
    if non_v3:
        report.append('- 仅有其他方法历史结果、未发现V3导出的视频：'+', '.join(non_v3)+'。详见全部历史结果表。')
    report += ['',
        '## 后续验证建议', '',
        '结合逐目标结果检查漏跟、直接换号和恢复后的身份；确认稳定片段的最短时长及GT可观测时间口径后，再报告稳定跟踪率。', '',
        '## 统计口径与限制', '']
    report += [f'- **{name}**：{text}' for name,text in notes]
    report += ['', '## 文件', '',
               '- `跟踪时长统计.xlsx`：中文汇总、逐目标数据、全部历史记录和连续片段。',
               '- `latest_v3.csv` / `latest_v3_targets.csv`：主表的原始数值。',
               f'- `all_runs.csv` / `targets.csv` / `segments.csv`：全部历史记录及{len(segments)}个片段。',
               '- `comparisons.csv`：同run的V3相对其他方法的绝对差值和相对变化。',
               '- `analysis.json`：计算口径、源文件哈希、排除条目。',
               '- `source_artifacts.csv`：原有速度、阶段耗时、评测汇总、可视化视频路径。', '']
    (output/'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    artifacts=[]
    for run in sorted({r['run'] for r in rows}):
        rp=Path(run)
        for pattern in ['summary/*speed*.csv','summary/*stage*.csv','summary/stages.csv',
                        '**/stage_timings.json','**/*.timing.json','**/motchallenge_summary.csv','**/evaluation_protocol.json','visualizations/**/*.mp4']:
            artifacts.extend(dict(run=run,path=str(p),kind=pattern) for p in sorted(rp.glob(pattern)))
    write_csv(output/'source_artifacts.csv', artifacts)
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.chart import BarChart, Reference
    except ImportError:
        return
    wb=Workbook()
    wb.remove(wb.active)
    def sheet(name,data,keys=None):
        ws=wb.create_sheet(name)
        if not data:
            return ws
        keys=keys or list(dict.fromkeys(k for r in data for k in r))
        ws.append([labels.get(k,k) for k in keys])
        for r in data:
            ws.append([r.get(k) for k in keys])
        ws.freeze_panes='C2'
        ws.auto_filter.ref=ws.dimensions
        for cell in ws[1]:
            cell.font=Font(name='宋体',bold=True,color='FFFFFF')
            cell.fill=PatternFill('solid',fgColor='24476A')
            cell.alignment=Alignment(wrap_text=True,vertical='center')
        ws.row_dimensions[1].height=44
        from openpyxl.utils import get_column_letter
        for i,key in enumerate(keys,1):
            ws.column_dimensions[get_column_letter(i)].width=38 if key in ['sequence','run','tracker','path'] else 19
        for line in ws.iter_rows(min_row=2):
            for cell in line:
                if isinstance(cell.value,float):
                    cell.number_format='0.00'
        return ws
    sheet('口径说明',[{'项目':k,'说明':v} for k,v in notes])
    wb['口径说明'].column_dimensions['A'].width=24
    wb['口径说明'].column_dimensions['B'].width=120
    for line in wb['口径说明'].iter_rows(min_row=2):
        for cell in line:
            cell.alignment=Alignment(wrap_text=True,vertical='top')
        wb['口径说明'].row_dimensions[line[0].row].height=48
    ws=sheet('V3视频汇总',selected,main_keys)
    chart=BarChart()
    chart.title='累计正确匹配与最长连续占比'
    chart.y_axis.title='百分比'
    for col in [4,6]:
        chart.add_data(Reference(ws,min_col=col,min_row=1,max_row=1+len(selected)),titles_from_data=True)
    chart.set_categories(Reference(ws,min_col=1,min_row=2,max_row=1+len(selected)))
    chart.width=28
    chart.height=13
    ws.add_chart(chart,'A11')
    sheet('V3逐目标',selected_targets)
    sheet('摇晃四视频四方法',shaking,['sequence','tracker']+main_keys[1:])
    sheet('全部历史结果',rows)
    sheet('全部目标',targets)
    sheet('连续跟踪片段',segments)
    sheet('同run比较',comparisons)
    sheet('历史附件路径',artifacts)
    wb.save(output/'跟踪时长统计.xlsx')


def run_analysis(results_root, output, minimum_seconds):
    records, excluded = discover(results_root.resolve())
    output.mkdir(parents=True, exist_ok=False)
    rows, targets, segments, provenance = [], [], [], []
    for record in records:
        row, ts, ss, source = analyze_record(record, minimum_seconds)
        rows.append(row)
        targets.extend(ts)
        segments.extend(ss)
        provenance.append(source)
    selected = latest_v3(rows)
    selected_keys = {(r['run'], r['sequence'], r['tracker']) for r in selected}
    selected_targets = [t for t in targets if (t['run'],t['sequence'],t['tracker']) in selected_keys]
    # Pair comparisons only within the exact same run and sequence.
    comparisons = []
    for v3 in [r for r in rows if r['tracker']=='msdc_v3_replay']:
        for other in rows:
            if other['run'] != v3['run'] or other['sequence'] != v3['sequence'] or other is v3:
                continue
            comparisons.append(dict(run=v3['run'], sequence=v3['sequence'], baseline=other['tracker'],
                v3_effective_seconds=v3['effective_seconds'], baseline_effective_seconds=other['effective_seconds'],
                effective_seconds_delta=v3['effective_seconds']-other['effective_seconds'],
                effective_relative_percent=percent(v3['effective_seconds']-other['effective_seconds'], other['effective_seconds']),
                effective_rate_delta_pp=v3['effective_percent']-other['effective_percent'],
                longest_seconds_delta=v3['longest_seconds']-other['longest_seconds'],
                IDSW_delta=v3['IDSW']-other['IDSW'], IDF1_delta_pp=v3['IDF1']-other['IDF1']))
    for name, data in [('all_runs',rows),('latest_v3',selected),('targets',targets),
                       ('latest_v3_targets',selected_targets),('segments',segments),('comparisons',comparisons)]:
        write_csv(output / f'{name}.csv', data)
    manifest = dict(analysis_type='postprocess_existing_exports_not_new_benchmark',
                    matching='Per-run CLEAR convention; IoU >= 0.5; same class when required by saved evaluation protocol',
                    class_matching_modes=sorted({r['class_consistency_required'] for r in rows}),
                    denominator='Positive GT target frames / seqinfo FPS; NOT independently audited theoretical visibility',
                    effective='All CLEAR matched frames / FPS; identity switches do not remove matched frames',
                    longest='Sum of one longest consecutive-frame same-ID segment per GT ID',
                    sustained=f'Sum of whole same-ID segments lasting at least {minimum_seconds}s; supplementary retrospective definition',
                    recovery='Matched -> one or more unmatched annotated frames -> matched, within a consecutive GT episode',
                    scope='Existing evaluable historical runs, excluding explicit tests/smoke/partial; latest V3 per sequence in primary table',
                    uncertainty='Runs without class-consistent protocol retain historical class-agnostic matching. Annotation gaps excluded; completeness not verified. Historical configurations differ.',
                    fps='seqinfo.ini; constant-rate frame/FPS conversion, not per-frame timestamps',
                    counts=dict(records=len(rows), unique_sequences=len({r['sequence'] for r in rows}),
                                latest_v3_sequences=len(selected), targets=len(targets)),
                    sources=provenance, excluded=excluded,
                    code_sha256=digest(__file__), matching_code_sha256=digest(ROOT/'tools/evaluation/stable_tracking_analysis.py'))
    (output/'analysis.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    write_report(output, rows, targets, segments, selected, comparisons, manifest)
    print(json.dumps(dict(output=str(output), **manifest['counts'], all_clear_counts_verified=True), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, default=ROOT/'results')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--minimum-segment-seconds', type=float, default=1.0)
    args = p.parse_args()
    if args.minimum_segment_seconds <= 0:
        p.error('minimum-segment-seconds must be positive')
    run_analysis(args.results_root, args.output, args.minimum_segment_seconds)
