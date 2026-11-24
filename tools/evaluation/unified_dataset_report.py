"""Per-dataset, equal-dataset mean and pooled reports for unified full runs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.evaluation.summarize_replay_comparison import METRICS, LOWER_BETTER, STAGES
from tools.evaluation.class_mapping import class_names, normalize_prediction_class

METHODS = {'official_botsort': 'BoT-SORT', 'msdc_elt': 'MSDC-ELT',
           'bytetrack': 'ByteTrack', 'official_ocsort': 'OC-SORT'}


def csvout(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows):
    keys = list(rows[0])
    return '\n'.join(['| ' + ' | '.join(keys) + ' |', '|' + '---|' * len(keys)] + [
        '| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v)
                         for v in row.values()) + ' |' for row in rows])


def extract_metrics(row):
    """Input is native TrackEval detailed CSV; HOTA arrays average over thresholds."""
    values = {}
    for metric in METRICS:
        if metric in {'HOTA', 'DetA', 'AssA', 'LocA'}:
            samples = [float(v) for k, v in row.items()
                       if k.startswith(metric + '___') and k != metric + '___AUC']
            if not samples:
                raise ValueError(f'Missing threshold values for {metric}')
            values[metric] = 100 * sum(samples) / len(samples)
        elif metric in {'MOTA', 'MOTP', 'IDF1', 'IDP', 'IDR'}:
            values[metric] = 100 * float(row[metric])
        else:
            values[metric] = int(float(row[{'TP': 'CLR_TP', 'FP': 'CLR_FP', 'FN': 'CLR_FN'}.get(metric, metric)]))
    return values


def performance_table(metrics):
    return [{'Metric': m + (' ↓' if m in LOWER_BETTER else ' ↑'),
             **{label: metrics[label][m] for label in METHODS.values()}} for m in METRICS]


def speed_row(label, groups):
    frames = sum(g[0] for g in groups)
    total = sum(p['total_time_s'] for g in groups for p in g[1:])
    tracker = sum(g[2]['stage_seconds']['tracker'] for g in groups)
    return {'Method': label, 'Total Time (s)': total, 'Avg Time (ms/frame)': 1000 * total / frames,
            'Overall FPS': frames / total, 'Tracker Time (s)': tracker,
            'Tracker Time (ms/frame)': 1000 * tracker / frames, 'Tracker FPS': frames / tracker}


def detection_coverage(root, item):
    """Offline same-class per-GT overlap diagnostic, not one-to-one recall."""
    from tools.evaluation.msdc_dataset_benchmark import _iou
    seq = item['sequence']
    gt = Path(item['gt'])
    labels = (gt.parent / 'labels.txt').read_text(encoding='utf-8-sig').splitlines()
    if tuple(labels) != class_names():
        raise ValueError(f'{seq}: GT labels must follow the configured class order {class_names()!r}')
    cached = {}
    cache_path = root / 'detections' / f'{seq}_high_low_detections.jsonl'
    with cache_path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            by_class = {}
            for detection in row['low_boxes']:
                try:
                    class_id = normalize_prediction_class(detection)
                except ValueError:
                    # Missing/unknown classes cannot provide same-class coverage.
                    continue
                by_class.setdefault(class_id, []).append(detection)
            cached[row['frame_id']] = by_class
    totals = {}
    with gt.open(encoding='utf-8-sig') as f:
        for r in csv.reader(f):
            if not float(r[6]):
                continue
            cls = int(r[7])
            if not 1 <= cls <= len(labels):
                raise ValueError(f'{seq}: invalid GT class ID {cls}')
            box = tuple(map(float, r[2:6]))
            best = max((_iou(box, tuple(float(b[k]) for k in ('x', 'y', 'w', 'h')))
                        for b in cached[int(r[0])].get(cls, [])), default=0)
            counts = totals.setdefault(cls, [0, 0])
            counts[0] += 1
            counts[1] += best >= .5
    return [{'Dataset': seq, 'GT class': labels[c-1], 'GT boxes': n,
             'GT boxes with a cached detection at IoU>=0.5': hit,
             'Coverage (%)': 100 * hit/n, 'GT boxes without overlap': n-hit}
            for c, (n, hit) in sorted(totals.items())]


def build_dataset_report(root):
    root = Path(root)
    metadata_path = root / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    tracker_names = metadata.get('tracker_names', {kind: kind + '_replay' for kind in METHODS})
    inventory = json.loads((root / 'inventory.json').read_text())
    phases = json.loads((root / 'phase_results.json').read_text())
    sequences = [i['sequence'] for i in inventory]
    detailed = {}
    for kind, label in METHODS.items():
        with (root / 'eval' / tracker_names[kind] / 'pedestrian_detailed.csv').open() as f:
            detailed[label] = {r['seq']: extract_metrics(r) for r in csv.DictReader(f)}
    per_rows, speed_rows, stage_rows, sections, coverage_rows = [], [], [], [], []
    for idx, seq in enumerate(sequences):
        values = {label: detailed[label][seq] for label in METHODS.values()}
        perf = performance_table(values)
        speeds = [speed_row(label, [phases[tracker_names[kind]][idx]]) for kind, label in METHODS.items()]
        for label in METHODS.values():
            per_rows.append({'Dataset': seq, 'Method': label, 'Frames': inventory[idx]['frames'], **values[label]})
        for row in speeds:
            speed_rows.append({'Dataset': seq, 'Frames': inventory[idx]['frames'], **row})
        for kind, label in METHODS.items():
            group = phases[tracker_names[kind]][idx]
            for stage in [*STAGES, 'other']:
                seconds = sum(p['other_seconds'] if stage == 'other' else p['stage_seconds'][stage] for p in group[1:])
                stage_rows.append({'Dataset': seq, 'Method': label, 'Stage': stage,
                                   'Seconds': seconds, 'ms/frame': seconds * 1000 / group[0], 'Frames': group[0]})
        csvout(root / 'summary' / 'datasets' / seq / 'performance.csv', perf)
        csvout(root / 'summary' / 'datasets' / seq / 'speed.csv', speeds)
        sections += [f'## {seq}', '', f"完整帧数：{inventory[idx]['frames']}；容器标称帧数：{inventory[idx]['container_frames']}；GT 框数：{inventory[idx]['gt_rows']}；GT ID 数：{len(inventory[idx]['gt_ids'])}。按顺序解码直到视频结束，评测与视频导出均采用实际解码帧数。", '', table(perf), '', table(speeds), '']
        for label in METHODS.values():
            v = values[label]
            miss = 100 * v['FN'] / (v['TP'] + v['FN'])
            sections.append(f"- {label}：FN 占有效 GT 的 {miss:.2f}%，FP {v['FP']}，IDSW {v['IDSW']}；HOTA {v['HOTA']:.2f}，AssA {v['AssA']:.2f}，IDF1 {v['IDF1']:.2f}，MOTA {v['MOTA']:.2f}。")
        lowest = min(values, key=lambda label: values[label]['FP'])
        v = values[lowest]
        if v['FN'] / (v['TP'] + v['FN']) > .3:
            sections += ['', f"该数据集 {lowest} 的 FP 最低（{v['FP']}），同时 FN 达 {v['FN']}，存在大量漏检。低 FP 伴随较少有效输出，不能单独解释为性能更好；仅凭本次观测不能断定具体参数与漏检之间的因果关系。"]
        sections.append('')
        coverage = detection_coverage(root, inventory[idx])
        coverage_rows.extend(coverage)
        sections += ['检测缓存对各类 GT 的同类空间覆盖诊断（检查每个 GT 是否有类别相同且 IoU≥0.5 的低阈值检测；缺失或未知类别不计覆盖，允许同一检测覆盖多个 GT，非一对一检测召回指标）：', '', table(coverage), '']
    mean = {label: {m: sum(detailed[label][s][m] for s in sequences) / len(sequences)
                    for m in METRICS} for label in METHODS.values()}
    mean_perf = performance_table(mean)
    mean_speed = []
    for label in METHODS.values():
        rows = [r for r in speed_rows if r['Method'] == label]
        mean_speed.append({'Method': label, **{k: sum(r[k] for r in rows) / len(rows)
                           for k in rows[0] if k not in {'Dataset', 'Method', 'Frames'}}})
    csvout(root / 'summary/per_dataset_metrics.csv', per_rows)
    csvout(root / 'summary/per_dataset_speed.csv', speed_rows)
    csvout(root / 'summary/per_dataset_stages.csv', stage_rows)
    csvout(root / 'summary/detection_gt_coverage.csv', coverage_rows)
    csvout(root / 'summary/performance_macro_mean.csv', mean_perf)
    csvout(root / 'summary/speed_macro_mean.csv', mean_speed)
    deltas = []
    for scope in [*sequences, 'MACRO_MEAN']:
        vals = mean if scope == 'MACRO_MEAN' else {label: detailed[label][scope] for label in METHODS.values()}
        for baseline in ('BoT-SORT', 'ByteTrack', 'OC-SORT'):
            for metric in METRICS:
                a, b = vals['MSDC-ELT'][metric], vals[baseline][metric]
                deltas.append({'Dataset': scope, 'Method': 'MSDC-ELT', 'Baseline': baseline,
                               'Metric': metric, 'Direction': '↓' if metric in LOWER_BETTER else '↑',
                               'Difference': a-b, 'Relative change (%)': 100*(a-b)/abs(b) if b else 'N/A'})
    csvout(root / 'summary/msdc_deltas.csv', deltas)
    report = root / 'REPORT.md'
    original = report.read_text()
    a, b = mean['MSDC-ELT'], mean['BoT-SORT']
    joint = [
        '## 重点指标联合比较', '',
        f"{len(sequences)} 个序列等权平均下，MSDC-ELT 相对 BoT-SORT：HOTA {a['HOTA']-b['HOTA']:+.2f}、AssA {a['AssA']-b['AssA']:+.2f}、IDF1 {a['IDF1']-b['IDF1']:+.2f}、MOTA {a['MOTA']-b['MOTA']:+.2f} 个百分点；每序列平均 IDSW {a['IDSW']-b['IDSW']:+.2f}、FP {a['FP']-b['FP']:+.2f}、FN {a['FN']-b['FN']:+.2f}。关联指标与输出覆盖率需要共同解读，平均也不能代替各场景结果。", '',
        '本次使用当前项目实现和参数，具体数值见 config.json / effective_config。ByteTrack 首轮代价为 1−IoU×score，不能把代价上限直接解释成 IoU 下限；第二轮及未确认关联保留库内阈值。本次没有进行参数干预实验。BoT-SORT 包含其内部 GMC 更新耗时，ReID 关闭；MSDC 计时含 lifecycle update，并将实测诊断生成/导出时间归入 write_export。', '',
        '速度为同一设备、依次运行的一次完整测量，未进行重复计时或给出置信区间。Overall FPS 包含共享检测成本，Tracker FPS 只反映 update 调用；纯跟踪加速不会按同样倍数转化为整体加速。', '']
    for label in METHODS.values():
        groups = phases[tracker_names[next(k for k, v in METHODS.items() if v == label)]]
        s = speed_row(label, groups)
        detection = sum(g[1]['stage_seconds']['low_detection'] for g in groups)
        joint.append(f"- {label}：全部视频 Overall FPS {s['Overall FPS']:.2f}，Tracker FPS {s['Tracker FPS']:.2f}；低阈值检测占整体耗时 {100*detection/s['Total Time (s)']:.2f}%。")
    joint += ['', '定位与标注来源限制：检测从原视频独立推理，未读取 GT；逐序列 GT/检测坐标重合审计见后文。GT 的制作及人工修订流程尚未核实。若高度重合，LocA/MOTP 的解释存在限制，不能作为关联能力的主要依据；结果限定于所提供 GT。', '']
    intro = [f'# {len(sequences)} 个序列的统一评测', '',
             '等权平均（macro mean）对每个序列的每项指标取算术平均，计数项也表示每序列平均次数。下方总体聚合（pooled）由 TrackEval 跨序列汇总，计数为总和，比例按各指标的原始定义组合；二者不可混用。',
             '速度等权平均中的各列分别取各序列算术平均，平均 FPS 不等于平均帧数除以平均时间。总体速度使用总帧数/累计实测时间，适合衡量本次全部数据的处理吞吐率。比例指标为百分数，负 MOTA 按公式保留。时间/延迟 ↓，FPS ↑。', '',
             '## 表 1：各序列等权平均跟踪性能', '', table(mean_perf), '',
             '## 表 2：各序列等权平均速度', '', table(mean_speed), '', *joint, *sections,
             '## 汇总 CSV', '',
             '- `summary/per_dataset_metrics.csv`：各数据集完整 17 指标。',
             '- `summary/per_dataset_speed.csv`、`summary/per_dataset_stages.csv`：各数据集速度与分阶段耗时（含完整帧数）。',
             '- `summary/performance_macro_mean.csv`、`summary/speed_macro_mean.csv`：各序列等权平均。',
             '- `summary/performance.csv`、`summary/speed.csv`、`summary/stages.csv`：全部数据总体聚合。',
             '- `summary/msdc_deltas.csv`：MSDC 相对各 baseline 的差值与相对变化；比例指标差值单位为百分点。', '',
             '# 全部序列总体聚合及运行说明', '']
    report.write_text('\n'.join(intro) + original)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    build_dataset_report(parser.parse_args().run_root)
