"""Full-video, shared-detection comparison with unchanged project tracker config."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from target_module.image_detect_module.config import Config
from tools.evaluation.detection_replay_benchmark import dump_video_detections, replay_tracker_from_cache, load_detection_cache
from tools.evaluation.msdc_dataset_benchmark import resolve_single_sequence_dataset, read_video_info, VideoInfo, write_motchallenge_gt_sequence, write_diagnostics_for_tracker, _iou
from tools.evaluation.motchallenge_eval import run_motchallenge_eval
from tools.evaluation.render_mot_video import render_mot_video
from tools.evaluation.summarize_replay_comparison import METRICS, COUNTS, LOWER_BETTER, STAGES
from tools.evaluation.normalize_comparison_csv import normalize
from target_module.image_detect_module.paper_config import paper_environment, seed_paper_runtime

METHODS = {'official_botsort':'BoT-SORT','msdc_elt':'MSDC-ELT','bytetrack':'ByteTrack','official_ocsort':'OC-SORT'}

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str)+'\n', encoding='utf-8')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def csvout(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def count_video(path):
    cap=cv2.VideoCapture(str(path)); n=0
    if not cap.isOpened(): raise ValueError(f'Cannot open {path}')
    while cap.grab(): n+=1
    cap.release()
    return n

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root',type=Path,required=True)
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--msdc-variant', choices=['msdc_v3'], default='', help='Load the existing formal profile; omitted means unchanged base Config')
    args=ap.parse_args(); root=args.run_root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    save(root/'status.json',{'status':'running','started':time.strftime('%Y-%m-%d %H:%M:%S')})
    if os.environ.get('PAPER_EXPERIMENT_SPEC'):
        save(root/'experiment_environment.json', {
            **paper_environment(), **seed_paper_runtime(),
            'experiment_spec': str(Path(os.environ['PAPER_EXPERIMENT_SPEC']).resolve()),
        })
    config={k:getattr(Config,k) for k in dir(Config) if not k.startswith('_') and not callable(getattr(Config,k))}
    save(root/'config.json', config)
    videos=sorted(p for p in args.dataset_root.rglob('*') if p.suffix.lower() in {'.mp4','.avi','.mov','.mpeg','.mpg','.m4v'})
    if not videos: raise ValueError('No videos')
    tracker_names={k:(args.msdc_variant if k=='msdc_elt' and args.msdc_variant else k)+'_replay' for k in METHODS}
    names=list(tracker_names.values())
    msdc_note = f'MSDC 加载当前项目 {args.msdc_variant} 配置' if args.msdc_variant else 'MSDC 使用基础 Config'
    save(root/'run_metadata.json', {'tracker_names':tracker_names, 'msdc_variant':args.msdc_variant, 'baseline_input':'all', 'config':config, 'dataset_root':str(args.dataset_root.resolve()), 'timing_scope':'shared detection + replay + MOT render; measured diagnostic/export time excluded from tracker'})
    seqs=[]; phases={k:[] for k in names}; artifacts=[]; inventory=[]; audits=[]
    for video in videos:
        spec=resolve_single_sequence_dataset(video.parent, video=video)
        seq=spec.seq_name
        if seq in seqs: raise ValueError('Duplicate sequence name')
        seqs.append(seq); info=read_video_info(video)
        n=count_video(video)
        print(f'[FULL DECODE] {seq}: {n}; container={info.frame_count}',flush=True)
        rows=list(csv.reader(spec.gt_file.open(encoding='utf-8-sig')))
        assert rows and all(1<=int(r[0])<=n for r in rows)
        assert len({(r[0],r[1]) for r in rows})==len(rows)
        assert all(float(r[4])>0 and float(r[5])>0 for r in rows)
        write_motchallenge_gt_sequence(spec,root/'mot_gt',VideoInfo(info.fps,info.width,info.height,n))
        assert sha(spec.gt_file)==sha(root/'mot_gt'/seq/'gt/gt.txt')
        inv={'sequence':seq,'frames':n,'container_frames':info.frame_count,'video':str(video.resolve()),'video_sha256':sha(video),'gt':str(spec.gt_file.resolve()),'gt_sha256':sha(spec.gt_file),'gt_rows':len(rows),'gt_ids':sorted({r[1] for r in rows}),'gt_classes':sorted({r[7] for r in rows})}
        inventory.append(inv); save(root/'inventory.json',inventory)
        cache=root/'detections'/f'{seq}_high_low_detections.jsonl'
        dump_video_detections(input_video=video,output_cache=cache,file_type='visible',max_frames=0,progress_interval=500)
        from target_module.image_detect_module.target_detection import get_detector
        session=get_detector().processor.visible_detector.session
        save(root/'inference_environment.json', {'providers':session.get_providers(), 'input_shape':session.get_inputs()[0].shape})
        cached=load_detection_cache(cache)
        assert [r['frame_id'] for r in cached]==list(range(1,n+1))
        inv['cache_sha256']=sha(cache); save(root/'inventory.json',inventory)
        detection=read(cache.with_suffix('.timing.json'))
        best=[]; exact=0
        for r in rows:
            gt=tuple(map(float,r[2:6])); boxes=cached[int(r[0])-1]['low_boxes']
            coords=[tuple(float(b[k]) for k in ('x','y','w','h')) for b in boxes]
            best.append(max((_iou(gt,b) for b in coords),default=0))
            exact+=any(np.allclose(gt,b,rtol=0,atol=0.02) for b in coords)
        audits.append({'sequence':seq,'gt_boxes':len(rows),'matching_coordinates_within_0.02px':exact,'matched_coordinate_percent':100*exact/len(rows),'iou_at_least_0.95_percent':100*sum(x>=.95 for x in best)/len(best),'mean_best_iou_percent':100*float(np.mean(best))})
        save(root/'gt_detection_overlap.json',audits)
        for kind,label in METHODS.items():
            name=tracker_names[kind]
            print(f'[TRACKER] {label}',flush=True)
            variant=args.msdc_variant if kind=='msdc_elt' else ''
            mot=replay_tracker_from_cache(input_video=video,detection_cache=cache,output_root=root/'trackers',tracker_type=kind,variant=variant,seq_name=seq,file_type='visible',frame_rate=info.fps,max_frames=0,progress_interval=1000,baseline_input='all',preserve_config=not bool(variant))
            assert config=={k:getattr(Config,k) for k in config}
            assert sha(cache)==inv['cache_sha256']
            timing=read(mot.with_suffix('.timing.json'))
            assert timing['processed_frames']==n
            write_diagnostics_for_tracker(root/'mot_gt'/seq/'gt/gt.txt',mot,root/'diagnostics'/seq,name)
            out=root/'visualizations'/seq/f'{name}.mp4'
            render_mot_video(video,mot,out,file_type='visible',class_source='cache',detections_cache=cache,progress_interval=1000,
                             max_output_width=1920 if os.environ.get('PAPER_EXPERIMENT_SPEC') else 0)
            render=read(out.with_suffix('.timing.json'))
            assert render['processed_frames']==n and count_video(out)==n
            phases[name].append((n,detection,timing,render))
            artifacts.append({'method':label,'sequence':seq,'MOT':str(mot.relative_to(root)),'MP4':str(out.relative_to(root)),'tracker_timing':str(mot.with_suffix('.timing.json').relative_to(root)),'render_timing':str(out.with_suffix('.timing.json').relative_to(root)),'detection_cache':str(cache.relative_to(root)),'detection_timing':str(cache.with_suffix('.timing.json').relative_to(root)),'diagnostics':f'diagnostics/{seq}','TrackEval_summary':f'eval/{name}/pedestrian_summary.txt','TrackEval_detailed':f'eval/{name}/pedestrian_detailed.csv'})
            artifacts[-1]['per_frame_timing']=str(mot.with_suffix('.frames.jsonl').relative_to(root))
            artifacts[-1]['lifecycle_diagnostics']=f'trackers/{name}/diagnostics/{seq}' if kind=='msdc_elt' else 'N/A'
            save(root/'phase_results.json',phases)
    metrics=run_motchallenge_eval(root/'mot_gt',root/'trackers',root/'eval',trackers=names,sequences=seqs,print_results=False)
    performance=[{'Metric':m+(' ↓' if m in LOWER_BETTER else ' ↑'),**{METHODS[k]:metrics[tracker_names[k]][m] for k in METHODS}} for m in METRICS]
    speed=[]; stages=[]
    for kind,label in METHODS.items():
        name=tracker_names[kind]; ps=phases[name]; n=sum(p[0] for p in ps)
        total=sum(p['total_time_s'] for group in ps for p in group[1:]); tracker=sum(group[2]['stage_seconds']['tracker'] for group in ps)
        speed.append({'Method':label,'Total Time (s)':total,'Avg Time (ms/frame)':1000*total/n,'Overall FPS':n/total,'Tracker Time (s)':tracker,'Tracker Time (ms/frame)':1000*tracker/n,'Tracker FPS':n/tracker})
        for stage in [*STAGES,'other']:
            seconds=sum(p['other_seconds'] if stage=='other' else p['stage_seconds'][stage] for group in ps for p in group[1:])
            stages.append({'Method':label,'Stage':stage,'Seconds':seconds,'ms/frame':1000*seconds/n,'Frames':n})
        gt_count=sum(int(float(r[6]))!=0 for seq in seqs for r in csv.reader((root/'mot_gt'/seq/'gt/gt.txt').open()))
        assert metrics[name]['TP']+metrics[name]['FN']==gt_count
        assert metrics[name]['IDTP']+metrics[name]['IDFN']==gt_count
    csvout(root/'summary/performance.csv',performance); csvout(root/'summary/speed.csv',speed); csvout(root/'summary/stages.csv',stages); csvout(root/'summary/artifacts.csv',artifacts)
    def table(rows):
        keys=list(rows[0]); lines=['| '+' | '.join(keys)+' |','|'+'---|'*len(keys)]
        for row in rows: lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row.values())+' |')
        return lines
    lines=['# 四方法统一评测','',f'完整覆盖 {len(seqs)} 个视频、每方法 {sum(i["frames"] for i in inventory):,} 帧。原始 GT 原样复制，未删除标注、合并 ID 或抽样。','',
        f'四方法共享同一份低阈值检测缓存。三个 baseline 接收完整 low_boxes；MSDC 接收由同一缓存按 Config 拆分的高、低阈值框。{msdc_note}；其他方法保持当前 Config。没有根据 GT 调参。跟踪调用不读取 GT，配置及变体记录在 config.json、run_metadata.json 和 effective_config/。',
        'BoT-SORT/OC-SORT 使用项目 vendored 官方实现，ByteTrack 使用 supervision。类别相同才允许匹配，三类目标共同汇总，公开ID不按类别拆分。MOTP 是匹配框平均 IoU。比例指标乘 100；MOTA 若为负保留真实值，不截断。↑ 越高越好，↓ 越低越好。','',
        '## 表 1：跟踪性能对比','',*table(performance),'','## 表 2：速度对比','',*table(speed),'',
        '时间与延迟 ↓，FPS ↑。总耗时为共享检测、当前方法回放及 MOT 视频渲染的实测耗时之和，包含解码、检测、跟踪、渲染、编码和 MOT/缓存导出；检测只执行一次，但在各方法总时间中计入一次。多次视频读取、初始化和缓存读取也计入，因此属于离线多阶段速度，不等同于在线单流水线 FPS。纯跟踪计时包围 update 调用，并扣除 MSDC 内部实测的诊断生成/写盘 debug_s（归入 write_export）；原始 update 墙钟时间和扣除量保存在逐帧 frames.jsonl 和 timing.json。TrackEval、离线 GT 诊断和完整性校验不计入处理速度。','',
        '## 联合分析','']
    for s in speed:
        v=metrics[tracker_names[next(k for k,l in METHODS.items() if l==s['Method'])]]
        lines.append(f"- {s['Method']}：HOTA {v['HOTA']:.2f}，AssA {v['AssA']:.2f}，IDF1 {v['IDF1']:.2f}，MOTA {v['MOTA']:.2f}；IDSW {v['IDSW']}，FP {v['FP']}，FN {v['FN']}；Overall FPS {s['Overall FPS']:.2f}，Tracker FPS {s['Tracker FPS']:.2f}。")
    minfp=min(names,key=lambda k:metrics[k]['FP']); v=metrics[minfp]
    minfp_label=METHODS[next(k for k,n in tracker_names.items() if n==minfp)]
    lines+=['',f"FP 最低的方法为 {minfp_label}，FP={v['FP']}，FN={v['FN']}，漏检率为 {100*v['FN']/(v['TP']+v['FN']):.2f}%。低 FP 必须与漏检、关联指标及速度一起解释，不能单独作为优越性证据。",'', '## GT 与定位指标限制','']
    for a in audits: lines.append(f"{a['sequence']}：{a['matching_coordinates_within_0.02px']}/{a['gt_boxes']} 个 GT 框与检测坐标在 0.02 像素内一致（{a['matched_coordinate_percent']:.2f}%），最佳 IoU≥0.95 占 {a['iou_at_least_0.95_percent']:.2f}%。")
    lines+=['检测缓存由原视频经项目 ONNX 模型独立推理生成，未读取 GT。GT 的原始标注制作流程未核实；若与检测框高度重合，LocA/MOTP 的定位解释受限，不能作为跟踪关联能力的主要依据。未标注帧按空 GT 参与评测；结果依赖所提供标注的完整性。','', '## 分阶段耗时','',*table(stages),'','## 文件路径','',
        '- `summary/performance.csv`：表 1；`summary/speed.csv`：表 2；`summary/stages.csv`：分阶段耗时。',
        '- `eval/motchallenge_summary.csv`：完整汇总；各方法 `pedestrian_detailed.csv`：逐序列及聚合原始 TrackEval 结果。原始 detailed CSV 沿用 TrackEval 的 0–100 百分比标度。',
        '- `summary/artifacts.csv`：各方法 MOT txt、MP4、TrackEval summary/detailed、诊断 CSV 和计时 JSON 路径。',
        '- `inventory.json`、`config.json`、`gt_detection_overlap.json`、`phase_results.json`、`provenance.json`：输入哈希、配置、重合审计、原始计时与环境。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(root/'provenance.json',{'platform':platform.platform(),'python':sys.version,'model_sha256':sha(Path(Config.ONNX_VISIBLE_MODEL_PATH)),'script_sha256':sha(Path(__file__)),'config_file_sha256':sha(ROOT/'target_module/image_detect_module/config.py'),'timing':'sequential full passes','baseline_input':'all','tracker_parameters':msdc_note+'; baselines unchanged Config'})
    normalize(root)
    from tools.evaluation.unified_dataset_report import build_dataset_report
    build_dataset_report(root)
    save(root/'validation.json',{'passed':True,'videos':len(seqs),'frames_per_method':sum(i['frames'] for i in inventory),'methods':names,'gt_identical':True,'shared_cache_hash_unchanged':True,'full_visualizations_decoded':True,'gt_count_id_and_clear_consistent':True,'algorithm_config_restored_after_variant':True})
    save(root/'status.json',{'status':'complete','videos':len(seqs),'frames_per_method':sum(i['frames'] for i in inventory),'all_videos_fully_decoded':True,'all_cache_frames_contiguous':True,'all_config_values_unchanged':True})
    print(f'[COMPLETE] {root}',flush=True)

if __name__=='__main__': main()
