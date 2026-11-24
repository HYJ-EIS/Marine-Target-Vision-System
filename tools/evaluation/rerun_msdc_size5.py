"""Re-evaluate six existing sequences with current V3 size 5 and three baselines.

Restore unavailable detection caches exclusively from RAW high_det/low_det
input logs. Reproduce old size-12 MOT before accepting a restored cache.
This runner never uses GT as tracker input. It evaluates the historical frame
range (full shaking videos, 4772/5400 frames of the two older sequences).
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from target_module.image_detect_module.config import Config, MSDC_V3_CONFIG
from tools.evaluation.detection_replay_benchmark import replay_tracker_from_cache, load_detection_cache
from tools.evaluation.run_unified_comparison import save, read, sha, csvout, count_video
from tools.evaluation.msdc_dataset_benchmark import read_video_info, write_diagnostics_for_tracker
from tools.evaluation.render_mot_video import render_mot_video
from tools.evaluation.motchallenge_eval import run_motchallenge_eval
from tools.evaluation.unified_dataset_report import extract_metrics
from tools.evaluation.tracking_duration_report import run_analysis

METHODS={'msdc_elt':'MSDC-ELT v3 (5 px)','official_botsort':'BoT-SORT',
         'bytetrack':'ByteTrack','official_ocsort':'OC-SORT'}
NAMES={k:('msdc_v3' if k=='msdc_elt' else k)+'_replay' for k in METHODS}


def restore_cache(stage_log, destination):
    """Copy original detection fields; never read low_only or tracker output."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with Path(stage_log).open() as source, destination.open('w') as out:
        for line in source:
            row=json.loads(line)
            if row['frame_idx']!=n:
                raise ValueError(f'Non-contiguous input log at {n}: {stage_log}')
            n+=1
            entry={'frame_id':n,'file_type':row['file_type'],'replay_detections':True,
                   'high_boxes':[],'low_boxes':[]}
            for box in row['boxes']:
                key={'high_det':'high_boxes','low_det':'low_boxes'}.get(box['stage'])
                if key:
                    entry[key].append({k:box[k] for k in ('x','y','w','h','confidence','class')})
            out.write(json.dumps(entry,ensure_ascii=False)+'\n')
    if not n:
        raise ValueError('Empty input log')
    return n


def inputs():
    shaking=ROOT/'results/shaking_four_msdc_v3/20260930_143800'
    old=ROOT/'results/msdc_gated_hungarian_compare/20260907_215952_formal_5400'
    items=[]
    for entry in read(shaking/'inventory.json'):
        items.append(dict(entry,source_run=str(shaking),cache_source=str(shaking/'detections'/f"{entry['sequence']}_high_low_detections.jsonl"),input_origin='original_detection_cache'))
    for seq,folder,n in [('DJI_20250711140128_0002_V','USV_MOT标注数据集',4772),
                         ('DJI_20250916100639_0001_V','UAV_USV_MOT标注数据集',5400)]:
        items.append(dict(sequence=seq,frames=n,source_run=str(old),
            video=str(ROOT.parent/folder/(seq+'.MP4')),
            gt=str(old/'mot_gt'/seq/'gt/gt.txt'),input_origin='restored_raw_detection_input_log',
            stage_log=str(old/'trackers/msdc_v3_replay/diagnostics'/seq/'stage_observations.jsonl')))
    return items


def main(root):
    root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    save(root/'status.json',dict(status='running',phase='preflight'))
    config={k:getattr(Config,k) for k in dir(Config) if not k.startswith('_') and not callable(getattr(Config,k))}
    if MSDC_V3_CONFIG['MSDC_OUTPUT_MIN_BOX_SIZE']!=5:
        raise ValueError('Current formal V3 must use size 5')
    save(root/'config.json',config)
    save(root/'run_metadata.json',dict(tracker_names=NAMES,msdc_variant='msdc_v3',baseline_input='all',
        config=config,evaluation_scope='historical six-sequence observation windows',
        timing_scope='fresh replay + MOT render; cached detection, low_detection/high_split=0 (not executed)',
        detector_rerun=False,render_max_width=1920))
    snapshot={}
    for folder in ['target_module/image_detect_module','tools/evaluation','third_party/trackeval']:
        for path in (ROOT/folder).rglob('*.py'):
            relative=path.relative_to(ROOT)
            dest=root/'source_snapshot'/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
            snapshot[str(relative)]=sha(path)
    save(root/'source_manifest.json',snapshot)
    inventory=[]
    for entry in inputs():
        seq=entry['sequence']; old=Path(entry['source_run'])
        video=Path(entry['video'])
        if not video.is_file():raise FileNotFoundError(video)
        info=read_video_info(video)
        cfg=configparser.ConfigParser();cfg.read(old/'mot_gt'/seq/'seqinfo.ini')
        fps=float(cfg['Sequence']['frameRate'])
        old_config=read(old/'effective_config/msdc_v3_replay.json')
        differences={k:[old_config.get(k),v] for k,v in MSDC_V3_CONFIG.items() if old_config.get(k)!=v}
        assert differences=={'MSDC_OUTPUT_MIN_BOX_SIZE':[12,5]},differences
        cache=root/'detections'/f'{seq}_high_low_detections.jsonl'
        cache.parent.mkdir(parents=True,exist_ok=True)
        if entry['input_origin']=='original_detection_cache':
            original=Path(entry['cache_source'])
            assert sha(original)==entry['cache_sha256']
            shutil.copy2(original,cache)
            assert sha(video)==entry['video_sha256']
        else:
            assert restore_cache(entry['stage_log'],cache)==entry['frames']
            entry['stage_log_sha256']=sha(Path(entry['stage_log']))
            entry['video_sha256']=sha(video)
        cached=load_detection_cache(cache)
        assert [r['frame_id'] for r in cached]==list(range(1,entry['frames']+1))
        gt_source=old/'mot_gt'/seq/'gt/gt.txt'
        dest=root/'mot_gt'/seq/'gt/gt.txt';dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(gt_source,dest)
        gt=list(csv.reader(dest.open(encoding='utf-8-sig')))
        assert all(1<=int(r[0])<=entry['frames'] for r in gt)
        cfg['Sequence']['seqLength']=str(entry['frames'])
        with (root/'mot_gt'/seq/'seqinfo.ini').open('w') as out:cfg.write(out)
        entry.update(cache_sha256=sha(cache),gt_sha256=sha(dest),gt=str(gt_source),
                     fps=fps,container_frames=info.frame_count,width=info.width,height=info.height,
                     gt_rows=len(gt),gt_ids=sorted({r[1] for r in gt}),gt_classes=sorted({r[7] for r in gt}),
                     formal_profile_diff=differences,old_seqinfo_length=int(read_ini(old/'mot_gt'/seq/'seqinfo.ini')['seqLength']))
        inventory.append(entry)
        print(f'[PREFLIGHT] {seq}: {entry["frames"]} frames, {entry["input_origin"]}',flush=True)
    save(root/'inventory.json',inventory)
    artifacts=[]; speed=[]; stages=[]; controls=[]
    for item in inventory:
        seq=item['sequence'];n=item['frames'];video=Path(item['video'])
        cache=root/'detections'/f'{seq}_high_low_detections.jsonl'
        save(root/'status.json',dict(status='running',phase='size12_reproduction',sequence=seq))
        # A diagnostic reproduction, excluded from reported formal speed/metrics.
        control=replay_tracker_from_cache(input_video=video,detection_cache=cache,
            output_root=root/'historical_reproduction/trackers',tracker_type='msdc_elt',variant='msdc_v3',
            env_override={'MSDC_OUTPUT_MIN_BOX_SIZE':'12'},seq_name=seq,file_type='visible',
            frame_rate=item['fps'],max_frames=n,progress_interval=1000,baseline_input='all')
        historical=Path(item['source_run'])/'trackers/msdc_v3_replay/data'/f'{seq}.txt'
        # MOT numeric coordinates and ordering must reproduce byte-for-byte.
        assert control.read_bytes()==historical.read_bytes(),f'Size12 reproduction changed: {seq}'
        controls.append(dict(sequence=seq,identical=True,old_mot_sha256=sha(historical),new_mot_sha256=sha(control),
                             frames=n,scope='diagnostic historical replay, excluded from formal speed'))
        save(root/'historical_reproduction/validation.json',controls)
        for kind,label in METHODS.items():
            name=NAMES[kind]
            save(root/'status.json',dict(status='running',phase='tracking',sequence=seq,method=name))
            print(f'[TRACKER] {seq} {label}',flush=True)
            mot=replay_tracker_from_cache(input_video=video,detection_cache=cache,output_root=root/'trackers',
                tracker_type=kind,variant='msdc_v3' if kind=='msdc_elt' else '',seq_name=seq,
                file_type='visible',frame_rate=item['fps'],max_frames=n,progress_interval=1000,
                baseline_input='all',preserve_config=kind!='msdc_elt')
            assert {k:getattr(Config,k) for k in config}==config
            timing=read(mot.with_suffix('.timing.json'));assert timing['processed_frames']==n
            assert sha(cache)==item['cache_sha256']
            write_diagnostics_for_tracker(root/'mot_gt'/seq/'gt/gt.txt',mot,root/'diagnostics'/seq,name)
            artifacts.append(dict(sequence=seq,tracker=name,MOT=str(mot),
                tracking_timing=str(mot.with_suffix('.timing.json')),per_frame_timing=str(mot.with_suffix('.frames.jsonl')),
                diagnostics=str(root/'diagnostics'/seq),MP4=str(root/'visualizations'/seq/f'{name}.mp4'),
                render_timing=str(root/'visualizations'/seq/f'{name}.timing.json'),
                TrackEval_summary=str(root/'eval'/name/'pedestrian_summary.txt')))
        csvout(root/'summary/artifacts.csv',artifacts)
    assert read(root/'effective_config/msdc_v3_replay.json')['MSDC_OUTPUT_MIN_BOX_SIZE']==5
    save(root/'status.json',dict(status='running',phase='TrackEval'))
    metrics=run_motchallenge_eval(root/'mot_gt',root/'trackers',root/'eval',
        trackers=list(NAMES.values()),sequences=[i['sequence'] for i in inventory],print_results=False)
    rows=[]
    for kind,label in METHODS.items():
        for row in csv.DictReader((root/'eval'/NAMES[kind]/'pedestrian_detailed.csv').open()):
            rows.append(dict(sequence=row['seq'],method=label,tracker=NAMES[kind],**extract_metrics(row),
                             MT=int(float(row['MT'])),PT=int(float(row['PT'])),ML=int(float(row['ML']))))
    csvout(root/'summary/mot_metrics.csv',rows)
    # Duration analysis only sees this new run, with no historical rows mixed in.
    run_analysis(root,root/'duration',1.0)
    for item in inventory:
        seq=item['sequence'];n=item['frames'];cache=root/'detections'/f'{seq}_high_low_detections.jsonl'
        for kind,label in METHODS.items():
            name=NAMES[kind];mot=root/'trackers'/name/'data'/f'{seq}.txt'
            video=root/'visualizations'/seq/f'{name}.mp4'
            save(root/'status.json',dict(status='running',phase='render',sequence=seq,method=name))
            render_mot_video(item['video'],mot,video,file_type='visible',max_frames=n,
                scale=min(1,1920/item['width']),class_source='cache',detections_cache=cache,progress_interval=1000)
            render=read(video.with_suffix('.timing.json'));replay=read(mot.with_suffix('.timing.json'))
            assert render['processed_frames']==n and count_video(video)==n
            total=replay['total_time_s']+render['total_time_s']
            tracker=replay['stage_seconds']['tracker']
            speed.append(dict(sequence=seq,tracker=name,frames=n,total_seconds=total,
                mean_frame_ms=total*1000/n,mean_fps=n/total,tracker_seconds=tracker,tracker_ms=tracker*1000/n,
                tracker_fps=n/tracker,replay_seconds=replay['total_time_s'],render_seconds=render['total_time_s'],
                scope='cached detection replay + rendering; excludes detector and evaluation'))
            for stage in replay['stage_seconds']:
                seconds=replay['stage_seconds'][stage]+render['stage_seconds'][stage]
                stages.append(dict(sequence=seq,tracker=name,stage=stage,seconds=seconds,ms_per_frame=1000*seconds/n,
                    applicability='not executed: cached input' if stage in ('low_detection','high_split') else 'measured'))
            stages.append(dict(sequence=seq,tracker=name,stage='other',seconds=replay['other_seconds']+render['other_seconds'],
                               ms_per_frame=1000*(replay['other_seconds']+render['other_seconds'])/n,applicability='measured residual'))
            csvout(root/'summary/speed.csv',speed);csvout(root/'summary/stages.csv',stages)
    pooled=[]
    for name in NAMES.values():
        rs=[s for s in speed if s['tracker']==name]
        n=sum(r['frames'] for r in rs);total=sum(r['total_seconds'] for r in rs);t=sum(r['tracker_seconds'] for r in rs)
        pooled.append(dict(tracker=name,frames=n,total_seconds=total,mean_frame_ms=1000*total/n,mean_fps=n/total,
                           tracker_seconds=t,tracker_ms=1000*t/n,tracker_fps=n/t))
    csvout(root/'summary/speed_pooled.csv',pooled)
    save(root/'validation.json',dict(passed=True,sequences=len(inventory),methods=len(METHODS),
         frames_per_method=sum(i['frames'] for i in inventory),annotated_videos=len(speed),
         all_visualizations_fully_decoded=True,all_size12_mot_identical=True,actual_size5_verified=True,
         cache_hashes_unchanged=True,config_restored=True,wall_seconds=time.perf_counter()-started))
    save(root/'provenance.json',dict(platform=platform.platform(),python=sys.version,detector_run=False,
         timing='sequential CPU tracking, rendering and decoding; no detector costs included',
         raw_input_restoration='high_det and low_det only; no tracker outputs or GT used',
         model_sha256=sha(Path(Config.ONNX_VISIBLE_MODEL_PATH))))
    save(root/'status.json',dict(status='complete',wall_seconds=time.perf_counter()-started))
    print(f'[COMPLETE] {root}',flush=True)


def read_ini(path):
    cfg=configparser.ConfigParser();cfg.read(path);return cfg['Sequence']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    args=parser.parse_args()
    try:main(args.run_root.resolve())
    except Exception as exc:
        if args.run_root.exists():save(args.run_root/'status.json',dict(status='failed_partial',error=str(exc)))
        raise
