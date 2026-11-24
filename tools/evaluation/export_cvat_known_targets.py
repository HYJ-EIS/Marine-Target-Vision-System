"""MSDC pre-annotation with one known physical target per requested class."""
import argparse
import csv
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def fill_short_gaps(frames, labels, max_gap=2):
    """Interpolate bounded gaps independently per known target; never extrapolate."""
    output={f:{c:dict(b) for c,b in boxes.items()} for f,boxes in frames.items()}
    audit=[]
    for label in labels:
        observed=sorted(f for f in frames if label in frames[f])
        for left,right in zip(observed,observed[1:]):
            if not 1<=right-left-1<=max_gap: continue
            a,b=frames[left][label],frames[right][label]
            for f in range(left+1,right):
                alpha=(f-left)/(right-left)
                box=dict(a)
                for k in ('x','y','w','h'): box[k]=(1-alpha)*a[k]+alpha*b[k]
                box['confidence']=min(a.get('confidence',1),b.get('confidence',1))
                box['interpolated']=True
                output.setdefault(f,{})[label]=box
                audit.append({'frame':f,'class':label,'left_frame':left,'right_frame':right})
    return output,audit


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run(video,root,labels):
    import cv2
    from target_module.image_detect_module.config import Config
    from target_module.image_detect_module.target_detection import get_detector
    from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker
    from target_module.image_detect_module.utils.msdc_detection import run_msdc_low_threshold_detection,split_msdc_high_from_low_boxes
    from target_module.image_detect_module.utils.tracking_update import update_tracking_for_frame
    from tools.evaluation.export_mot_results import format_mot_result_line
    from tools.evaluation.render_mot_video import render_mot_video
    from tools.evaluation.stage_timing import StageTiming
    if len(set(labels))!=len(labels) or any(x not in Config.CLASSES for x in labels): raise ValueError('Invalid labels')
    root.mkdir(parents=True,exist_ok=False); (root/'gt').mkdir()
    save(root/'status.json',{'status':'partial'})
    save(root/'config.json',{k:v for k,v in vars(Config).items() if k.isupper()})
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError('Cannot open video')
    width,height=int(cap.get(3)),int(cap.get(4)); fps=cap.get(cv2.CAP_PROP_FPS); expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'VIDEO {video.name}: {width}x{height}, {fps}fps, reported={expected}',flush=True)
    timing=StageTiming(); detector=get_detector()
    if detector.processor is None: raise RuntimeError('No detector')
    providers=detector.processor.visible_detector.session.get_providers()
    print('PROVIDERS',providers,flush=True)
    tracker=MSDCLifecycleTracker(config=Config,file_type='visible',debug_dir=root/'diagnostics',processor=detector.processor)
    frames={}; previous={}; n=0; ambiguous=[]
    with (root/'raw_msdc.txt').open('w') as raw, (root/'frames.jsonl').open('w') as audit, (root/'detections.jsonl').open('w') as detections:
        try:
            while True:
                t=time.perf_counter(); ok,frame=cap.read(); timing.add('read_decode',t)
                if not ok: break
                n+=1
                t=time.perf_counter(); all_low=run_msdc_low_threshold_detection(detector,frame,'visible'); timing.add('low_detection',t)
                t=time.perf_counter(); low=[b for b in all_low if b.get('class') in labels]; high=split_msdc_high_from_low_boxes(low,'visible'); timing.add('high_split',t)
                t=time.perf_counter(); boxes=update_tracking_for_frame('msdc_elt',None,tracker,detector,frame,n-1,'visible',high,low); timing.add('tracker',t)
                t=time.perf_counter(); chosen={}
                for b in boxes: raw.write(format_mot_result_line(n,b)+'\n')
                for tid,label in enumerate(labels,1):
                    candidates=[b for b in boxes if b.get('class')==label]
                    if len(candidates)>1: ambiguous.append({'frame':n,'class':label,'candidates':len(candidates)})
                    if not candidates: continue
                    b=dict(max(candidates,key=lambda b:(b['track_id']==previous.get(label),float(b.get('confidence',0)))))
                    previous[label]=b['track_id']; b['raw_track_id']=b['track_id']; b['track_id']=tid
                    x=max(0.,min(width,float(b['x']))); y=max(0.,min(height,float(b['y'])))
                    r=max(x,min(width,float(b['x'])+float(b['w']))); bottom=max(y,min(height,float(b['y'])+float(b['h'])))
                    b.update(x=x,y=y,w=r-x,h=bottom-y)
                    if b['w']>0 and b['h']>0: chosen[label]=b
                frames[n]=chosen
                audit.write(json.dumps({'frame':n,'raw_boxes':boxes,'selected':chosen},ensure_ascii=False)+'\n')
                detections.write(json.dumps({'frame_id':n,'low_boxes':all_low,'high_boxes':high},ensure_ascii=False)+'\n')
                timing.add('write_export',t); timing.frames=n
                if n%100==0: print(f'PROGRESS {video.name} {n}/{expected}',flush=True)
        finally: cap.release()
    timing.write(root/'tracking.timing.json',providers=providers)
    # Independent complete decoding pass guards against incomplete output.
    check=cv2.VideoCapture(str(video)); verified=0
    while check.grab(): verified+=1
    check.release()
    if verified!=n or n==0 or (expected>0 and expected!=n): raise ValueError(f'Frame mismatch {n}/{verified}/{expected}')
    completed,interpolated=fill_short_gaps(frames,labels)
    save(root/'interpolated_frames.json',interpolated); save(root/'ambiguous_frames.json',ambiguous)
    gaps=[]
    for label in labels:
        start=None
        for f in range(1,n+2):
            missing=f<=n and label not in completed[f]
            if missing and start is None: start=f
            if not missing and start is not None: gaps.append({'class':label,'start':start,'end':f-1,'length':f-start}); start=None
    with (root/'missing_ranges.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['class','start','end','length']); w.writeheader(); w.writerows(gaps)
    with (root/'gt/gt.txt').open('w') as gt,(root/'annotations_mot.txt').open('w') as mot,(root/'class_cache.jsonl').open('w') as cache:
        for f,targets in sorted(completed.items()):
            cache.write(json.dumps({'frame_id':f,'boxes':list(targets.values())})+'\n')
            for label in labels:
                if label not in targets: continue
                b=targets[label]; tid=labels.index(label)+1
                gt.write(f"{f},{tid},{b['x']:.2f},{b['y']:.2f},{b['w']:.2f},{b['h']:.2f},1,{tid},1\n")
                mot.write(format_mot_result_line(f,b)+'\n')
    (root/'gt/labels.txt').write_text('\n'.join(labels)+'\n')
    (root/'seqinfo.ini').write_text(f'[Sequence]\nname={video.stem}\nimDir=img1\nframeRate={fps:g}\nseqLength={n}\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n')
    with zipfile.ZipFile(root/'cvat_mot_1_1.zip','w',zipfile.ZIP_DEFLATED) as z:
        for rel in ['gt/gt.txt','gt/labels.txt','seqinfo.ini']: z.write(root/rel,rel)
    render_mot_video(video,root/'annotations_mot.txt',root/'visualizations/annotated.mp4',file_type='visible',class_source='cache',detections_cache=root/'class_cache.jsonl',progress_interval=250)
    check=cv2.VideoCapture(str(root/'visualizations/annotated.mp4')); rendered=0
    while check.grab(): rendered+=1
    check.release(); assert rendered==n
    summary={'status':'complete','input':str(video.resolve()),'frames':n,'fps':fps,'labels':labels,'annotated_boxes':{label:sum(label in r for r in completed.values()) for label in labels},'interpolated_boxes':{label:sum(r['class']==label for r in interpolated) for label in labels},'unannotated_frames':{label:sum(label not in r for r in completed.values()) for label in labels},'identity_policy':'One physical target per class, as specified by user; fixed export IDs. Raw MSDC IDs retained separately.','visibility':'1 is an unmeasured placeholder','source_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'metrics':'N/A: pre-annotation, no manual GT'}
    summary['pipeline_finished'] = True
    if any(count == 0 for count in summary['annotated_boxes'].values()):
        summary['status'] = 'needs_annotation_review'
    save(root/'summary.json',summary)
    (root/'IMPORT_README.md').write_text(f'# CVAT 导入\n\n用原视频 `{video.name}` 建立独立任务，起始帧 0，步长 1。创建标签：'+', '.join(labels)+'。使用 Upload annotations → MOT 1.1 上传 `cvat_mot_1_1.zip`。\n\nID/类别号按 labels.txt 的顺序从 1 开始。依据每类只有一个真实目标的先验合并原始轨迹 ID；原始输出保存在 raw_msdc.txt 与 frames.jsonl。仅对内部连续缺失 1–2 帧线性插值，长缺口及首尾不外推。插值记录见 interpolated_frames.json，剩余缺口见 missing_ranges.csv。visibility=1 是未测量占位值。该结果为待人工核验的预标注，不是人工 GT 或跟踪性能评测。\n\n格式依据：https://docs.cvat.ai/docs/dataset_management/formats/format-mot/\n',encoding='utf-8')
    save(root/'status.json',{'status':summary['status'],'pipeline_finished':True,'frames':n,'visualization_frames':rendered})
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--labels',nargs='+',required=True)
    a=p.parse_args(); run(a.input,a.output,a.labels)
