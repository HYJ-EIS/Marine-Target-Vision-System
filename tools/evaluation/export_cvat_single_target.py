"""MS-DC-ELT pre-annotation for a video known to contain one target."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.experiments.run_msdc_ablation import FORMAL_V3_ENV


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--label', required=True)
    args = p.parse_args()
    os.environ.update(FORMAL_V3_ENV)
    # Importing the variants also loads the package's Config through __init__.
    # Refresh it before detector/tracker modules bind their Config references.
    import target_module.image_detect_module.config as config_module
    importlib.reload(config_module)
    import cv2
    from target_module.image_detect_module.config import Config
    from target_module.image_detect_module.target_detection import get_detector
    from target_module.image_detect_module.utils.lifecycle_tracker import MSDCLifecycleTracker
    from target_module.image_detect_module.utils.msdc_detection import (
        run_msdc_low_threshold_detection, split_msdc_high_from_low_boxes,
    )
    from target_module.image_detect_module.utils.tracking_update import update_tracking_for_frame
    from tools.evaluation.export_mot_results import format_mot_result_line
    from tools.evaluation.stage_timing import StageTiming
    from tools.evaluation.render_mot_video import render_mot_video

    if args.label not in Config.CLASSES:
        raise ValueError('Label must be in Config.CLASSES')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'gt').mkdir()
    (root / 'status.json').write_text('{"status":"partial"}\n')
    timing = StageTiming()
    detector = get_detector()
    if detector.processor is None or detector.processor.visible_detector is None:
        raise RuntimeError('Visible detector unavailable')
    providers = detector.processor.visible_detector.session.get_providers()
    print('PROVIDERS', providers, flush=True)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError('Cannot open input video')
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = MSDCLifecycleTracker(config=Config, file_type='visible',
        debug_dir=root / 'diagnostics', processor=detector.processor)
    config = {k: v for k, v in vars(Config).items() if k.isupper()}
    (root / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    raw_ids, rejected = Counter(), Counter()
    missing, multi = [], []
    previous_id = None
    count = 0
    with (root / 'raw_msdc.txt').open('w') as raw, \
         (root / 'single_target.txt').open('w') as selected, \
         (root / 'gt/gt.txt').open('w') as gt, \
         (root / 'frames.jsonl').open('w') as audit, \
         (root / 'class_cache.jsonl').open('w') as cache:
        try:
            while True:
                t = time.perf_counter()
                ok, frame = cap.read()
                timing.add('read_decode', t)
                if not ok:
                    break
                count += 1
                t = time.perf_counter()
                all_low = run_msdc_low_threshold_detection(detector, frame, 'visible')
                timing.add('low_detection', t)
                t = time.perf_counter()
                rejected.update(b.get('class', 'unknown') for b in all_low if b.get('class') != args.label)
                low = [b for b in all_low if b.get('class') == args.label]
                high = split_msdc_high_from_low_boxes(low, 'visible')
                timing.add('high_split', t)
                t = time.perf_counter()
                boxes = update_tracking_for_frame('msdc_elt', None, tracker,
                    detector, frame, count - 1, 'visible', high, low)
                timing.add('tracker', t)
                t = time.perf_counter()
                boxes = [b for b in boxes if b.get('class') == args.label]
                for b in boxes:
                    raw.write(format_mot_result_line(count, b) + '\n')
                    raw_ids[int(b['track_id'])] += 1
                chosen = None
                if boxes:
                    # Keep the previous raw identity when available; confidence breaks ties.
                    chosen = dict(max(boxes, key=lambda b: (
                        b['track_id'] == previous_id, float(b.get('confidence', 0)))))
                    previous_id = chosen['track_id']
                    x = max(0., min(width, float(chosen['x'])))
                    y = max(0., min(height, float(chosen['y'])))
                    right = max(x, min(width, float(chosen['x']) + float(chosen['w'])))
                    bottom = max(y, min(height, float(chosen['y']) + float(chosen['h'])))
                    chosen.update(x=x, y=y, w=right-x, h=bottom-y, track_id=1)
                    if chosen['w'] <= 0 or chosen['h'] <= 0:
                        chosen = None
                if chosen:
                    selected.write(format_mot_result_line(count, chosen) + '\n')
                    gt.write(f"{count},1,{chosen['x']:.2f},{chosen['y']:.2f},{chosen['w']:.2f},{chosen['h']:.2f},1,1,1\n")
                else:
                    missing.append(count)
                if len(boxes) > 1:
                    multi.append(count)
                audit.write(json.dumps({'frame_id': count, 'raw_boxes': boxes,
                    'selected_raw_id': previous_id if chosen else None,
                    'exported': bool(chosen)}, ensure_ascii=False) + '\n')
                cache.write(json.dumps({'frame_id': count, 'boxes': [chosen] if chosen else []}) + '\n')
                timing.add('write_export', t)
                timing.frames = count
                if count % 250 == 0:
                    print(f'PROGRESS {count}/{expected}, annotated={count-len(missing)}', flush=True)
        finally:
            cap.release()
    timing.write(root / 'tracking.timing.json', providers=providers,
                 expected_frames=expected, decoded_frames=count)
    if count != expected:
        raise RuntimeError(f'Partial decode: {count}/{expected}')
    (root / 'gt/labels.txt').write_text(args.label + '\n')
    (root / 'seqinfo.ini').write_text(
        f'[Sequence]\nname={Path(args.input).stem}\nimDir=img1\nframeRate={fps:g}\n'
        f'seqLength={count}\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n')
    package = root / 'cvat_mot_1_1.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('gt/gt.txt', 'gt/labels.txt', 'seqinfo.ini'):
            z.write(root / name, name)
    render_mot_video(args.input, root / 'single_target.txt',
        root / 'visualizations/single_target.mp4', file_type='visible',
        class_source='cache', detections_cache=root / 'class_cache.jsonl', progress_interval=500)
    summary = dict(status='complete', input=str(Path(args.input).resolve()),
        frames=count, fps=fps, width=width, height=height,
        annotated_frames=count-len(missing), missing_frames=missing,
        multiple_raw_tracks_frames=multi, raw_track_counts=dict(raw_ids),
        filtered_detection_counts=dict(rejected),
        identity_policy='User-provided single target prior; exported ID=1; prefer previous raw ID, then confidence.',
        visibility_policy='1 is an unmeasured placeholder, not estimated visibility.',
        interpolation='No boxes generated for missing frames.',
        metrics='N/A: pre-annotation without ground truth; not a formal benchmark.')
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (root / 'status.json').write_text('{"status":"complete"}\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('missing_frames', 'multiple_raw_tracks_frames')}, ensure_ascii=False), flush=True)
    print('PACKAGE', package.resolve(), flush=True)


if __name__ == '__main__':
    main()
