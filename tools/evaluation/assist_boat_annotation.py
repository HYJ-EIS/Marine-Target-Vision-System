"""Propagate a visually supplied ship box, saving every-frame crops for review.
This is assisted annotation, not MSDC output or an evaluation result.
"""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from cv_utils import imwrite_unicode


def propagate(video, output, box):
    output.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video)); ok,frame=cap.read()
    if not ok: raise ValueError('Cannot decode input')
    x,y,w,h=map(int,box); template=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
    fixed_template=template.copy()
    rows=[]; tiles=[]; page=0; f=0
    while ok:
        f+=1; gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        score=1.
        if f>1:
            margin=max(30,round(w*.5)); sx=max(0,x-margin); sy=max(0,y-margin)
            ex=min(frame.shape[1],x+w+margin); ey=min(frame.shape[0],y+h+margin)
            if frame.shape[1]>1000:
                sx=0; sy=frame.shape[0]//2; ex=frame.shape[1]; ey=frame.shape[0]
                template=fixed_template
            result=cv2.matchTemplate(gray[sy:ey,sx:ex],template,cv2.TM_CCOEFF_NORMED)
            _,score,_,loc=cv2.minMaxLoc(result); x=sx+loc[0]; y=sy+loc[1]
            if frame.shape[1]<=1000: template=gray[y:y+h,x:x+w].copy()
        rows.append({'frame':f,'x':x,'y':y,'w':w,'h':h,'match_score':score,'source':'visual_seed_and_per_frame_template_tracking'})
        # All frames are retained for visual inspection, with context around the box.
        px=max(0,x-25); py=max(0,y-25); crop=frame[py:min(frame.shape[0],y+h+25),px:min(frame.shape[1],x+w+25)].copy()
        cv2.rectangle(crop,(x-px,y-py),(x-px+w,y-py+h),(0,255,0),1)
        tile=cv2.resize(crop,(200,130)); cv2.putText(tile,f'{f}  {score:.2f}',(4,15),0,.45,(0,0,255),1)
        tiles.append(tile)
        if len(tiles)==48:
            imwrite_unicode(str(output/f'review_{page:02d}.jpg'),np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,48,6)])); tiles=[]; page+=1
        ok,frame=cap.read()
    cap.release()
    if tiles:
        while len(tiles)<48: tiles.append(np.zeros((130,200,3),np.uint8))
        imwrite_unicode(str(output/f'review_{page:02d}.jpg'),np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,48,6)]))
    (output/'boat_boxes.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'seed.json').write_text(json.dumps({'frame':1,'xywh':box,'source':'assistant_visual_localization','video':str(video)},indent=2)+'\n')
    print(video.name,len(rows),'min_score',min(r['match_score'] for r in rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--video',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--box',type=int,nargs=4,required=True); a=p.parse_args(); propagate(a.video,a.output,a.box)
