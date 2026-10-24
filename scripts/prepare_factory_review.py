"""Prepare factory head proposals for explicit visual review, never auto-label.

Source train/test groups stay separate. Unknown/occluded heads must be excluded
from binary PPE training. The resulting labels are crop-level supervision, not
independent person-detector or event ground truth.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.core import RunConfig
from app.inference import create_backend
from app.scenario_models import person_regions,pose_head_proposal
from app.video import MODEL


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT/'data/datasets/factory-expansion')
    parser.add_argument('--output',type=Path,default=ROOT/'data/datasets/factory-review')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for name in ('frames','crops','cards','sheets'):(args.output/name).mkdir(exist_ok=True)
    cfg=RunConfig(fast_motion=False)
    backend=create_backend(MODEL,.35,config=cfg)
    backend.refine_head_pose=True
    manifest=json.loads((args.input/'manifest.json').read_text(encoding='utf-8'))
    candidates=[];frames=[]
    for source in manifest['files']:
        cap=cv2.VideoCapture(str(args.input/source['file']))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        fractions=(.3,.7) if source['source_split']=='train' else (.5,)
        for fraction in fractions:
            frame_index=round((count-1)*fraction);cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Cannot decode '+source['file'])
            frame_id=Path(source['file']).stem+f'-f{frame_index:05d}'
            frame_file=f'frames/{frame_id}.jpg'
            cv2.imwrite(str(args.output/frame_file),frame,[cv2.IMWRITE_JPEG_QUALITY,95])
            people=[p for p in backend.predict_people(frame) if p.confidence>=.35]
            frame_rows=[]
            for person in people:
                h,w=frame.shape[:2]
                proposal=pose_head_proposal(person,w,h)
                a,b,c,d=proposal.box if proposal else person_regions(person,w,h)['head']
                bw,bh=c-a,d-b
                x1,y1=max(0,int(a-bw*.15)),max(0,int(b-bh*.15))
                x2,y2=min(w,int(c+bw*.15)),min(h,int(d+bh*.15))
                if min(x2-x1,y2-y1)<12:continue
                identifier=f'{len(candidates)+1:04d}'
                crop=frame[y1:y2,x1:x2]
                crop_file=f'crops/{identifier}.png';cv2.imwrite(str(args.output/crop_file),crop)
                pa,pb,pc,pd=map(int,np.clip(person.box,[0,0,0,0],[w,h,w,h]))
                context=frame[pb:pd,pa:pc]
                if not context.size:continue
                card=Image.new('RGB',(360,300),'#f4f6f3');draw=ImageDraw.Draw(card)
                image=Image.fromarray(crop[:,:,::-1]);image.thumbnail((174,236))
                factor=min(174/image.width,236/image.height)
                image=image.resize((max(1,round(image.width*factor)),max(1,round(image.height*factor))))
                card.paste(image,(4,28))
                image=Image.fromarray(context[:,:,::-1]);image.thumbnail((174,236));card.paste(image,(182,28))
                draw.text((5,5),identifier+' '+frame_id,fill='#111111')
                draw.text((5,267),f'{x2-x1}x{y2-y1} px / '+('pose proposal' if proposal else 'REGION ONLY'),fill='#111111')
                draw.text((5,284),source['source_split']+' / UNREVIEWED',fill='#111111')
                card.save(args.output/f'cards/{identifier}.jpg',quality=92)
                row={'id':identifier,'source_video':source['file'],'source_sha256':source['sha256'],
                    'source_split':source['source_split'],'frame_index':frame_index,'time_s':frame_index/fps,
                    'frame_file':frame_file,'crop_file':crop_file,'person_box':list(person.box),
                    'person_confidence':person.confidence,'head_box':list(map(float,(a,b,c,d))),
                    'crop_box':[x1,y1,x2,y2],'proposal_source':'measured_face_landmarks' if proposal else 'person_region_only',
                    'pose_available':person.keypoints is not None,'review_state':'unreviewed',
                    'label':None,'geometry_reviewed':False,'reviewer':None,
                    'scope':'head crop only; not a complete frame/event annotation'}
                candidates.append(row);frame_rows.append(identifier)
            frames.append({'id':frame_id,'frame_file':frame_file,'source_video':source['file'],
                           'source_split':source['source_split'],'frame_index':frame_index,'candidates':frame_rows,
                           'all_visible_people_reviewed':False})
            (args.output/'proposals.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
            (args.output/'frames.json').write_text(json.dumps(frames,indent=2),encoding='utf-8')
            print('FACTORY_REVIEW_FRAME',frame_id,len(frame_rows),'total',len(candidates),flush=True)
        cap.release()
    for start in range(0,len(candidates),12):
        sheet=Image.new('RGB',(1440,900),'white')
        for offset,row in enumerate(candidates[start:start+12]):
            image=Image.open(args.output/f"cards/{row['id']}.jpg")
            sheet.paste(image,((offset%4)*360,(offset//4)*300))
        sheet.save(args.output/f'sheets/sheet-{start//12+1:02d}.jpg',quality=94)
    print('FACTORY_REVIEW_READY',len(candidates),'UNREVIEWED proposals',flush=True)


if __name__=='__main__':main()
