"""CPU presentation replay of saved production NN measurements. No inference or rules."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from app.core import Person
from app.preview_motion import PreviewMotion

KINDS={'helmet':'no_helmet','vest':'no_vest','smoking':'possible_smoking'}
TITLES={'helmet':'Каски','vest':'Сигнальные жилеты','smoking':'Возможное курение'}
STATE_TEXT={'helmet':{'violation':'NO HELMET','clear':'HELMET'},'vest':{'violation':'NO VEST','clear':'VEST'},'smoking':{'violation':'POSSIBLE SMOKING','clear':'SIGN CLEARED'}}
HEADER,FOOTER=74,64
FONTS={}
def font(size):
    if size not in FONTS:
        for name in [r'C:\Windows\Fonts\segoeui.ttf','DejaVuSans.ttf']:
            try:FONTS[size]=ImageFont.truetype(name,size);break
            except OSError:pass
        else:FONTS[size]=ImageFont.load_default()
    return FONTS[size]
def canvas_size(size):return (size[0],size[1]+HEADER+FOOTER)
def pattern_row(states,tid,pattern):
    matches=[row for key,row in states.get(str(tid),states.get(tid,{})).items() if key.split(':',1)[0]==KINDS[pattern]]
    return next((r for r in matches if r.get('state')=='violation'),matches[0] if matches else {})
def draw_people(frame,people,states,pattern,fresh):
    out=frame.copy();h,w=out.shape[:2]
    for p in people:
        row=pattern_row(states,p.track_id,pattern);state=row.get('state','unknown')
        if row.get('observed_state','unknown')=='unknown':state='unknown'
        color={'violation':(55,65,235),'clear':(80,190,95),'unknown':(60,190,220)}.get(state,(60,190,220))
        predicted=not fresh or not p.observed
        if predicted:color=(220,165,65)
        x1,y1,x2,y2=[int(v) for v in p.box];x1=max(0,min(w-1,x1));x2=max(0,min(w-1,x2));y1=max(0,min(h-1,y1));y2=max(0,min(h-1,y2))
        if x2<=x1 or y2<=y1:continue
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        text=('~ ' if predicted else '')+f'ID {p.track_id} | '+STATE_TEXT[pattern].get(state,'UNKNOWN')
        (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.53,1)
        tx=min(max(0,x1),max(0,w-tw-12));ty=y1-th-12 if y1>=th+14 else min(h-th-10,y2+2)
        cv2.rectangle(out,(tx,ty),(tx+tw+10,ty+th+10),(25,42,43),-1)
        cv2.putText(out,text,(tx+5,ty+th+4),cv2.FONT_HERSHEY_SIMPLEX,.53,(240,248,240),1,cv2.LINE_AA)
    return out

def decorate(frame,pattern,now,events,source_label='',duration=None):
    """All presentation bands are outside the original image, never over people."""
    h,w=frame.shape[:2];out=np.full((h+HEADER+FOOTER,w,3),(29,53,55),dtype=np.uint8);out[HEADER:HEADER+h]=frame
    # PIL touches only small external strips, not the source image.
    top=Image.new('RGB',(w,HEADER),(19,47,50));d=ImageDraw.Draw(top)
    d.text((20,8),'LinQ Safety · '+TITLES[pattern],font=font(25),fill=(242,248,239))
    d.text((20,43),source_label or 'Реальная видеозапись · демонстрация работы модели',font=font(15),fill=(183,211,199))
    bottom=Image.new('RGB',(w,FOOTER),(19,47,50));d=ImageDraw.Draw(bottom)
    fired=[e for e in events if e.get('kind')==KINDS[pattern] and float(e.get('alert_s',float('inf')))<=now+1e-8]
    ids=sorted({int(e.get('track_id',0)) for e in fired})
    text=f'Нарушений: {len(fired)}'+('  ·  ID '+', '.join(map(str,ids)) if ids else '')
    d.text((20,7),text,font=font(21),fill=(245,235,207) if fired else (226,240,232))
    timing=f'{now:.2f} с'+(f' / {duration:.2f} с' if duration is not None else '')
    box=d.textbbox((0,0),timing,font=font(18));d.text((w-(box[2]-box[0])-20,9),timing,font=font(18),fill=(224,237,230))
    d.text((20,38),'Красная рамка — подтверждённый паттерн · ~ ID — прогноз положения',font=font(14),fill=(183,211,199))
    out[:HEADER]=cv2.cvtColor(np.asarray(top),cv2.COLOR_RGB2BGR);out[HEADER+h:]=cv2.cvtColor(np.asarray(bottom),cv2.COLOR_RGB2BGR)
    return out

def transcode(raw,output):
    try:
        import imageio_ffmpeg
        exe=imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:exe=shutil.which('ffmpeg')
    if not exe:return False
    return subprocess.run([exe,'-hide_banner','-loglevel','error','-y','-i',str(raw),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=False).returncode==0

def replay(report_path,output,source_label='',overwrite=False):
    report_path=report_path.resolve();output=output.resolve();data=json.loads(report_path.read_text(encoding='utf-8-sig'));pattern=data['pattern'];source=Path(data['source']['path'])
    if output==report_path or output==source or output.suffix.lower()!='.mp4':raise ValueError('Choose a separate MP4 output')
    if output.exists() and not overwrite:raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as f:source_hash=hashlib.file_digest(f,'sha256').hexdigest()
    if source_hash!=data['source']['sha256']:raise ValueError('Source SHA-256 does not match the measured report')
    cap=cv2.VideoCapture(str(source));fps=float(data['source']['fps']);w=int(data['source']['width']);h=int(data['source']['height'])
    scale=min(1.,1280/max(w,h));size=(round(w*scale)//2*2,round(h*scale)//2*2);scale=size[0]/w
    target_size=canvas_size(size);raw=output.with_name('.'+output.stem+'.replay.raw.mp4')
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,target_size)
    if not cap.isOpened() or not writer.isOpened():raise RuntimeError('Source or output codec unavailable')
    measurements={int(m['source_frame']):m for m in data['measurements']};limit=int(data.get('decoded_frames',data['source']['frames']));motion=PreviewMotion();visual=None;states={};frames=0
    try:
        for index in range(limit):
            ok,frame=cap.read()
            if not ok:break
            now=index/fps;preview=cv2.resize(frame,size,interpolation=cv2.INTER_AREA);fresh=index in measurements
            if fresh:
                m=measurements[index];states=m['states'];people=[]
                for row in m['people']:
                    fields={**row,'box':tuple(row['box']),'keypoints':np.asarray(row['keypoints'],np.float32) if row.get('keypoints') is not None else None}
                    people.append(Person(**fields))
                visual={'time':float(m['source_time_s']),'people':people,'gray':cv2.cvtColor(preview,cv2.COLOR_BGR2GRAY),'width':w}
                motion.project(visual,preview,now)
                shown=[]
                for p in people:
                    points=p.keypoints.copy() if p.keypoints is not None else None
                    if points is not None:points[:,:2]*=scale
                    shown.append(replace(p,box=tuple(v*scale for v in p.box),keypoints=points))
            else:shown=motion.project(visual,preview,now) if visual else []
            content=draw_people(preview,shown,states,pattern,fresh)
            writer.write(decorate(content,pattern,now,data['events'],source_label,limit/fps));frames+=1
    finally:cap.release();writer.release()
    if frames!=limit:raise RuntimeError(f'Source decode ended at {frames}/{limit}; original report preserved')
    h264=transcode(raw,output)
    if h264:raw.unlink()
    else:raw.replace(output)
    metadata={'purpose':'CPU presentation replay of existing production measurements; not a precision benchmark','replay_of':str(report_path),'measured_report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'source_sha256':source_hash,'source_label':source_label,'output':str(output),'output_size':target_size,'fps':fps,'frames':frames,'duration_s':frames/fps,'h264':h264,'inference_calls':0,'rule_calls':0,'measurement_timestamps_unchanged':True,'unassociated_raw_proposals_shown':False,'display_only_projection':'PreviewMotion, never a rule vote; ~ ID marks projection','layout':'Complete original aspect image; header/footer outside the image','technical_details':'NN age, raw objects, confirmations and limitations remain in original immutable measured report','events':len(data['events'])}
    output.with_suffix('.replay.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(metadata,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-label',default='');p.add_argument('--overwrite',action='store_true');a=p.parse_args();replay(a.replay_report,a.output,a.source_label,a.overwrite)
if __name__=='__main__':main()
