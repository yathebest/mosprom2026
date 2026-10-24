"""Prepare RGB action clips from the authors' public Figshare archive + URFD.

Figshare labels are clip-level fall/ADL, never fabricated frame-level onset labels.
All locations, attempts and actions of one Figshare subject stay in one split.
URFD retains the previously frozen whole-video protocol.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
import zipfile

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ARTICLE='https://api.figshare.com/v2/articles/28596332'


def frames_from_video(path, destination, start_s=None, count=16):
    cap=cv2.VideoCapture(str(path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or n<8 or fps<=0:raise ValueError('Unreadable/too short RGB video')
        # New author videos are approximately 2s. Preserve their full chronology.
        positions=np.linspace(0,n-1,count).round().astype(int) if start_s is None else np.clip(
            np.round((start_s+np.arange(count)/8)*fps).astype(int),0,n-1)
        wanted=set(positions.tolist());images={};index=0
        while index<=int(positions.max()):
            ok,frame=cap.read()
            if not ok:break
            if index in wanted:
                images[index]=cv2.resize(frame,(224,224),interpolation=cv2.INTER_AREA)
            index+=1
        if any(int(i) not in images for i in positions):raise ValueError('Truncated RGB video')
        destination.mkdir(parents=True,exist_ok=True)
        for i,frame_index in enumerate(positions):
            if not cv2.imwrite(str(destination/f'{i:02d}.jpg'),images[int(frame_index)],
                               [cv2.IMWRITE_JPEG_QUALITY,88]):raise OSError('JPEG write failed')
        return {'source_frames':n,'source_fps':fps,'source_duration_s':n/fps,
                'sampled_frame_indices':positions.tolist()}
    finally:cap.release()


def prepare(archive,output,workers=4,limit=None):
    if (output/'fall-summary.json').exists():raise ValueError('Use a new output; do not mix dataset versions')
    output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    rng=random.Random(74291);subjects=list(range(1,30));rng.shuffle(subjects)
    split={s:('train' if i<20 else 'val' if i<24 else 'test') for i,s in enumerate(subjects)}
    excluded=[];records=[]
    with zipfile.ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if re.match(r'^VideoDataset/(ADL|Fall)/SBJ_\d+_LOC\d+/[^/]+/[^/]+\.mp4$',n))
        if limit:names=names[:limit]
        def process(name):
            data=z.read(name);digest=hashlib.sha256(data).hexdigest()
            parts=name.split('/');subject=int(re.search(r'SBJ_(\d+)',name)[1])
            metadata=json.loads(z.read('/'.join(parts[:-1])+'/metadata.json'))
            if int(metadata['subjectId'])!=subject:raise ValueError('Subject metadata/path mismatch')
            with tempfile.TemporaryDirectory(prefix='linq-fall-video-') as temp:
                video=Path(temp)/'clip.mp4';video.write_bytes(data)
                info=frames_from_video(video,output/'clips'/digest)
            return {'id':digest,'source':'figshare-28596332-v2','source_path':name,
                    'video_sha256':digest,'subject':f'figshare-sbj-{subject:02d}',
                    'location':metadata['locationId'],'action':metadata['actionId'],
                    'label':int(parts[1]=='Fall'),'split':split[subject],
                    'frames':f'clips/{digest}','label_scope':'whole action clip; no fall-onset timestamps',**info}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[(name,pool.submit(process,name)) for name in names]
            for i,(name,future) in enumerate(futures):
                try:records.append(future.result())
                except Exception as error:excluded.append({'path':name,'reason':type(error).__name__+': '+str(error)[:160]})
                if i%100==0:print('FIGSHARE_PREPARED',i,'/',len(names),flush=True)
    # Exact duplicate videos cannot appear on opposite sides of the split.
    unique={}
    for row in records:
        if row['video_sha256'] in unique:
            excluded.append({'path':row['source_path'],'reason':'exact duplicate RGB video'})
        else:unique[row['video_sha256']]=row
    records=list(unique.values())
    protocol=json.loads((ROOT/'docs/evaluation_protocol.json').read_text())
    urfd=ROOT/'data/datasets/urfd';labels={}
    for name in ['urfall-cam0-falls.csv','urfall-cam0-adls.csv']:
        with (urfd/name).open() as stream:
            for row in csv.reader(stream):
                if row:labels.setdefault(row[0],{})[int(row[1])-1]=int(row[2])
    for sequence,part in protocol['split'].items():
        source=urfd/(sequence+'-cam0-rgb.mp4')
        if not source.exists():raise FileNotFoundError(source)
        digest=hashlib.sha256(source.read_bytes()).hexdigest();cap=cv2.VideoCapture(str(source))
        fps=cap.get(5);duration=cap.get(7)/max(fps,1);cap.release()
        positive=sequence.startswith('fall-')
        transitions=[i for i,v in labels[sequence].items() if v==0]
        floor=[i for i,v in labels[sequence].items() if v==1]
        onset=(min(transitions or floor)/fps) if positive else None
        start=max(0,onset-.4) if positive else max(0,(duration-2)/2)
        identifier='urfd-'+digest
        info=frames_from_video(source,output/'clips'/identifier,start_s=start)
        records.append({'id':identifier,'source':'urfd-rgb-preview','source_path':source.name,
            'video_sha256':digest,'subject':None,'location':'urfd','label':int(positive),
            'split':'val' if part=='validation' else part,'frames':f'clips/{identifier}',
            'label_scope':'2s window using author posture annotations' if positive else 'ADL video window',
            'event_onset_source_s':onset,**info})
    groups={part:{r['subject'] for r in records if r['split']==part and r['subject']} for part in ['train','val','test']}
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    manifest=output/'fall-manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    summary={'unique_rgb_source_videos':len(records),'previous_urfd_source_videos':70,
        'expansion_factor_source_videos':len(records)/70,'clips':len(records),'frames_per_clip':16,
        'splits':{part:{'clips':sum(r['split']==part for r in records),
            'fall':sum(r['split']==part and r['label']==1 for r in records),
            'adl':sum(r['split']==part and r['label']==0 for r in records),
            'figshare_subjects':sorted(groups[part])} for part in groups},
        'excluded':excluded,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'figshare_subject_disjoint_verified':True,'urfd_split':'original frozen video split; subject IDs unavailable',
        'frame_onset_labels_available':'URFD only; Figshare evaluation is clip classification, not event timing',
        'sources':[{'url':ARTICLE,'license':'CC BY 4.0','author':'Ivan Ursul','version':2},
                   {'url':'https://fenix.ur.edu.pl/~mkepski/ds/uf.html','license':'CC BY-NC-SA 4.0'}]}
    (output/'fall-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (output/'SOURCE-NOTICES.txt').write_text('Figshare 28596332 v2, Ivan Ursul, CC BY 4.0.\nURFD, Bogdan Kwolek and Michal Kepski, CC BY-NC-SA 4.0.\nDerived clips retain source attribution in fall-manifest.jsonl. Mixed research dataset; not a commercial license grant.\n',encoding='utf-8')
    shutil.copyfile(ROOT/'data/models/fall-resnet18.pt',output/'initial-fall-resnet18.pt')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    a=p.parse_args();prepare(Path(a.archive),Path(a.output),a.workers,a.limit)
