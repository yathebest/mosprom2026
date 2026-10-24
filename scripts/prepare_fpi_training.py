"""Normalize FPI boxes and prevent exact/near-image leakage across splits.

Random author filenames do not expose original camera/video IDs. Perceptual
grouping reduces image leakage; it cannot prove camera-independent evaluation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import cv2
import numpy as np
from fpi_labels import read_labels


class HashTree:
    def __init__(self):self.root=None
    def add(self,value,index):
        node=[value,index,{}]
        if self.root is None:self.root=node;return
        current=self.root
        while True:
            distance=(value^current[0]).bit_count()
            if distance not in current[2]:current[2][distance]=node;return
            current=current[2][distance]
    def query(self,value,radius):
        pending=[self.root] if self.root else []
        while pending:
            current=pending.pop();distance=(value^current[0]).bit_count()
            if distance<=radius:yield current[1]
            pending.extend(v for d,v in current[2].items() if distance-radius<=d<=distance+radius)


def inspect(item):
    split,file=item;raw=file.read_bytes();gray=cv2.imdecode(np.frombuffer(raw,np.uint8),0)
    if gray is None:raise ValueError('Cannot decode '+str(file))
    dct=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].reshape(-1)[1:]
    phash=sum(int(v>np.median(dct))<<i for i,v in enumerate(dct))
    try:records=read_labels(file.parent.parent/'labels'/(file.stem+'.txt'));error=None
    except ValueError as exc:records=[];error=str(exc)
    return {'source_split':split,'file':str(file),'sha256':hashlib.sha256(raw).hexdigest(),
        'phash':phash,'labels':records,'label_error':error,'size':[int(gray.shape[1]),int(gray.shape[0])]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('data/datasets/fpi-det/reorganized_dataset'))
    p.add_argument('--output',type=Path,default=Path('data/datasets/fpi-prepared'));a=p.parse_args()
    cv2.setNumThreads(1);a.output.mkdir(parents=True,exist_ok=True)
    cache=a.output/'image-audit.json'
    if cache.exists():rows=json.loads(cache.read_text())
    else:
        files=[(split,file) for split in ['train','val','test'] for file in sorted((a.source/split/'images').glob('*.jpg'))]
        rows=[]
        with ThreadPoolExecutor(max_workers=4) as executor:
            for row in executor.map(inspect,files):
                rows.append(row)
                if len(rows)%2000==0:print('FPI_AUDIT',len(rows),flush=True)
        cache.write_text(json.dumps(rows))
    quarantined=[r for r in rows if r.get('label_error')]
    rows=[r for r in rows if not r.get('label_error')]
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    tree=HashTree()
    for i,row in enumerate(rows):
        for other in tree.query(row['phash'],4):parent[find(i)]=find(other)
        tree.add(row['phash'],i)
    groups={}
    for i,row in enumerate(rows):groups.setdefault(find(i),[]).append(row)
    rank={'train':0,'val':1,'test':2};kept={k:[] for k in rank};removed=[]
    for group in groups.values():
        priority=max(group,key=lambda r:rank[r['source_split']])['source_split'];seen=set()
        for row in group:
            if row['source_split']!=priority or row['sha256'] in seen:removed.append(row);continue
            seen.add(row['sha256']);kept[priority].append(row)
    for split,items in kept.items():
        for kind in ['images','labels']:(a.output/split/kind).mkdir(parents=True,exist_ok=True)
        for row in items:
            source=Path(row['file']);dest=a.output/split/'images'/source.name
            if not dest.exists():
                try:os.link(source,dest)
                except OSError:shutil.copyfile(source,dest)
            (a.output/split/'labels'/(source.stem+'.txt')).write_text('\n'.join(' '.join(map(str,r)) for r in row['labels'])+'\n')
        expected={Path(r['file']).name for r in items}
        if {p.name for p in (a.output/split/'images').glob('*.jpg')}!=expected:raise ValueError('Prepared directory contains stale images; use a fresh output folder')
    import yaml
    (a.output/'data.yaml').write_text(yaml.safe_dump({'path':str(a.output.resolve()),'train':'train/images','val':'val/images','test':'test/images','names':{0:'phone',1:'head'}}))
    report={'source':'https://github.com/KvCgRv/FPI-Det','license':'MIT (authors statement)',
        'protocol':'Parse concatenated 5-token records; remove exact box duplicates. Group pHash Hamming<=4 transitively. Held-out priority test > val > train. Never move held-out images into training.',
        'counts':{k:len(v) for k,v in kept.items()},'removed_images':len(removed),'groups':len(groups),
        'quarantined_invalid_annotations':[{'file':Path(r['file']).name,'split':r['source_split'],'error':r['label_error']} for r in quarantined],
        'class_counts':{k:{str(c):sum(r[0]==c for row in v for r in row['labels']) for c in [0,1]} for k,v in kept.items()},
        'warning':'Author filenames do not expose camera/video IDs; camera-independent generalization is not established.',
        'kept':{k:[{'file':Path(r['file']).name,'sha256':r['sha256']} for r in v] for k,v in kept.items()},
        'removed':[{'file':Path(r['file']).name,'split':r['source_split']} for r in removed]}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print('FPI_PREPARED',json.dumps({k:v for k,v in report.items() if k not in {'kept','removed'}}),flush=True)


if __name__=='__main__':main()
