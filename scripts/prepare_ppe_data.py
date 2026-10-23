"""Build separate, reproducible tasks without treating unlabelled PPE as absent.

Images remain in ignored source folders. Prepared datasets contain real files
and source/group manifests. Source test/validation images never enter training.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml

SITE_NAMES=['Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest','Person',
            'Safety Cone','Safety Vest','machinery','vehicle']
ALIASES={'helmet':'hardhat','no_helmet':'no_hardhat','bare_head':'no_hardhat',
         'safety_vest':'vest','no_safety_vest':'no_vest'}
TASKS={'people':['person'],'head_ppe':['hardhat','no_hardhat'],'vest_ppe':['vest','no_vest']}


def crop_quality(crop):
    """Conservative classifier visibility gate; this does not verify labels."""
    h,w=crop.shape[:2]
    if min(h,w)<24 or h*w<1024:return 'too_small'
    if not .2<=w/h<=4:return 'extreme_aspect'
    if cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()<15:
        return 'blurred'
    return None


def normalized(s):
    s=re.sub('[^a-z0-9]+','_',s.lower()).strip('_')
    return ALIASES.get(s,s)


def source_split(path):
    parts=set(path.parts)
    return 'test' if 'test' in parts else 'val' if parts & {'val','valid','validation'} else 'train'


def original_group(path):
    # Roboflow augmentation variants must share a split, even with different pixels.
    return re.sub(r'(?:_jpg|_png)?\.rf\.[a-f0-9]+$','',path.stem,flags=re.I)


def records(root):
    for source in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if not (source/'.extracted').exists():raise ValueError(f'Source extraction incomplete: {source}')
        configs=list(source.rglob('*.yaml'))+list(source.rglob('*.yml'))
        names=None
        for cfg in configs:
            parsed=yaml.safe_load(cfg.read_text(encoding='utf-8'))
            if isinstance(parsed,dict) and parsed.get('names'):
                n=parsed['names'];names={int(k):v for k,v in n.items()} if isinstance(n,dict) else dict(enumerate(n));break
        if names is None and source.name=='site_safety':names=dict(enumerate(SITE_NAMES))
        if names is None:
            class_files=list(source.rglob('classes.txt'))
            if class_files:names=dict(enumerate(line.strip() for line in class_files[0].read_text().splitlines() if line.strip()))
        annotations=list(source.rglob('Annotations'))
        xmls={p.stem:p for folder in annotations for p in folder.glob('*.xml')}
        native_splits={}
        for split in ['train','val','test']:
            for split_file in source.rglob(split+'.txt'):
                if split_file.parent.name=='Main':
                    for line in split_file.read_text().splitlines():
                        if line.strip():native_splits[line.split()[0]]=split
        for image in sorted(p for p in source.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}):
            labels=[]
            if xmls:
                xml=xmls.get(image.stem)
                if xml is None:continue
                node=ET.parse(xml).getroot();width=float(node.findtext('size/width'));height=float(node.findtext('size/height'))
                for obj in node.findall('object'):
                    label=obj.findtext('name');box=obj.find('bndbox')
                    # SHWD's "person" is an uncovered HEAD, not a person box.
                    label='hardhat' if label=='hat' else 'no_hardhat' if label=='person' else normalized(label)
                    a,b,c,d=[float(box.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
                    labels.append([label,(a+c)/2/width,(b+d)/2/height,(c-a)/width,(d-b)/height])
                complete={'head_ppe'}
            else:
                if names is None:continue
                relative=image.relative_to(source)
                parts=list(relative.parts)
                if 'images' not in parts:continue
                parts[parts.index('images')]='labels'
                label_path=(source/Path(*parts)).with_suffix('.txt')
                if not label_path.exists():continue
                for line in label_path.read_text().splitlines():
                    fields=line.split()
                    if len(fields)!=5:raise ValueError(f'Expected detection boxes, not segmentation: {label_path}')
                    label=normalized(names[int(fields[0])]);box=list(map(float,fields[1:]))
                    if not all(np.isfinite(box)) or not all(0<=v<=1 for v in box) or min(box[2:])<=0:
                        raise ValueError(f'Invalid bbox: {label_path}')
                    labels.append([label,*box])
                source_classes={normalized(n) for n in names.values()}
                complete={task for task,classes in TASKS.items() if set(classes)<=source_classes}
                # This aggregate mixes several annotation schemes. Use only
                # explicitly labelled positive crops until completeness and the
                # meaning of head/person have been reviewed for each subsource.
                if source.name=='hardhat_vest':complete=set()
            yield {'path':str(image.resolve()),'source':source.name,'split':native_splits.get(image.stem,source_split(image)),
                   'group':original_group(image),'labels':labels,'complete_tasks':sorted(complete)}


def prepare(root,output,summary_path,extra_roots=()):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Output is not empty; choose a new output to avoid stale files')
    manifest=[];hashes={};groups={};buckets=defaultdict(list);duplicates=0;unusable=0
    # Read held-out partitions first: duplicates and related variants in train
    # inherit the held-out split. This keeps source test images out of train.
    raw=[rec for source_root in [root,*extra_roots] for rec in records(source_root)]
    raw.sort(key=lambda r:({'test':0,'val':1,'train':2}[r['split']],-len(r['complete_tasks']),r['path']))
    print(f'Indexed {len(raw)} annotated source images',flush=True)
    for number,record in enumerate(raw):
        if number%1000==0:print(f'Deduplicating {number}/{len(raw)}',flush=True)
        if not record['complete_tasks'] and not any(v[0] in {'hardhat','vest'} for v in record['labels']):
            unusable+=1;continue
        path=Path(record['path']);data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f'Unreadable image: {path}')
        gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(32,32)).astype(np.float32)
        low=cv2.dct(gray)[:8,:8].ravel();bits=low>np.median(low[1:])
        phash=sum(int(v)<<i for i,v in enumerate(bits))
        candidates=set(i for k in range(8) for i in buckets[(k,(phash>>(k*8))&255)])
        near=next((i for i in candidates if (manifest[i]['phash']^phash).bit_count()<=2),None)
        duplicate=hashes.get(digest,near)
        if duplicate is not None:
            duplicates+=1
            previous=manifest[duplicate]
            groups[(record['source'],record['group'])]=previous['split']
            if len(record['complete_tasks'])>len(previous['complete_tasks']):
                record.update(split=previous['split'],sha256=digest,phash=phash,width=image.shape[1],height=image.shape[0])
                manifest[duplicate]=record
            hashes[digest]=duplicate
            continue
        # Group keys may over-group coincident source names; that is conservative
        # for leakage. Exact/perceptual matches are handled globally above.
        group=(record['source'],record['group'])
        split=groups.setdefault(group,record['split'])
        if split=='train':
            bucket=int(hashlib.sha256('|'.join(group).encode()).hexdigest()[:8],16)%100
            split='val' if bucket<10 else 'test' if bucket<20 else 'train'
        record.update(split=split,sha256=digest,phash=phash,width=image.shape[1],height=image.shape[0])
        hashes[digest]=len(manifest)
        for k in range(8):buckets[(k,(phash>>(k*8))&255)].append(len(manifest))
        manifest.append(record)
    rank={'test':0,'val':1,'train':2}
    for rec in manifest:
        group_split=groups.get((rec['source'],rec['group']),rec['split'])
        if rank[group_split]<rank[rec['split']]:rec['split']=group_split
    task_counts={};crop_counts=defaultdict(Counter)
    for task,classes in TASKS.items():
        print(f'Writing {task}',flush=True)
        counts=Counter();folder=output/task
        for rec in manifest:
            name=rec['sha256'];split=rec['split'];src=Path(rec['path'])
            selected=[v for v in rec['labels'] if v[0] in classes]
            if task in rec['complete_tasks']:
                im=folder/'images'/split/(name+src.suffix.lower());lab=folder/'labels'/split/(name+'.txt')
                im.parent.mkdir(parents=True,exist_ok=True);lab.parent.mkdir(parents=True,exist_ok=True)
                if not im.exists():shutil.copyfile(src,im)
                lab.write_text('\n'.join(str(classes.index(v[0]))+' '+' '.join(map(str,v[1:])) for v in selected)+'\n',encoding='utf-8')
                counts[split]+=1
            if task in {'head_ppe','vest_ppe'} and selected:
                frame=cv2.imread(str(src));h,w=frame.shape[:2]
                for i,(label,cx,cy,bw,bh) in enumerate(selected):
                    # Context is needed around a helmet box; no synthetic
                    # "unknown" examples or missing-annotation negatives.
                    x1,y1=max(0,int((cx-bw*.65)*w)),max(0,int((cy-bh*.65)*h))
                    x2,y2=min(w,int((cx+bw*.65)*w)),min(h,int((cy+bh*.65)*h))
                    if min(x2-x1,y2-y1)<16:continue
                    dest=output/(task+'_classifier')/split/label/(name+f'-{i}.jpg')
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    crop=frame[y1:y2,x1:x2]
                    if crop_quality(crop):continue
                    cv2.imwrite(str(dest),crop);crop_counts[task][split+':'+label]+=1
        folder.mkdir(parents=True,exist_ok=True)
        config={'path':str(folder.resolve()),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(classes))}
        (folder/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
        task_counts[task]=dict(counts)
    with (output/'manifest.jsonl').open('w',encoding='utf-8') as f:
        for rec in manifest:f.write(json.dumps(rec)+'\n')
    summary={'raw_images':len(raw),'unique_images':len(manifest),'duplicates_removed':duplicates,
             'excluded_incomplete_annotation_images':unusable,
             'sources':dict(Counter(r['source'] for r in manifest)),'splits':dict(Counter(r['split'] for r in manifest)),
             'tasks':task_counts,'classification_crops':{k:dict(v) for k,v in crop_counts.items()},
             'baseline_construction_train_images':1132,'upstream_claimed_train_images':68253,
             'upstream_4x_train_target':273012,'actual_unique_train_images':sum(r['split']=='train' for r in manifest),
             'upstream_4x_reached':sum(r['split']=='train' for r in manifest)>=273012,
             'limitations':['Source group metadata may be incomplete; camera-disjoint independence is not established.',
                            'No unobservable-class crops are fabricated; manual visibility labels remain required.',
                            'Head and torso classifier crops need human label/geometry review before a full training run.']}
    Path(summary_path).parent.mkdir(parents=True,exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    try:
        from scripts.finalize_ppe_data import finalize
    except ModuleNotFoundError:
        from finalize_ppe_data import finalize
    summary=finalize(output,summary_path,output.parent/'ppe-excluded-manifest.jsonl')
    print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/datasets/ppe-sources')
    p.add_argument('--extra-root',action='append',default=[])
    p.add_argument('--output',default='data/datasets/ppe-prepared');p.add_argument('--summary',default='docs/ppe-dataset-summary.json')
    a=p.parse_args();prepare(a.root,a.output,a.summary,a.extra_root)
