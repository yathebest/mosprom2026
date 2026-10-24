"""Create a bounded, reproducible experiment without changing the full corpus.

Selection uses source, labels and image identifiers, never model predictions.
Native train/val/test membership is retained. This is an exploratory subset.
"""
import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time

from train_ppe import QC_EXCLUDED_IMAGE_SHA256


def rank(path):
    return hashlib.sha256(('linq-fast-v1/42/'+path.name).encode()).digest()


def choose(paths, limit, metadata, task, crop=False):
    buckets=defaultdict(list)
    supported={'people':{'person'},'head_ppe':{'hardhat','no_hardhat'},
               'vest_ppe':{'vest','no_vest'}}[task]
    for path in paths:
        digest=path.stem[:64]
        if crop and digest in QC_EXCLUDED_IMAGE_SHA256:continue
        row=metadata[digest]
        labels=tuple(sorted({x[0] for x in row['labels']} & supported))
        areas=[x[3]*x[4] for x in row['labels'] if x[0] in supported]
        size='small' if areas and min(areas)<.01 else 'other'
        buckets[(row['source'],labels,size)].append(path)
    buckets=[deque(sorted(values,key=rank)) for _,values in sorted(buckets.items())]
    selected=[];per_image=defaultdict(int)
    while buckets and len(selected)<limit:
        remaining=[]
        for bucket in buckets:
            while bucket:
                path=bucket.popleft();digest=path.stem[:64]
                if not crop or per_image[digest]<4:
                    selected.append(path);per_image[digest]+=1;break
            if bucket:remaining.append(bucket)
            if len(selected)>=limit:break
        buckets=remaining
    return selected


def prepare(source, output, workers=8, tiny=False):
    if output.exists():raise ValueError('Use a new output directory; do not mix experiment subsets')
    output.mkdir(parents=True)
    metadata={r['sha256']:r for r in map(json.loads,(source/'manifest.jsonl').read_text().splitlines())}
    jobs=[];selected=[];counts={};groups={};started=time.monotonic()
    train_limits={'people':1200,'head_ppe':2400,'vest_ppe':2020}
    for task in train_limits:
        dest=output/task;dest.mkdir()
        shutil.copyfile(source/task/'data.yaml',dest/'data.yaml')
        for split in ['train','val','test']:
            limit=(train_limits[task] if split=='train' else 400) if not tiny else 24
            files=choose(list((source/task/'images'/split).glob('*')),limit,metadata,task)
            counts[f'{task}/{split}']=len(files)
            for image in files:
                digest=image.stem[:64];row=metadata[digest]
                if row['split']!=split:raise ValueError('Source split mismatch')
                group=(row['source'],row['group'])
                if group in groups and groups[group]!=split:raise ValueError('Source-group leakage')
                groups[group]=split
                label=source/task/'labels'/split/(image.stem+'.txt')
                jobs.extend([(image,dest/'images'/split/image.name),(label,dest/'labels'/split/label.name)])
                selected.append({'task':task,'split':split,'file':image.name,'sha256':digest})
    for task,limit in [('head_ppe',2000),('vest_ppe',1600)]:
        for split in ['train','val','test']:
            for folder in sorted((source/(task+'_classifier')/split).iterdir()):
                max_count=(limit if split=='train' else 256) if not tiny else 12
                files=choose(list(folder.glob('*.jpg')),max_count,metadata,task,crop=True)
                counts[f'{task}_classifier/{split}/{folder.name}']=len(files)
                for image in files:
                    digest=image.stem[:64];row=metadata[digest]
                    if row['split']!=split:raise ValueError('Crop split mismatch')
                    group=(row['source'],row['group'])
                    if group in groups and groups[group]!=split:raise ValueError('Crop source-group leakage')
                    groups[group]=split
                    target=output/(task+'_classifier')/split/folder.name/image.name
                    jobs.append((image,target))
                    selected.append({'task':task+'_classifier','split':split,'class':folder.name,
                                     'file':image.name,'sha256':digest})
    for _,target in jobs:target.parent.mkdir(parents=True,exist_ok=True)
    def copy(pair):
        src,dst=pair;shutil.copyfile(src,dst)
        if dst.stat().st_size!=src.stat().st_size:raise IOError(f'Copy size mismatch: {src.name}')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(copy,pair) for pair in jobs]
        for index,future in enumerate(as_completed(futures),1):
            future.result()
            if index%500==0:print(f'Local data cache {index}/{len(jobs)} files',flush=True)
    digests={x['sha256'] for x in selected}
    (output/'manifest.jsonl').write_text('\n'.join(json.dumps(metadata[x]) for x in sorted(digests))+'\n')
    (output/'selection.jsonl').write_text('\n'.join(map(json.dumps,selected))+'\n')
    report={'policy':'fast-v1-seed42-source-label-size-round-robin','counts':counts,
            'source_manifest_sha256':hashlib.sha256((source/'manifest.jsonl').read_bytes()).hexdigest(),
            'selection_sha256':hashlib.sha256((output/'selection.jsonl').read_bytes()).hexdigest(),
            'source_group_disjoint':True,'heldout_membership_preserved':True,
            'test_scope':'fixed bounded subset, not the entire held-out corpus',
            'max_classifier_crops_per_source_image_per_class':4,
            'elapsed_seconds':time.monotonic()-started,
            'warning':'Short fine-tuning experiment; no >0.95 quality guarantee.'}
    (output/'fast-data-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--tiny',action='store_true')
    a=p.parse_args();prepare(Path(a.source),Path(a.output),a.workers,a.tiny)
