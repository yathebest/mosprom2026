"""Download reproducible, source-split factory CCTV clips for annotation.

The source labels are clip-level behaviours, NOT PPE/phone/fall ground truth.
Downloaded clips have no automatically accepted safety annotations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests

REPO='Voxel51/Safe_and_Unsafe_Behaviours'
REVISION='7369ff1830b5f6c00f7d7383eeeaa90704bfc5b5'
BASE=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'
DEBUG_CLIPS={name+'.mp4' for name in ['0_te26','0_te9','1_te6','1_te2','2_te6','2_te12',
    '3_te4','3_te2','4_te17','4_te19','5_te8','5_te15','6_te4','6_te11','7_te1','7_te6']}


def select(samples,per_class,tag,exclude=()):
    result=[]
    for category in range(8):
        rows=sorted((x for x in samples if tag in x['tags'] and
                     Path(x['filepath']).name.startswith(str(category)+'_') and
                     Path(x['filepath']).name not in exclude),key=lambda x:x['filepath'])
        if len(rows)<per_class:raise ValueError('Not enough distinct source clips')
        indices=[round(i*(len(rows)-1)/max(1,per_class-1)) for i in range(per_class)]
        result.extend(rows[i] for i in indices)
    return result


def download(root,selection):
    split,row=selection;name=Path(row['filepath']).name;target=root/split/name
    target.parent.mkdir(parents=True,exist_ok=True)
    expected=row['metadata']['size_bytes'];url=BASE+row['filepath']
    if not target.exists() or target.stat().st_size!=expected:
        for attempt in range(3):
            try:
                with requests.get(url,stream=True,timeout=(20,90)) as response:
                    response.raise_for_status()
                    temporary=target.with_suffix('.mp4.part')
                    with temporary.open('wb') as output:
                        for block in response.iter_content(1024*1024):output.write(block)
                if temporary.stat().st_size!=expected:raise ValueError('Incomplete download')
                temporary.replace(target);break
            except (requests.RequestException,ValueError):
                if attempt==2:raise
                time.sleep(1+attempt)
    with target.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    print('FACTORY_CLIP',split,name,expected,flush=True)
    return {'file':str(target.relative_to(root)).replace('\\','/'),'source_url':url,
            'source_split':split,'sha256':sha,'source_behavior_label':row['ground_truth']['label'],
            'safety_annotations':'unreviewed',**row['metadata']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('data/datasets/factory-expansion'))
    parser.add_argument('--train-per-class',type=int,default=4)
    parser.add_argument('--test-per-class',type=int,default=2)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    response=requests.get(BASE+'samples.json',timeout=60);response.raise_for_status()
    samples=response.json()['samples']
    previous=Path('data/datasets/factory-cctv/manifest.json')
    excluded=DEBUG_CLIPS | ({x['filename'] for x in json.loads(previous.read_text(encoding='utf-8'))['files']} if previous.exists() else set())
    selected=[('train',x) for x in select(samples,args.train_per_class,'train')]
    selected += [('test',x) for x in select(samples,args.test_per_class,'test',excluded)]
    size=sum(row['metadata']['size_bytes'] for _,row in selected)
    print('FACTORY_SELECTION',len(selected),'bytes',size,'revision',REVISION,flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows=list(executor.map(lambda item:download(args.output,item),selected))
    manifest={'original_source':'https://data.mendeley.com/datasets/xjmtb22pff/1',
        'authors':['Oguzhan Onal','Emre Dandil'],'license':'CC-BY-4.0',
        'mirror':REPO,'mirror_revision':REVISION,
        'protocol':'Original train/test tags; evenly spaced filenames within each behaviour class. Previously inspected 16 test clips excluded from the new test selection.',
        'annotation_policy':'No source behaviour label is treated as a PPE, phone, smoking or fall label. Visual annotation and review required.',
        'files':rows}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FACTORY_DOWNLOAD_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
