"""Reproduce source-disjoint iSafety debug/validation clips without credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import requests

ROOT=Path(__file__).resolve().parents[1]
REV='8336f1c57076d598677effa1da9870b300790804'
BASE='https://huggingface.co/datasets/raiyaanabdullah/isafety-bench'


def group(name):return re.split(r'_(?:trim|clip)\d*_',name)[0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--fire',type=int,default=12)
    p.add_argument('--fall',type=int,default=12);p.add_argument('--normal',type=int,default=24)
    p.add_argument('--output',type=Path,default=ROOT/'data/datasets/isafety-expanded');a=p.parse_args()
    original=json.loads((ROOT/'data/datasets/isafety-inspection/manifest.json').read_text())
    used={group(r['video_name']) for r in original['files']};rows=[]
    tree=json.loads((ROOT/'data/cache/isafety/tree.json').read_text())
    checksums={r['path']:r.get('lfs',{}).get('oid') for r in tree if r['type']=='file'}
    for folder,action,count in [('hazard','fire incident',a.fire),('hazard','person falling down',a.fall),('normal',None,a.normal)]:
        annotations=json.loads((ROOT/f'data/cache/isafety/annotations_{folder}.json').read_text())
        selected=0
        for source in sorted(annotations,key=lambda r:r['video_name']):
            if action and action not in source['gt_actions']:continue
            name=source['video_name'];source_group=group(name)
            if source_group in used:continue
            relative=f'{folder}/{name}';target=a.output/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                response=requests.get(f'{BASE}/resolve/{REV}/{relative}',timeout=120);response.raise_for_status()
                temporary=target.with_suffix('.part');temporary.write_bytes(response.content);temporary.replace(target)
            digest=hashlib.sha256(target.read_bytes()).hexdigest()
            if checksums.get(relative) and checksums[relative]!=digest:raise ValueError(f'Source checksum mismatch: {relative}')
            rows.append({**source,'folder':folder,'source_group':source_group,'selection_action':action or 'normal author clip',
                'file':relative,'sha256':digest,'bytes':target.stat().st_size})
            used.add(source_group);selected+=1;print('ISAFETY_DOWNLOAD',relative,flush=True)
            if selected>=count:break
    report={'source':BASE,'revision':REV,'license':'CC-BY-NC-SA-4.0',
        'protocol':'One clip per source group, excluding initial inspection sources. Author clip actions only; not precise event onset or box ground truth. Not used for training.',
        'files':rows}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
