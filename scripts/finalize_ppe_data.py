"""Keep only source images that contribute detector examples or usable crops."""
import argparse
from collections import Counter
import json
from pathlib import Path


def finalize(data,summary_path,excluded):
    data=Path(data);summary_path=Path(summary_path);excluded=Path(excluded)
    summary=json.loads(summary_path.read_text());manifest=data/'manifest.jsonl'
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    used={p.stem for task in ['people','head_ppe','vest_ppe'] for p in (data/task/'images').rglob('*') if p.is_file()}
    used.update(p.stem[:64] for task in ['head_ppe_classifier','vest_ppe_classifier'] for p in (data/task).rglob('*.jpg'))
    kept=[r for r in rows if r['sha256'] in used];removed=[r for r in rows if r['sha256'] not in used]
    summary.setdefault('unique_images_before_crop_filter',summary['unique_images'])
    summary.setdefault('splits_before_crop_filter',summary['splits'])
    summary['unique_images']=len(kept)
    summary['excluded_images_without_usable_crops']=summary['unique_images_before_crop_filter']-len(kept)
    summary['sources']=dict(Counter(r['source'] for r in kept))
    summary['splits']=dict(Counter(r['split'] for r in kept))
    summary['actual_unique_train_images']=summary['splits'].get('train',0)
    summary['upstream_4x_reached']=summary['actual_unique_train_images']>=summary['upstream_4x_train_target']
    if removed:
        excluded.parent.mkdir(parents=True,exist_ok=True)
        with excluded.open('a',encoding='utf-8') as f:
            for row in removed:f.write(json.dumps(row)+'\n')
        temporary=manifest.with_suffix('.tmp')
        with temporary.open('w',encoding='utf-8') as f:
            for row in kept:f.write(json.dumps(row)+'\n')
        temporary.replace(manifest)
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    print('Contributing unique images:',len(kept),'train:',summary['actual_unique_train_images'],flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/datasets/ppe-expanded')
    p.add_argument('--summary',default='docs/ppe-dataset-summary.json')
    p.add_argument('--excluded',default='data/datasets/ppe-rejected-crops/unused-manifest.jsonl')
    a=p.parse_args();finalize(a.data,a.summary,a.excluded)
