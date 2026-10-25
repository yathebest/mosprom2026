"""Package actual labelled data, train-only variants and two bounded notebooks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

cv2.setNumThreads(1)

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def rank(text):return hashlib.sha256(('targeted-20261024/42/'+text).encode()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare_people(output,annotations):
    source=ROOT/'data/datasets/ppe-expanded';metadata={r['sha256']:r for r in map(json.loads,(source/'manifest.jsonl').read_text().splitlines())}
    dest=output/'people';rows=[]
    for split in ['train','val','test']:
        for kind in ['images','labels']:(dest/kind/split).mkdir(parents=True,exist_ok=True)
        images=sorted(p for p in (source/'people/images'/split).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
        for image in images:
            m=metadata[image.stem];label=source/'people/labels'/split/(image.stem+'.txt')
            if m['split']!=split or 'people' not in m['complete_tasks']:raise ValueError('Person annotation completeness/split mismatch')
            if digest(image)!=image.stem:raise ValueError('Source SHA mismatch')
            shutil.copyfile(image,dest/'images'/split/image.name);shutil.copyfile(label,dest/'labels'/split/label.name)
            rows.append({'file':f'images/{split}/{image.name}','split':split,'source':m['source'],'source_group':m['source']+'/'+m['group'],
                'source_image_sha256':image.stem,'image_sha256':image.stem,'derived':False,'labels':'complete upstream person boxes'})
        print('PERSON_NATIVE',split,len(images),flush=True)
    # Deterministic train-only far-view augmentation; these are derived views,
    # not new cameras, people, independent observations or a claimed 4x corpus.
    train=sorted([r for r in rows if r['split']=='train' and Path(r['file']).suffix.lower()=='.jpg'],key=lambda r:rank(r['file']))
    added=0
    for row in train:
        if added>=1500:break
        frame=cv2.imread(str(dest/row['file']));h,w=frame.shape[:2]
        labels=[list(map(float,line.split())) for line in (dest/'labels/train'/Path(row['file']).with_suffix('.txt').name).read_text().splitlines() if line.strip()]
        scale=.55 if int(rank(row['file'])[:2],16)%2 else .7
        if labels and any(min(box[3]*w,box[4]*h)*scale<4 for box in labels):continue
        nw,nh=round(w*scale),round(h*scale);left=(w-nw)//2;top=(h-nh)//2
        canvas=np.full_like(frame,114);canvas[top:top+nh,left:left+nw]=cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_AREA)
        name='far-'+Path(row['file']).name;target=dest/'images/train'/name
        cv2.imwrite(str(target),canvas,[cv2.IMWRITE_JPEG_QUALITY,93])
        transformed=[[int(c),(x*nw+left)/w,(y*nh+top)/h,bw*nw/w,bh*nh/h] for c,x,y,bw,bh in labels]
        (dest/'labels/train'/Path(name).with_suffix('.txt')).write_text(''.join(' '.join(map(str,b))+'\n' for b in transformed))
        rows.append({**row,'file':'images/train/'+name,'derived':True,'image_sha256':digest(target),'transform':{'type':'far_view_letterbox','scale':scale}});added+=1
    review=json.loads(Path(annotations).read_text(encoding='utf-8'))
    for number,item in enumerate(review['frames']):
        if not item['source_video'].startswith('train/'):raise ValueError('No source test frame may enter training')
        if not item.get('complete_visible_person_annotation'):continue
        frame=ROOT/'data/datasets/factory-review'/item['frame_file'];image=cv2.imread(str(frame));h,w=image.shape[:2]
        if [w,h]!=[item['width'],item['height']]:raise ValueError('Reviewed coordinates use a different image size')
        name=f'factory-reviewed-{number:03}.jpg';target=dest/'images/train'/name;shutil.copyfile(frame,target)
        boxes=[]
        for a,b,c,d in item['boxes']:
            if not 0<=a<c<=w or not 0<=b<d<=h:raise ValueError('Invalid reviewed box')
            boxes.append([0,(a+c)/2/w,(b+d)/2/h,(c-a)/w,(d-b)/h])
        (dest/'labels/train'/Path(name).with_suffix('.txt')).write_text(''.join(' '.join(map(str,b))+'\n' for b in boxes))
        rows.append({'file':'images/train/'+name,'split':'train','source':'factory_assistant_review','source_group':item['source_video'],
            'source_image_sha256':digest(frame),'image_sha256':digest(target),'derived':False,'labels':'Assistant visually annotated visible people; not expert ground truth'})
    for number,item in enumerate(review['negative_patches']):
        if not item['source_video'].startswith('train/') or not item.get('reviewed_no_person'):raise ValueError('Negative patch not reviewed/source-train')
        frame=cv2.imread(str(ROOT/'data/datasets/factory-review'/item['frame_file']));a,b,c,d=map(int,item['crop_box']);patch=frame[b:d,a:c]
        if min(patch.shape[:2])<24:raise ValueError('Negative patch too small')
        for variant in range(4):
            # Each variation remains the same explicitly reviewed empty region.
            view=patch if variant==0 else cv2.flip(patch,1) if variant==1 else cv2.convertScaleAbs(patch,alpha=.8 if variant==2 else 1.1,beta=5)
            name=f'factory-negative-{number:03}-{variant}.jpg';target=dest/'images/train'/name
            cv2.imwrite(str(target),view,[cv2.IMWRITE_JPEG_QUALITY,93]);(dest/'labels/train'/Path(name).with_suffix('.txt')).write_text('')
            rows.append({'file':'images/train/'+name,'split':'train','source':'factory_reviewed_negative','source_group':item['source_video'],
                'source_image_sha256':digest(ROOT/'data/datasets/factory-review'/item['frame_file']),'image_sha256':digest(target),
                'derived':True,'transform':{'type':'reviewed_empty_patch','crop_box':item['crop_box'],'variant':variant},'labels':'Reviewed no-person machine/fan/floor patch'})
    (dest/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    shutil.copyfile(annotations,dest/'factory-annotation-review.json')
    return {'images':dict(Counter(r['split'] for r in rows)),'sources':dict(Counter(r['source'] for r in rows)),
        'derived_train_views':sum(r['derived'] for r in rows),'existing_prepared_source_splits_preserved':True,
        'warning':'Added factory examples are assistant-reviewed training material, not a factory benchmark. Far-view/negative variants do not count as independent source images.'}


def prepare_clothing(output,review_path):
    review=json.loads(Path(review_path).read_text(encoding='utf-8'));dest=output/'signal_clothing';rows=[]
    reviewed_images={r['source_image_sha256'] for r in review['rows']}
    excluded_geometry=0
    for split in ['train','val','test']:(dest/'images'/split).mkdir(parents=True,exist_ok=True)
    # Reviewed decisions have priority; never derive ordinary from no_vest.
    for i,row in enumerate(review['rows']):
        if row['decision']=='exclude':continue
        if row.get('geometry_source')=='upstream NoVest region; no matched person annotation':
            excluded_geometry+=1;continue
        if row['decision'] not in {'hi_vis','ordinary','unobservable'}:raise ValueError('Unknown clothing review decision')
        split=row['split'];source=Path(row['crop_file'])
        if not source.is_absolute():source=(ROOT/source) if (ROOT/source).is_file() else Path(review_path).parent/source
        name=f'reviewed-{i:05}.jpg';target=dest/'images'/split/name
        image=cv2.imread(str(source))
        if image is None:raise ValueError('Missing reviewed clothing crop')
        if row.get('crop_sha256') and digest(source)!=row['crop_sha256']:raise ValueError('Reviewed crop changed')
        shutil.copyfile(source,target)
        rows.append({'file':f'images/{split}/{name}','split':split,'label':row['decision'],
            'source_group':row['source_group'],'source_image_sha256':row['source_image_sha256'],
            'source':row.get('source','assistant_review'),'image_sha256':digest(target),
            'reviewer':row['reviewer'],'annotation_scope':'visible clothing appearance, not physical reflectivity certification',
            'crop_policy':row.get('crop_policy','person_body_context_1.05')})
    # Explicit worn-vest positives are compatible with the broader policy.
    # Extract full PERSON crops from fully annotated images; do not mix tight
    # vest-object crops with the deployment whole-person crop geometry.
    original=ROOT/'data/datasets/ppe-expanded';metadata=[json.loads(x) for x in (original/'manifest.jsonl').read_text().splitlines()]
    groups={r['source_group']:r['split'] for r in rows};added=Counter()
    for m in sorted(metadata,key=lambda r:rank(r['sha256'])):
        if m['source']!='site_safety' or 'people' not in m['complete_tasks']:continue
        # Never override a reviewed ambiguous/ordinary decision with another
        # upstream box on the same image. This also prevents near-identical
        # person crops with contradictory labels from entering the corpus.
        if m['sha256'] in reviewed_images:continue
        split=m['split']
        if added[split]>={'train':1800,'val':250,'test':250}[split]:continue
        persons=[r[1:] for r in m['labels'] if r[0]=='person'];vests=[r[1:] for r in m['labels'] if r[0]=='vest']
        if not vests:continue
        group=m['source']+'/'+m['group']
        if group in groups and groups[group]!=split:raise ValueError('Clothing group leakage')
        source=Path(m['path']);frame=cv2.imread(str(source))
        if frame is None:raise ValueError('Missing original person-clothing image')
        h,w=frame.shape[:2]
        for person_index,(x,y,bw,bh) in enumerate(persons):
            a,b,c,d=(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h
            matching=[v for v in vests if a<=v[0]*w<=c and b<=v[1]*h<=d]
            if not matching or min(c-a,d-b)<24:continue
            # Vests must have a unique owner. Overlapping people are not labels.
            v=matching[0]
            owners=sum(px-pw/2<=v[0]<=px+pw/2 and py-ph/2<=v[1]<=py+ph/2 for px,py,pw,ph in persons)
            if owners!=1:continue
            pad_x=(c-a)*.025;pad_y=(d-b)*.025
            x1,y1,x2,y2=max(0,int(a-pad_x)),max(0,int(b-pad_y)),min(w,int(np.ceil(c+pad_x))),min(h,int(np.ceil(d+pad_y)))
            name=f'vest-positive-{m["sha256"]}-{person_index}.jpg';target=dest/'images'/split/name
            cv2.imwrite(str(target),frame[y1:y2,x1:x2],[cv2.IMWRITE_JPEG_QUALITY,95])
            rows.append({'file':f'images/{split}/{name}','split':split,'label':'hi_vis','source':'site_safety_explicit_vest',
                'source_group':group,'source_image_sha256':m['sha256'],'image_sha256':digest(target),
                'reviewer':'upstream explicit vest; unique geometric person association',
                'crop_policy':'person_body_context_1.05','evidence_type':'explicit_vest'})
            groups[group]=split;added[split]+=1
            if added[split]>={'train':1800,'val':250,'test':250}[split]:break
    # Independently reviewed negatives/unknowns are required; do not fill gaps
    # by relabelling absent detector boxes or unreviewed upstream no_vest crops.
    groups={};source_hashes={}
    for row in rows:
        for mapping,key in [(groups,row['source_group']),(source_hashes,row['source_image_sha256'])]:
            if key in mapping and mapping[key]!=row['split']:raise ValueError('Clothing group/source-image leakage')
            mapping[key]=row['split']
    (dest/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    shutil.copyfile(review_path,dest/'manual-review.json')
    counts=Counter((r['split'],r['label']) for r in rows)
    for split in ['train','val']:
        for label in ['hi_vis','ordinary','unobservable']:
            if counts[split,label]<5:raise ValueError(f'Insufficient reviewed {split}/{label}: {counts[split,label]}')
    return {'classes':['hi_vis','ordinary','unobservable'],'counts':{f'{s}/{l}':n for (s,l),n in sorted(counts.items())},
        'policy':'Vest OR visible reflective work uniform compliant; uncertain visibility is unobservable.',
        'source_group_disjoint':True,'negative_labels':'Manual review only; no automatic no_vest-to-ordinary conversion',
        'excluded_unmatched_person_geometry':excluded_geometry,
        'warning':'Assistant reviewed clothing appearance; no physical garment certification, no independent factory event metric. Original source test is not used for model selection.'}


def create_input(task,root,summary,output):
    folder=root/task
    marker={'version':1,'task':task,'data_directory':'people' if task=='person' else 'signal_clothing',
        'weights_file':'weights/yolo12m.pt' if task=='person' else 'weights/torso-backbone.pt','summary':summary,
        'budget':{'train_minutes':250,'outer_graceful_stop_minutes':270,'outer_hard_stop_minutes':285},
        'test_policy':'Test remains sealed; training and checkpoint/threshold selection use train and validation only.'}
    marker['initial_weights_sha256']=digest(folder/marker['weights_file'])
    write_json(folder/'LINQ_TARGETED_INPUT.json',marker)
    archive=output/f'linq-{task}-training-input.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(folder))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Archive CRC failed')
    return {'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive),'summary':summary}


def main():
    p=argparse.ArgumentParser();p.add_argument('--person-review',type=Path,required=True)
    p.add_argument('--clothing-review',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.work.exists() and any(a.work.iterdir()):raise ValueError('Use a new work directory to avoid stale data')
    a.work.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
    person_root=a.work/'person';clothing_root=a.work/'clothing';person_root.mkdir();clothing_root.mkdir()
    person_summary=prepare_people(person_root,a.person_review)
    clothing_summary=prepare_clothing(clothing_root,a.clothing_review)
    for root in [person_root,clothing_root]:
        (root/'weights').mkdir()
    shutil.copyfile(ROOT/'data/models/yolo12m.pt',person_root/'weights/yolo12m.pt')
    with zipfile.ZipFile('C:/Users/User/Downloads/linq-night-trained-models.zip') as z:
        with z.open('vest_ppe-classifier.pt') as src,(clothing_root/'weights/torso-backbone.pt').open('wb') as dst:shutil.copyfileobj(src,dst)
    sources={'site_safety':{'url':'https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow','license':'CC BY 4.0, source declaration'},
        'construction':{'url':'https://docs.ultralytics.com/datasets/detect/construction-ppe/','license':'AGPL-3.0; source notices retained'},
        'factory':{'url':'https://data.mendeley.com/datasets/xjmtb22pff/1','authors':['Oguzhan Onal','Emre Dandil'],'license':'CC BY 4.0'},
        'software':{'model':'YOLO12m / Ultralytics','license':'AGPL-3.0 or appropriate upstream commercial license'},
        'scope':'Prepared private/research training bundle; this is not a new unified license or a public redistribution grant.'}
    for root in [person_root,clothing_root]:write_json(root/'SOURCES.json',sources)
    reports=[create_input('person',a.work,person_summary,a.output),create_input('clothing',a.work,clothing_summary,a.output)]
    write_json(a.output/'input-archives.json',reports);print(json.dumps(reports,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
