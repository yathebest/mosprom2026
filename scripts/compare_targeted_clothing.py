"""CPU-only candidate audit; retain exported thresholds, never calibrate on test.

Source manifest and manual review are needed to restore legacy proposal crops.
Only legacy vest positives are comparable with the owner's broader uniform policy.
"""
import argparse, json, hashlib, time
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=ROOT/'data/cache/targeted-training/prepared-inputs/clothing/signal_clothing')
parser.add_argument('--candidate-dir',type=Path,default=ROOT/'data/cache/targeted-completed-20261025/clothing/clothing')
parser.add_argument('--legacy-model',type=Path,default=ROOT/'data/models/ppe-torso-classifier.onnx')
parser.add_argument('--output',type=Path,default=ROOT/'docs/targeted-clothing-comparison-20261025.json')
args=parser.parse_args()
DATA=args.data
CAND=args.candidate_dir
CLASSES=['hi_vis','ordinary','unobservable']
rows=list(map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines()))
source={r['sha256']:r for r in map(json.loads,(ROOT/'data/datasets/ppe-expanded/manifest.jsonl').read_text().splitlines())}
manual=json.loads((ROOT/'data/cache/targeted-training/clothing-review/review.json').read_text())['rows']
manual_by_sha={r['crop_sha256']:r for r in manual}
def session(path):
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

candidate=session(CAND/'signal-clothing-candidate.onnx');legacy=session(args.legacy_model)
def infer(model,image):
    im=np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR),np.float32)
    x=(im.transpose(2,0,1)[None]/255-np.array([.485,.456,.406],np.float32)[None,:,None,None])/np.array([.229,.224,.225],np.float32)[None,:,None,None]
    logits=model.run(None,{model.get_inputs()[0].name:np.ascontiguousarray(x)})[0][0]
    z=np.exp(logits-logits.max());return z/z.sum()

def old_crop(row):
    native=source.get(row['source_image_sha256'])
    if not native:return None,'no annotated legacy proposal'
    manual_row=manual_by_sha.get(row['image_sha256'])
    if manual_row:
        box=manual_row.get('source_no_vest_box')
        if box is None:return None,'no annotated legacy proposal'
        a,b,c,d=box;label='no_vest'
    else:
        try:idx=int(Path(row['file']).stem.rsplit('-',1)[1])
        except ValueError:return None,'no legacy label index'
        persons=[v[1:] for v in native['labels'] if v[0]=='person']
        if idx>=len(persons):return None,'invalid legacy person index'
        px,py,pw,ph=persons[idx];matching=[v for v in native['labels'] if v[0]=='vest' and px-pw/2<=v[1]<=px+pw/2 and py-ph/2<=v[2]<=py+ph/2]
        if not matching:return None,'no matched explicit vest'
        label,cx,cy,bw,bh=matching[0];w,h=native['width'],native['height']
        a,b,c,d=(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h
    w,h=native['width'],native['height']
    x1,y1=max(0,int(a-(c-a)*.15)),max(0,int(b-(d-b)*.15))
    x2,y2=min(w,int(c+(c-a)*.15)),min(h,int(d+(d-b)*.15))
    if min(x2-x1,y2-y1)<20:return None,'legacy minimum crop pixels'
    return Image.open(native['path']).convert('RGB').crop((x1,y1,x2,y2)),label

thresholds=json.loads((CAND/'candidate.json').read_text())['thresholds']
predictions=[];start=time.monotonic()
for i,row in enumerate(r for r in rows if r['split'] in {'val','test'}):
    image=Image.open(DATA/row['file']);native_size=image.size
    scores=infer(candidate,image);pred=int(scores.argmax());accepted=bool(scores[pred]>=thresholds[pred])
    item={**row,'domain':'factory' if row['source_group'].startswith('factory:') else 'upstream',
        'native_crop_size':list(native_size),'runtime_minimum_geometry_eligible':min(native_size)>=20,
        'scores':scores.tolist(),'raw_prediction':CLASSES[pred],
        'calibrated_prediction':CLASSES[pred] if accepted else 'abstain'}
    if row['split']=='test':
        crop,reason=old_crop(row);item['legacy_proposal_status']=reason
        if crop is not None:
            old=infer(legacy,crop);item['legacy_scores_no_vest_vest']=old.tolist()
            item['legacy_accepted_vest']=bool(old.argmax()==1 and old[1]>=.8)
    predictions.append(item)
    if i%40==0:print('CPU_AUDIT',i,round(time.monotonic()-start,1),flush=True)

def metrics(items,key):
    matrix=np.zeros((3,4),np.int64)
    for r in items:matrix[CLASSES.index(r['label']),CLASSES.index(r[key]) if r[key] in CLASSES else 3]+=1
    out={}
    for i,name in enumerate(CLASSES):
        tp=int(matrix[i,i]);fp=int(matrix[:,i].sum()-tp);fn=int(matrix[i].sum()-tp)
        precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
        out[name]={'support':int(matrix[i].sum()),'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':2*precision*recall/max(1e-12,precision+recall)}
    return {'n':len(items),'confusion_truth_by_prediction_or_abstain':matrix.tolist(),'classes':out,
        'macro_f1':sum(x['f1'] for x in out.values())/3,
        'coverage':float(matrix[:,:3].sum()/max(1,len(items)))}

summary={}
for split in ['val','test']:
    summary[split]={}
    for domain in ['all','factory','upstream']:
        subset=[r for r in predictions if r['split']==split and (domain=='all' or r['domain']==domain)]
        summary[split][domain]={key:metrics(subset,key) for key in ['raw_prediction','calibrated_prediction']}
testold=[r for r in predictions if r['split']=='test' and 'legacy_accepted_vest' in r]
oldtp=sum(r['legacy_accepted_vest'] and r['label']=='hi_vis' for r in testold)
oldfp=sum(r['legacy_accepted_vest'] and r['label']!='hi_vis' for r in testold)
legacy_summary={'n_evaluable_with_valid_annotated_proposal':len(testold),'accepted_vest_tp':oldtp,'accepted_vest_fp':oldfp,
 'precision':oldtp/max(1,oldtp+oldfp),'hi_vis_recall':oldtp/max(1,sum(r['label']=='hi_vis' for r in testold)),
 'threshold':.8,'policy':'Only accepted vest positives scored; no_vest predictions are not no-hi-vis evidence. Annotated proposal crops1.3 are oracle geometry, not whole detector pipeline; candidate uses person context1.05.'}
report={'task':'signal_clothing_completed_candidate_audit','providers':candidate.get_providers(),'threads':2,
 'candidate_sha256':hashlib.sha256((CAND/'signal-clothing-candidate.onnx').read_bytes()).hexdigest(),
 'legacy_sha256':hashlib.sha256(args.legacy_model.read_bytes()).hexdigest(),
 'preprocessing':'RGB PIL BILINEAR224; ImageNet normalization; metadata-matched',
 'fixed_validation_thresholds':thresholds,'test_used_for_calibration':False,
 'candidate_metadata':candidate.get_modelmeta().custom_metadata_map,
 'summary':summary,'legacy_accepted_vest_positive_only':legacy_summary,'elapsed_cpu_s':time.monotonic()-start,
 'limitations':['Prepared development splits, assistant labels; not expert certification or camera-disjoint blind factory benchmark.',
 'No original factory TEST is in training or this prepared held-out test; factory support exists in validation only.',
 'Crop-level static classifier comparison, not temporal full-pipeline event or tracker evaluation.',
 'Thresholds were kept exactly as exported; test cannot be used to lower thresholds.'],
 'predictions':predictions,
 'deployment_decision':'reject' if any(r['calibrated_prediction']=='hi_vis' and r['label']!='hi_vis' for r in predictions if r['split']=='test') else 'manual_review_required',
 'deployment_scope':'Automatic person compliance/clear in current production state machine',
 'deployment_reason':'ordinary/unobservable gates disabled; accepted hi_vis false positives on ordinary and unobservable reference examples; no independent factory held-out test. Preserve existing classifier and broad-policy negative abstention.'}
target=args.output;target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'summary':summary,'legacy':legacy_summary,'path':str(target)},ensure_ascii=False),flush=True)
