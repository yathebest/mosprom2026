import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def load_script(name):
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_clothing_calibration_disables_a_class_with_unavoidable_false_compliance():
    trainer=load_script('train_signal_clothing')
    # High-confidence ordinary predictions include a compliant uniform. A high
    # confidence score alone must not enable absence-of-signal-clothing alerts.
    scores=np.array([[.001,.998,.001]]*11+[[.998,.001,.001]]*10+[[.001,.001,.998]]*10)
    truth=np.array([1]*10+[0]+[0]*10+[2]*10)
    thresholds,passed=trainer.calibrate(scores,truth)
    assert not passed[1] and thresholds[1]>1
    report=trainer.classification_report(scores,truth,thresholds)
    assert report['classes']['ordinary']['tp']==0
    assert report['classes']['ordinary']['fn']==10


def test_clothing_factory_samples_get_bounded_weight_without_duplicate_group_bias():
    trainer=load_script('train_signal_clothing')
    rows=[]
    for label in trainer.CLASSES:
        rows.extend([{'label':label,'source_group':'factory:one'}]*2)
        rows.extend({'label':label,'source_group':'site_safety/'+str(i)} for i in range(10))
    weights=trainer.balanced_weights(rows)
    assert abs(sum(weights)-1)<1e-8
    assert abs(sum(w for r,w in zip(rows,weights) if r['source_group'].startswith('factory:'))-.35)<1e-8


def test_notebooks_are_self_contained_and_do_not_stage_test_or_share_gpu(tmp_path):
    generator=load_script('build_targeted_notebooks')
    generator.build('both',tmp_path)
    data=json.loads((tmp_path/'LinQ-Two-GPU-Parallel-4h30.ipynb').read_text(encoding='utf-8'))
    sources=[''.join(c['source']) for c in data['cells'] if c['cell_type']=='code']
    for index,body in enumerate(sources):compile(body,f'cell{index}','exec')
    full='\n'.join(sources)
    assert 'TRAIN_MINUTES=250' in full and "('train','val')" in full
    assert "'CUDA_VISIBLE_DEVICES':str(index if TASK=='both' else 0)" in full
    assert "'train_targeted_person.py'" in full and "'train_signal_clothing.py'" in full
    assert 'GRACEFUL_STOP_REQUEST' in full and 'RESULT_ARCHIVE' in full


def test_person_far_views_transform_all_boxes_and_preserve_heldout(tmp_path,monkeypatch):
    builder=load_script('build_targeted_training_bundle');monkeypatch.setattr(builder,'ROOT',tmp_path)
    source=tmp_path/'data/datasets/ppe-expanded';metadata=[];originals={}
    for split,value in [('train',50),('val',100),('test',150)]:
        for folder in ['images','labels']:(source/'people'/folder/split).mkdir(parents=True,exist_ok=True)
        original=tmp_path/(split+'.jpg');Image.fromarray(np.full((64,64,3),value,np.uint8)).save(original)
        sha=builder.digest(original);image=source/'people/images'/split/(sha+'.jpg');image.write_bytes(original.read_bytes());originals[split]=image.read_bytes()
        (source/'people/labels'/split/(sha+'.txt')).write_text('0 .5 .5 .5 .5\n')
        metadata.append({'sha256':sha,'split':split,'complete_tasks':['people'],'source':'fixture','group':split})
    (source/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metadata))
    annotations=tmp_path/'review.json';annotations.write_text(json.dumps({'frames':[],'negative_patches':[]}))
    output=tmp_path/'output';output.mkdir();report=builder.prepare_people(output,annotations)
    assert report['images']=={'train':2,'val':1,'test':1} and report['derived_train_views']==1
    for split in ['val','test']:assert next((output/'people/images'/split).glob('*.jpg')).read_bytes()==originals[split]
    label=next((output/'people/labels/train').glob('far-*.txt')).read_text().split()
    assert int(label[0])==0 and .25<float(label[3])<.4 and .45<float(label[1])<.55
