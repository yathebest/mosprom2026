"""Installer transaction/contract tests; no real weights or NN runtimes load."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT=Path(__file__).resolve().parents[1]


@pytest.fixture
def installer(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('targeted_installer_test',PROJECT/'scripts/install_targeted_candidates.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    calls=[]
    def load(path,load_external_data=False):
        calls.append(('onnx',Path(path).name))
        return SimpleNamespace(metadata_props=[SimpleNamespace(key=k,value=v)
            for k,v in json.loads(Path(path).read_text()).items()])
    def yolo(path,task):
        calls.append(('checkpoint',Path(path).name))
        assert task=='detect'
        return SimpleNamespace(names=json.loads(Path(path).read_text())['names'])
    monkeypatch.setitem(sys.modules,'onnx',SimpleNamespace(load=load,checker=SimpleNamespace(check_model=lambda graph:calls.append(('check',len(graph.metadata_props))))))
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=yolo))
    module.test_calls=calls
    return module


def clothing_meta(**changes):
    values={'class_names':json.dumps(['hi_vis','ordinary','unobservable']),
            'policy':'vest_or_reflective_uniform','crop_policy':'person_body_context_1.05',
            'class_thresholds':json.dumps([.9,1.01,.9])}
    values.update(changes);return values


def candidate(installer,tmp_path,task,meta=None,decision=None,checkpoint_names=None):
    folder=tmp_path/(task+'-candidate');folder.mkdir(exist_ok=True)
    file=folder/('person-yolo12m-candidate.onnx' if task=='person' else 'signal-clothing-candidate.onnx')
    file.write_text(json.dumps(meta if meta is not None else {'names':repr({0:'person'})} if task=='person' else clothing_meta()))
    (folder/'candidate.json').write_text(json.dumps({'onnx_sha256':installer.digest(file)}))
    if task=='person':
        weights=folder/'person/weights';weights.mkdir(parents=True,exist_ok=True)
        (weights/'best.pt').write_text(json.dumps({'names':checkpoint_names or {0:'person'}}))
    report=tmp_path/(task+'-comparison.json')
    report.write_text(json.dumps({'deployment_decision':decision or ('accept' if task=='person' else 'accept_positive_only')}))
    return folder,file,report


def test_wrong_hash_rejected_before_onnx_parser(installer,tmp_path):
    _,file,_=candidate(installer,tmp_path,'person')
    with pytest.raises(ValueError,match='checksum'):
        installer.verified_onnx(file,'0'*64,'person')
    assert installer.test_calls==[]


@pytest.mark.parametrize('names',[{0:'head_proposal'},{'0':'head_proposal'},['head_proposal']])
def test_onnx_class_zero_must_really_be_person(installer,tmp_path,names):
    _,file,_=candidate(installer,tmp_path,'person',{'names':repr(names)})
    with pytest.raises(ValueError,match='Class zero must be person'):
        installer.verified_onnx(file,installer.digest(file),'person')


def test_checkpoint_class_zero_is_checked_independently(installer,tmp_path):
    folder,_,report=candidate(installer,tmp_path,'person',checkpoint_names={0:'head_proposal'})
    models=tmp_path/'models'
    with pytest.raises(ValueError,match='class 0 = person'):
        installer.install(person=folder,reports={'person':report},models=models)
    assert not models.exists()


@pytest.mark.parametrize('change',[
    {'policy':'vest_only'}, {'crop_policy':'proposal_context_1.3'},
    {'class_names':json.dumps(['ordinary','hi_vis','unobservable'])},
    {'class_thresholds':json.dumps([.9,1.01])},
    {'class_thresholds':json.dumps({'0':.9,'1':1.01,'2':.9})},
])
def test_clothing_policy_class_order_and_crop_contract_are_fixed(installer,tmp_path,change):
    _,file,_=candidate(installer,tmp_path,'clothing',clothing_meta(**change))
    with pytest.raises(ValueError,match='contract differs'):
        installer.verified_onnx(file,installer.digest(file),'clothing')


@pytest.mark.parametrize('thresholds',[
    [.9,float('nan'),.9], [.9,float('inf'),.9], [.7,1.01,.9],
    [.9,1.02,.9], [.9,1.01,1.02], [True,1.01,.9], [.9,'1.01',.9],
])
def test_all_thresholds_are_finite_runtime_supported_numbers(installer,tmp_path,thresholds):
    _,file,_=candidate(installer,tmp_path,'clothing',clothing_meta(class_thresholds=json.dumps(thresholds)))
    with pytest.raises(ValueError,match='finite numbers'):
        installer.verified_onnx(file,installer.digest(file),'clothing')


@pytest.mark.parametrize('ordinary',[.9,1.0])
def test_ordinary_class_cannot_be_enabled_by_installer(installer,tmp_path,ordinary):
    _,file,_=candidate(installer,tmp_path,'clothing',clothing_meta(class_thresholds=json.dumps([.9,ordinary,.9])))
    with pytest.raises(ValueError,match='positive-only'):
        installer.verified_onnx(file,installer.digest(file),'clothing')


@pytest.mark.parametrize('task,decision',[('person','reject'),('clothing','accept'),('clothing','reject')])
def test_quality_gate_must_explicitly_approve_matching_deployment_mode(installer,tmp_path,task,decision):
    folder,_,report=candidate(installer,tmp_path,task,decision=decision)
    models=tmp_path/'models';models.mkdir();prior=models/'existing.onnx';prior.write_bytes(b'unchanged')
    with pytest.raises(ValueError,match='did not approve'):
        installer.install(**{task:folder},reports={task:report},models=models)
    assert installer.test_calls==[] and prior.read_bytes()==b'unchanged'
    assert list(models.iterdir())==[prior]


def test_positive_only_install_preserves_disabled_ordinary_threshold_exactly(installer,tmp_path):
    folder,file,report=candidate(installer,tmp_path,'clothing')
    original=file.read_bytes();models=tmp_path/'models'
    result=installer.install(clothing=folder,reports={'clothing':report},models=models)
    installed=models/'signal-clothing.onnx'
    assert installed.read_bytes()==original
    meta=installer.verified_onnx(installed,installer.digest(installed),'clothing')
    assert json.loads(meta['class_thresholds'])==[.9,1.01,.9]
    assert result['files']['signal-clothing.onnx']['sha256']==installer.digest(file)
    assert 'ordinary class disabled' in result['decisions']['clothing']['mode']
    assert not result['linq_compiled'] and not result['event_accuracy_measured']


@pytest.mark.parametrize('failure',['copy_exception','bad_staging_digest'])
def test_install_rollback_restores_old_removes_new_and_cleans_only_its_staging(installer,tmp_path,monkeypatch,failure):
    person,_,person_report=candidate(installer,tmp_path,'person')
    clothing,clothing_file,clothing_report=candidate(installer,tmp_path,'clothing')
    models=tmp_path/'models';models.mkdir()
    old={'person-yolo12m.pt':b'old-person','signal-clothing.onnx':b'old-clothing',
         'targeted-installation.json':b'{"old_report":true}', 'unrelated.installing':b'do-not-delete'}
    for name,data in old.items():(models/name).write_bytes(data)
    copy=installer.shutil.copyfile
    def failing_copy(source,target,*args,**kwargs):
        if Path(source)==clothing_file:
            Path(target).write_bytes(b'partial or corrupted staging')
            if failure=='copy_exception':raise OSError('simulated interrupted copy')
            return str(target)
        return copy(source,target,*args,**kwargs)
    monkeypatch.setattr(installer.shutil,'copyfile',failing_copy)
    with pytest.raises((OSError,ValueError)):
        installer.install(person,clothing,{'person':person_report,'clothing':clothing_report},models)
    for name,data in old.items():assert (models/name).read_bytes()==data
    assert not (models/'person-yolo12m.onnx').exists()
    assert sorted(p.name for p in models.glob('*.installing'))==['unrelated.installing']


def test_disabled_ordinary_stays_unknown_through_runtime_and_id7_ui():
    from app.core import Person,RunConfig
    from app.scenario_models import CropClassifier
    from app.scenarios import ScenarioRules
    classifier=CropClassifier.__new__(CropClassifier)
    classifier.clothing=True;classifier.labels=['hi_vis','ordinary','unobservable'];classifier.class_thresholds=[.9,1.01,.9]
    classifier.mean=np.zeros((1,3,1,1),np.float32);classifier.std=np.ones((1,3,1,1),np.float32)
    classifier.input_name='images';classifier.size=224;classifier.provider='CPUExecutionProvider'
    classifier.region='torso';classifier.name='signal-clothing'
    classifier.session=SimpleNamespace(run=lambda *args:[np.array([[0.,30.,0.]],np.float32)])
    person=Person((20,10,80,190),.95,track_id=7);frame=np.zeros((220,100,3),np.uint8)
    rules=ScenarioRules(RunConfig(vest_detection=True));events=[]
    for now in [0,.5,1,1.5,2,3]:
        observation=classifier.predict(frame,person,[])
        assert observation is None
        events.extend(rules.step([person],[],now,100,220)[0])
    assert not events and rules.ppe_status[7]['vest']=='unknown'
    script="""
const assert=require('node:assert/strict'),ui=require('./app/static/incident-ui.js');
const html=ui.statesMarkup({},[{track_id:7,tracking:'measured',ppe:{vest:{state:'unknown',observed_state:'unknown',confirmed_state:'unknown',fresh:true,reason:'no_reliable_evidence',observed_at_s:3}}}],{now:3,labels:{no_vest:'Сигнальная одежда'}});
assert.match(html,/Человек #7/);assert.match(html,/Сигнальная одежда/);
assert.match(html,/Нет подтверждённого состояния/);
assert.doesNotMatch(html,/data-confirmed-state="(?:violation|clear)"|Жилет отсутствует|Жилет обнаружен|ordinary/);
"""
    result=subprocess.run(['node','-e',script],cwd=PROJECT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
