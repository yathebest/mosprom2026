from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pytest

from app.core import Person,RunConfig
from app import scenario_models as models
from app.scenario_models import Observation,ScenarioModels


def test_phone_checkpoint_never_supplies_people_head_or_other_coco_classes():
    detector=SimpleNamespace(name='phone-objects',predict=lambda frame:[
        Observation((10,10,30,40),.99,'head_proposal','phone-objects'),
        Observation((10,10,50,90),.99,'person','phone-objects'),
        Observation((10,10,70,90),.99,'truck','phone-objects'),
        Observation((22,35,30,45),.7,'cell_phone','phone-objects')])
    scenario=ScenarioModels.__new__(ScenarioModels)
    scenario.models=[detector];scenario.classifiers=[];scenario.person_confidence=.4
    scenario.allowed={'cell_phone','head_proposal','person','truck'}
    scenario.ppe_crops=False;scenario.phone_crops=False;scenario.smoking_crops=False
    result=scenario.predict(np.zeros((100,100,3),np.uint8),[Person((10,10,50,90),.9)])
    assert [o.label for o in result]==['cell_phone']


def test_phone_crop_keeps_phone_only_even_with_other_scenarios_enabled():
    detector=SimpleNamespace(name='phone-objects',predict=lambda frame:[
        Observation((1,1,8,8),.8,'head_proposal','phone-objects'),
        Observation((15,25,20,35),.65,'cell_phone','phone-objects')])
    scenario=ScenarioModels.__new__(ScenarioModels)
    scenario.models=[detector];scenario.classifiers=[];scenario.person_confidence=.4
    scenario.allowed={'cell_phone','person','truck'};scenario.ppe_crops=False
    scenario.phone_crops=True;scenario.smoking_crops=False
    result=scenario.predict(np.zeros((200,200,3),np.uint8),[Person((50,20,100,180),.9)])
    assert len(result)>=1 and {o.label for o in result}=={'cell_phone'}
    assert any(o.model=='phone-objects:person_crop' for o in result)


@pytest.mark.parametrize('class_zero',['head_proposal','person'])
def test_phone67_contract_supports_candidate_and_reversible_baseline_without_lowering_threshold(tmp_path,monkeypatch,class_zero):
    calls=[]
    def detector(path,confidence):
        calls.append(confidence)
        return SimpleNamespace(name='phone-objects',names={0:class_zero,67:'cell_phone'})
    monkeypatch.setattr(models,'OnnxDetector',detector)
    result=ScenarioModels(tmp_path,RunConfig(phone_detection=True,phone_confidence=.3,object_confidence=.5))
    assert calls==[.3]
    assert result.models[0].names[0]==class_zero


def test_incorrect_phone_class_mapping_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(models,'OnnxDetector',lambda *args:SimpleNamespace(name='phone-objects',names={0:'cell_phone',67:'head_proposal'}))
    with pytest.raises(ValueError,match='67'):
        ScenarioModels(tmp_path,RunConfig(phone_detection=True))


@pytest.mark.parametrize('decision,checksum,message',[
    ('reject',None,'approve'),('accept','0'*64,'differs')])
def test_phone_install_rejects_unapproved_or_changed_artifact_before_mutating_runtime(tmp_path,decision,checksum,message):
    from scripts.install_phone_candidate import install
    candidate=tmp_path/'candidate.onnx';candidate.write_bytes(b'candidate')
    models=tmp_path/'models';models.mkdir();runtime=models/'phone-objects.onnx';runtime.write_bytes(b'original')
    report={'deployment_decision':decision,'confidence':.3,'phone_only_decoder_enabled':False,
        'models':{'candidate':{'sha256':checksum or hashlib.sha256(b'candidate').hexdigest()}}}
    comparison=tmp_path/'comparison.json';comparison.write_text(json.dumps(report))
    with pytest.raises(ValueError,match=message):install(candidate,comparison,models)
    assert runtime.read_bytes()==b'original'
    assert not (models/'phone-objects.onnx.installing').exists()
