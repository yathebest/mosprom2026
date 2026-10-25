import numpy as np
from app.core import Person,RunConfig
from app.scenario_models import Observation,CropClassifier
from app.scenarios import ScenarioRules


def run(label):
    rules=ScenarioRules(RunConfig(vest_detection=True,ppe_hold=.3))
    p=Person((10,0,60,180),.95,track_id=1)
    events=[]
    for t in [0.,.2,.4,.6]:
        events.extend(rules.step([p],[Observation((15,20,55,150),.99,label,'signal-clothing-classifier')],t,100,200)[0])
    return rules,events


def test_legacy_no_vest_is_not_a_violation_of_reflective_uniform_policy():
    rules,events=run('no_vest')
    assert not events and rules.ppe_status[1]['vest']=='unknown'
    assert rules.ppe_details[1]['vest']['reason']=='signal_clothing_policy_unverified'


def test_explicit_reflective_uniform_complies_and_explicit_ordinary_can_alert():
    rules,events=run('hi_vis_clothing')
    assert not events and rules.ppe_status[1]['vest']=='worn'
    rules,events=run('no_hi_vis_clothing')
    assert len(events)==1 and events[0]['kind']=='no_vest'
    assert events[0]['evidence_label']=='no_hi_vis_clothing'


def test_whole_person_classifier_geometry_and_unobservable_abstention(monkeypatch):
    class Session:
        def run(self,*args):self.input=args[1]['images'];return [np.array([[12.,0.,0.]],np.float32)]
    classifier=CropClassifier.__new__(CropClassifier);classifier.clothing=True
    classifier.class_thresholds=[.9,.98,.9];classifier.labels=['hi_vis','ordinary','unobservable']
    classifier.mean=np.zeros((1,3,1,1),np.float32);classifier.std=np.ones((1,3,1,1),np.float32)
    classifier.input_name='images';classifier.size=224;classifier.provider='CPUExecutionProvider'
    classifier.session=Session();classifier.region='torso';classifier.name='signal-clothing-classifier'
    person=Person((20,10,80,190),.9)
    result=classifier.predict(np.zeros((220,100,3),np.uint8),person,[])
    assert result.label=='hi_vis_clothing' and result.owner_box==person.box
    assert result.box==(18,5,82,195)
    classifier.session.run=lambda *args:[np.array([[0.,0.,12.]],np.float32)]
    assert classifier.predict(np.zeros((220,100,3),np.uint8),person,[]) is None
