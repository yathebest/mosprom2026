import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from app.core import Person,Rules,RunConfig,Zone
from app.inference import PersonConfirmation,PeopleTracker,semantic_person_evidence
from app.inference import validate_person_class


def test_neutral_head_weights_cannot_be_installed_as_the_person_detector():
    import pytest
    validate_person_class(SimpleNamespace(names={0:'person',67:'cell phone'}))
    with pytest.raises(ValueError,match='Phone/head'):
        validate_person_class(SimpleNamespace(names={0:'head_proposal',67:'cell_phone'}))


def person(tid=1,**kwargs):
    return Person((10,10,50,110),.9,track_id=tid,**kwargs)


def test_machine_with_repeated_high_confidence_body_pose_remains_unknown():
    points=np.zeros((17,3),np.float32)
    for i in range(5,17):points[i]=[20+i,20+i*5,.95]
    confirmation=PersonConfirmation(.4,3)
    for now in range(10):
        p=person(keypoints=points,detection_sources=('person:1',))
        assert not semantic_person_evidence(p)
        assert not confirmation.update(p,float(now))


def test_far_person_without_pose_can_confirm_by_two_models_not_two_scales():
    confirmation=PersonConfirmation(.4,3)
    for now in [0.,1.]:assert not confirmation.update(person(detection_sources=('person:0','person:1')),now)
    assert confirmation.update(person(detection_sources=('person:0','person:1')),2.)
    # Source names are model identities: one model remains one vote.
    assert not semantic_person_evidence(person(detection_sources=('person:1','person:1')))


def test_predicted_weak_duplicate_and_spatial_jump_do_not_supply_three_fresh_votes():
    confirmation=PersonConfirmation(.4,3)
    strong=lambda **kw:person(detection_sources=('person:0','person:1'),**kw)
    assert not confirmation.update(strong(),0.)
    assert not confirmation.update(strong(),0.)
    assert not confirmation.update(strong(),float('nan'))
    assert not confirmation.update(strong(observed=False),1.)
    weak=strong();weak.confidence=.2
    assert not confirmation.update(weak,1.5)
    assert not confirmation.update(strong(),2.)
    jump=strong();jump.box=(500,500,540,600)
    assert not confirmation.update(jump,2.5)
    assert len(confirmation.states[1]['votes'])==1


def test_people_tracker_keeps_tentative_box_without_exposing_id_or_rule_access(monkeypatch):
    class Boxes:
        def __init__(self,data,shape):self.data=data
    monkeypatch.setitem(sys.modules,'ultralytics.engine.results',SimpleNamespace(Boxes=Boxes))
    class ByteTracker:
        def update(self,boxes):return [[*row[:4],11,row[4],0,i] for i,row in enumerate(boxes.data)]
    from app.inference import StableTracks
    tracker=PeopleTracker.__new__(PeopleTracker);tracker.tracker=ByteTracker();tracker.confidence=.4
    tracker.stable=StableTracks(10,3);tracker.confirmation=PersonConfirmation(.4,3)
    tracker.fps=10;tracker.frame=0;tracker.recent={}
    cfg=RunConfig(fast_motion=False,zones=[Zone(points=[(0,0),(1,0),(1,1),(0,1)])])
    rules=Rules(cfg)
    ids=[]
    for now in [0.,.2,.4]:
        output=tracker.update([person(detection_sources=('person:0','person:1'))],(120,100,3),now)
        assert len(output)==1
        p=output[0];ids.append(p.track_id)
        if now<.4:
            assert p.tracking_state=='tentative' and not p.alert_eligible and p.observed
            assert rules.step(output,now,100,120)[0]==[] and rules.state_rows=={}
        else:assert p.tracking_state=='confirmed' and p.alert_eligible
    assert ids[:2]==[-1,-1] and ids[2]>0


def test_saved_factory_helmet_people_confirm_but_machine_id3_never_does():
    path=Path(__file__).parent/'fixtures/person_false_positive_case.json'
    measurements=json.loads(path.read_text(encoding='utf-8'))['measurements']
    confirmation=PersonConfirmation(.4,3);ever=set();first_confirmed={}
    for measurement in measurements:
        for row in measurement['people']:
            row=dict(row)
            if row['keypoints'] is not None:row['keypoints']=np.asarray(row['keypoints'],np.float32)
            p=Person(**row)
            if confirmation.update(p,measurement['source_time_s']):
                ever.add(p.track_id);first_confirmed.setdefault(p.track_id,measurement['source_time_s'])
    assert ever=={1,2}
    assert all(timestamp>=.39 for timestamp in first_confirmed.values())
