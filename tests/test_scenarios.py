import numpy as np
import pytest

from app.core import Person,RunConfig,Zone
from app.scenario_models import Observation
from app.scenarios import ScenarioRules,base_config


def person(tid=1,box=(40,0,60,100),keypoints=None):return Person(box,.95,keypoints,tid)
def obj(label,box=(43,0,57,22),score=.95):return Observation(box,score,label,'test-detector')
def execute(config,people,objects,details=None):
    rules=ScenarioRules(config);events=[]
    for t in np.arange(0,4,.1):events.extend(rules.step(people,objects,float(t),100,100,details)[0])
    return events,rules
def zone(kind):return Zone(name=kind,kind=kind,points=[(0,0),(1,0),(1,1),(0,1)])


def test_unknown_ppe_does_not_mean_missing():
    events,rules=execute(RunConfig(helmet_detection=True),[person()],[])
    assert events==[] and rules.ppe_status[1]['helmet']=='unknown'


def test_direct_no_helmet_requires_confirmation_and_emits_once():
    events,_=execute(RunConfig(helmet_detection=True),[person()],[obj('no_hardhat')])
    assert len(events)==1 and events[0]['kind']=='no_helmet' and events[0]['alert_s']==1.5


def test_equally_strong_contradictory_ppe_requires_review():
    events,rules=execute(RunConfig(helmet_detection=True),[person()],[obj('hardhat'),obj('no_hardhat')])
    assert events==[] and rules.ppe_status[1]['helmet']=='unknown'
    assert rules.ppe_details[1]['helmet']['reason']=='conflicting_models'


def test_ppe_is_attached_to_correct_worker():
    events,rules=execute(RunConfig(helmet_detection=True),[person(),person(2,(75,0,95,100))],
                         [obj('hardhat'),obj('no_hardhat',(78,0,92,22))])
    assert [e['track_id'] for e in events]==[2]
    assert rules.ppe_status[1]['helmet']=='worn'


def test_overlapping_people_do_not_get_ambiguous_ppe_assignments():
    events,_=execute(RunConfig(helmet_detection=True),[person(),person(2)], [obj('no_hardhat')])
    assert not events


def test_brief_negative_ppe_frame_cannot_confirm():
    rules=ScenarioRules(RunConfig(helmet_detection=True))
    for i in range(40):
        events,_=rules.step([person()],[obj('no_hardhat')] if i==3 else [],i/10,100,100)
        assert not events


@pytest.mark.parametrize('key,label,kind',[('vest','no_safety_vest','no_vest'),('mask','no_mask','no_mask')])
def test_other_explicit_ppe_classes(key,label,kind):
    cfg=RunConfig(**{key+'_detection':True})
    events,_=execute(cfg,[person()],[obj(label,(42,0,58,75))])
    assert [e['kind'] for e in events]==[kind]


def test_no_harness_requires_configured_height_region():
    cfg=RunConfig(harness_detection=True,zones=[Zone(name='height',kind='height',points=[(.7,0),(1,0),(1,1),(.7,1)])])
    assert not execute(cfg,[person()],[obj('no_harness',(40,0,60,100))])[0]
    cfg=RunConfig(harness_detection=True,zones=[zone('height')])
    events,_=execute(cfg,[person()],[obj('no_harness',(40,0,60,100))])
    assert events[0]['kind']=='no_harness_at_height' and events[0]['zone']=='height'


@pytest.mark.parametrize('kind',['under_load','under_conveyor'])
def test_forbidden_passage_uses_explicit_context_polygon(kind):
    cfg=RunConfig(zones=[zone(kind)],fast_motion=False)
    events,_=execute(cfg,[person()],[])
    assert len(events)==1 and events[0]['kind']==kind
    assert events[0]['context_source']=='operator_polygon'
    assert base_config(cfg).zones==[]


def test_barrier_requires_observed_entry_not_person_already_inside():
    cfg=RunConfig(zones=[Zone(name='b',kind='barrier',points=[(.3,0),(1,0),(1,1),(.3,1)])])
    assert not execute(cfg,[person()],[])[0]
    rules=ScenarioRules(cfg);rules.step([person(box=(0,0,20,100))],[],0,100,100)
    events=[]
    for i in range(1,30):events.extend(rules.step([person()],[],i/10,100,100)[0])
    assert [e['kind'] for e in events]==['barrier_crossing']


def test_phone_requires_hand_and_movement_or_working_area():
    k=np.zeros((17,3),np.float32);k[9]=[50,50,.95]
    cfg=RunConfig(phone_detection=True)
    assert not execute(cfg,[person(keypoints=k)],[obj('cell_phone',(45,45,55,55))])[0]
    events,_=execute(cfg,[person(keypoints=k)],[obj('cell_phone',(45,45,55,55))],{1:{'speed':1.0}})
    assert [e['kind'] for e in events]==['possible_phone_use']
    assert not execute(cfg,[person()],[obj('cell_phone',(45,45,55,55))],{1:{'speed':1}})[0]
    # Reject a spurious phone bbox spanning most of a person even near a wrist.
    assert not execute(cfg,[person(keypoints=k)],[obj('cell_phone',(20,10,80,90))],{1:{'speed':1}})[0]


def test_fire_is_deduplicated_and_allowed_zone_suppresses_it():
    fire=obj('fire',(40,30,60,60));cfg=RunConfig(fire_detection=True)
    events,_=execute(cfg,[],[fire]);assert [e['kind'] for e in events]==['fire']
    cfg=RunConfig(fire_detection=True,zones=[zone('fire_allowed')])
    assert not execute(cfg,[],[fire])[0]


def test_smoking_upstream_label_and_allowed_place():
    smoking=obj('smooking',(40,30,60,60))
    assert execute(RunConfig(smoking_detection=True),[],[smoking])[0][0]['kind']=='possible_smoking'
    assert not execute(RunConfig(smoking_detection=True,zones=[zone('smoking_allowed')]),[],[smoking])[0]


def test_proximity_requires_moving_detected_equipment_or_explicit_machine_zone():
    cfg=RunConfig(proximity_detection=True)
    assert not execute(cfg,[person()],[obj('truck',(60,40,80,80))])[0]
    rules=ScenarioRules(cfg);events=[]
    for i in range(30):
        x=60+i*.7
        events.extend(rules.step([person()],[obj('truck',(x,40,x+20,80))],i/10,100,100)[0])
    assert events and events[0]['kind']=='equipment_proximity'
    assert events[0]['distance_m'] is None
    events,_=execute(RunConfig(proximity_detection=True,zones=[zone('equipment')]),[person()],[])
    assert events[0]['context_source']=='operator_polygon'


def test_load_posture_has_explicit_object_and_pose_limits():
    k=np.zeros((17,3),np.float32)
    k[5]=[15,20,.95];k[6]=[20,20,.95];k[11]=[60,60,.95];k[12]=[65,60,.95];k[9]=[60,70,.95]
    cfg=RunConfig(load_posture_detection=True)
    events,_=execute(cfg,[person(box=(0,0,100,100),keypoints=k)],[obj('suitcase',(58,60,78,90))],{1:{'speed':1}})
    assert events and events[0]['kind']=='unsafe_load_posture'
    assert not execute(cfg,[person(box=(0,0,100,100),keypoints=k)],[],{1:{'speed':1}})[0]
