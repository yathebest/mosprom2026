import numpy as np

from app.core import Person, Rules, RunConfig, Zone
from app.inference import PeopleTracker
from app.scenario_models import Observation, person_regions
from app.scenarios import ScenarioRules, person_for


def test_associated_weak_detection_remains_visible_without_alert_permission():
    tracker=PeopleTracker(10,.4)
    for t in [-.2,-.1,0]:
        first=tracker.update([Person((10,10,50,110),.95,detection_sources=('person:0','person:1'))],(120,100,3),t)[0]
    weak=tracker.update([Person((11,10,51,110),.2)],(120,100,3),.1)[0]
    assert first.track_id>0 and first.track_id==weak.track_id and weak.observed and not weak.alert_eligible
    assert weak.confidence==.2


def test_real_time_gap_does_not_reuse_id_when_live_processing_is_slow():
    from app.inference import StableTracks
    tracker=StableTracks(10)
    first=tracker.update([Person((10,10,50,110),.95,track_id=1)],0)[0].track_id
    second=tracker.update([Person((10,10,50,110),.95,track_id=1)],2)[0].track_id
    assert second!=first


def test_unobserved_track_expires_and_cannot_create_a_zone_alert():
    tracker=PeopleTracker(10,.4)
    for t in [-.2,-.1,0]:
        tracker.update([Person((10,10,50,110),.95,detection_sources=('person:0','person:1'))],(120,100,3),t)
    predicted=tracker.update([],(120,100,3),.1)
    assert len(predicted)==1 and not predicted[0].observed
    config=RunConfig(zones=[Zone(points=[(0,0),(1,0),(1,1),(0,1)])],zone_hold=.2,fast_motion=False)
    rules=Rules(config)
    for i in range(10):assert rules.step(predicted,i/10,100,120)[0]==[]
    assert tracker.update([],(120,100,3),.6)==[]


def test_predicted_gap_resets_pending_confirmation():
    cfg=RunConfig(zones=[Zone(points=[(0,0),(1,0),(1,1),(0,1)])],zone_hold=.3,fast_motion=False)
    rules=Rules(cfg)
    measured=Person((10,10,50,90),.9,track_id=1)
    predicted=Person(measured.box,.9,track_id=1,observed=False,alert_eligible=False)
    rules.step([measured],0,100,100)
    rules.step([predicted],.2,100,100)
    assert rules.step([measured],.4,100,100)[0]==[]


def test_stale_ppe_can_be_displayed_but_not_accumulate_confirmation():
    rules=ScenarioRules(RunConfig(helmet_detection=True,ppe_hold=.3))
    people=[Person((10,0,50,100),.9,track_id=1)]
    negative=[Observation((15,0,45,30),.9,'no_hardhat','test')]
    assert rules.step(people,negative,0,100,100)[0]==[]
    for t in [.1,.2,.3,.4]:assert rules.step(people,[],t,100,100)[0]==[]
    assert rules.ppe_status[1]['helmet']=='missing'
    assert not rules.ppe_details[1]['helmet']['fresh']
    rules.step(people,[],.6,100,100)
    assert rules.ppe_status[1]['helmet']=='unknown'


def test_weak_person_never_confirms_strong_ppe_violation():
    rules=ScenarioRules(RunConfig(vest_detection=True,ppe_hold=.3,signal_clothing_policy='vest_only'))
    people=[Person((10,0,50,100),.2,track_id=1,alert_eligible=False)]
    obs=[Observation((12,25,48,80),.95,'no_vest','test')]
    for i in range(20):assert rules.step(people,obs,i/10,100,100)[0]==[]


def test_crop_ownership_does_not_transfer_to_neighbour():
    people=[Person((10,0,50,100),.9,track_id=1),Person((40,0,80,100),.9,track_id=2)]
    obs=Observation((40,5,50,20),.95,'hardhat','crop',owner_box=(10,0,50,100))
    assert person_for(obs,people,'head')==1
    duplicate=Person((10,0,50,100),.9,track_id=3)
    assert person_for(obs,people+[duplicate],'head') is None


def test_head_and_torso_crops_preserve_frame_bounds():
    regions=person_regions(Person((-10,-10,60,150),.9),100,100)
    assert set(regions)=={'head','torso'}
    assert all(0<=x<=100 for box in regions.values() for x in box)


def test_classifier_proposals_are_owned_before_tracker_assigns_ids():
    from app.scenario_models import ScenarioModels
    people=[Person((0,0,40,100),.9),Person((60,0,100,100),.9)]
    objects=[Observation((5,2,35,25),.9,'hardhat','test'),
             Observation((65,2,95,25),.95,'no_hardhat','test')]
    class Detector:
        name='test'
        def predict(self,frame):return objects
    class Classifier:
        region='head'
        def __init__(self):self.seen=[]
        def predict(self,frame,person,proposals):
            self.seen.append([o.label for o in proposals])
    pipeline=ScenarioModels.__new__(ScenarioModels)
    classifier=Classifier();pipeline.models=[Detector()];pipeline.classifiers=[classifier]
    pipeline.allowed={'hardhat','no_hardhat'};pipeline.ppe_crops=False
    pipeline.phone_crops=False;pipeline.max_people=8
    pipeline.predict(np.zeros((100,100,3),np.uint8),people)
    assert classifier.seen==[['hardhat'],['no_hardhat']]
