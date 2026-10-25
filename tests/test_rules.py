import numpy as np
import pytest
from pydantic import ValidationError
from app.core import Gate, Person, Rules, RunConfig, Zone, posture


def config(**kwargs):
    return RunConfig(zones=[Zone(points=[(.4,.6),(.8,.6),(.8,.95),(.4,.95)])],
                     fast_motion=False,**kwargs)


def p(box=(50,10,70,80),tid=1,kpts=None):
    return Person(box,.9,kpts,tid)


def test_zone_uses_feet_and_emits_once_for_long_episode():
    engine = Rules(config())
    events = []
    for t in np.arange(0,20,.1):
        fired,_ = engine.step([p()],float(t),100,100)
        events.extend(fired)
    assert len(events)==1
    assert events[0]['alert_s']==1
    assert events[0]['start_s']==0


def test_box_overlapping_zone_does_not_count_if_feet_are_outside():
    engine = Rules(config())
    events = []
    for t in np.arange(0,3,.1):
        fired,_ = engine.step([p((10,0,50,55))],float(t),100,100)
        events.extend(fired)
    assert not events


def test_occlusion_does_not_accumulate_confirmation_time():
    engine = Rules(config())
    engine.step([p()],0,100,100)
    engine.step([p()],.2,100,100)
    engine.step([],1,100,100)
    fired,_ = engine.step([p()],1.5,100,100)
    assert not fired
    assert not engine.step([p()],1.7,100,100)[0]


def test_short_false_frame_does_not_rearm_but_genuine_exit_does():
    gate = Gate()
    emissions = []
    for i in range(70):
        t=i/10
        condition = not (2<=t<2.2 or 3<=t<5.2)
        event=gate.update(t,condition,1,2,.8)
        if event:
            emissions.append((t,event))
    assert len(emissions)==2
    assert emissions[0][1][1]==1 and emissions[1][1][1]==2


def keypoints(lying=False):
    k=np.zeros((17,3),np.float32)
    for i,x in [(5,40),(6,45)]:
        k[i]=[x,30,.95]
    for i,x in [(11,42),(12,47)]:
        k[i]=[x,65,.95]
    if lying:
        for i,x in [(11,77),(12,82)]:
            k[i]=[x,35,.95]
    return k


def test_missing_keypoints_are_unknown():
    assert posture(p())=='unknown'
    assert posture(p(kpts=np.zeros((17,3))))=='unknown'


def test_single_horizontal_frame_is_not_a_fall():
    engine=Rules(RunConfig(fast_motion=False,fall_detection=True))
    engine.step([p((30,0,60,100),kpts=keypoints())],0,100,100)
    fired,_=engine.step([p((30,20,95,60),kpts=keypoints(True))],.1,100,100)
    assert not fired
    for t in np.arange(.2,2,.1):
        fired,_=engine.step([p((30,0,60,100),kpts=keypoints())],float(t),100,100)
        assert not fired


def test_fall_requires_observed_transition_and_persistent_horizontal_pose():
    engine=Rules(RunConfig(fast_motion=False,fall_detection=True))
    # A measured upright frame is not a confirmed upright state. Establish
    # the normal state before testing the actual upright -> down transition.
    for t in np.arange(0,2.5,.1):
        assert not engine.step([p((30,0,60,100),kpts=keypoints())],float(t),100,100)[0]
    events=[]
    for t in np.arange(2.5,9,.1):
        fired,_=engine.step([p((30,20,95,60),kpts=keypoints(True))],float(t),100,100)
        events.extend(fired)
    assert [e['kind'] for e in events]==['possible_fall']
    assert events[0]['alert_s']==3.5


def test_already_lying_person_is_not_reported_as_observed_fall():
    engine=Rules(RunConfig(fast_motion=False,fall_detection=True))
    events=[]
    for t in np.arange(0,7,.1):
        events.extend(engine.step([p((30,20,95,60),kpts=keypoints(True))],float(t),100,100)[0])
    assert [e['kind'] for e in events]==['person_down']


@pytest.mark.parametrize('fps',[5,10,25])
def test_decision_uses_source_time_not_processing_frame_count(fps):
    engine=Rules(config())
    times=[]
    for i in range(fps*3):
        times.extend(e['alert_s'] for e in engine.step([p()],i/fps,100,100)[0])
    assert times==[1.0]


def test_camera_motion_is_not_mislabeled_as_calibrated_speed():
    engine=Rules(RunConfig(fast_motion=True,speed_threshold=1,motion_hold=.5))
    events=[]
    for i in range(30):
        x=i*5
        events.extend(engine.step([p((x,0,x+10,20))],i/10,1000,100)[0])
    assert len(events)==1 and events[0]['kind']=='fast_motion'
    assert 2.4<events[0]['speed_heights_s']<2.6
    assert events[0]['speed_m_s'] is None


def test_invalid_zone_is_rejected():
    with pytest.raises(ValidationError):
        Zone(points=[(0,0),(1,1),(1,0),(0,1)])
    with pytest.raises(ValidationError):
        Zone(points=[(-1,0),(1,0),(1,1)])
