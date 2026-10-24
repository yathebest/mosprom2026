import numpy as np

from app.core import Rules, RunConfig
from app.inference import YoloBackend


def test_temporal_clip_score_alone_does_not_create_fall_alert():
    rules=Rules(RunConfig(fall_detection=True,body_hold=.2,fast_motion=False))
    scene={'lying_probability':.02,'fall_probability':.99,'fall_threshold':.54}
    assert all(not rules.step([],t,100,100,scene)[0] for t in [0,.1,.2,.3,.4])


def test_temporal_score_and_rgb_posture_confirm_one_fall_episode():
    rules=Rules(RunConfig(fall_detection=True,body_hold=.2,fast_motion=False))
    scene={'lying_probability':.95,'fall_probability':.97,'fall_threshold':.54}
    events=[]
    for t in [0,.1,.2,.3,.4]:events.extend(rules.step([],t,100,100,scene)[0])
    assert len(events)==1
    assert events[0]['kind']=='possible_fall'
    assert events[0]['source']=='rgb_temporal_fusion'
    assert events[0]['score']==.97 and events[0]['alert_s']==.2


def test_temporal_warmup_preserves_existing_person_down_rule():
    rules=Rules(RunConfig(fall_detection=True,lying_hold=1,fast_motion=False))
    scene={'lying_probability':.95,'fall_probability':None,'fall_threshold':.54}
    events=[]
    for t in np.arange(0,1.2,.1):events.extend(rules.step([],float(t),100,100,scene)[0])
    assert len(events)==1 and events[0]['kind']=='person_down'


def test_backend_temporal_hook_uses_source_time_and_keeps_distinct_scores():
    class Temporal:
        threshold=.54
        def predict(self,frame,now):self.timestamp=now;return .91
    backend=YoloBackend.__new__(YoloBackend)
    backend.temporal_fall=Temporal();backend.scene_state={'lying_probability':.1,'scope':'scene'}
    backend.predict_temporal(np.zeros((10,10,3),np.uint8),12.5)
    assert backend.temporal_fall.timestamp==12.5
    assert backend.scene_state['lying_probability']==.1
    assert backend.scene_state['fall_probability']==.91


def test_temporal_window_resamples_slow_camera_without_joining_outages():
    from collections import deque
    import torch
    from app.fall_temporal import TemporalFallClassifier
    class Network:
        def encode(self,x):return x.mean().expand(1,512)
        def classify_features(self,x):self.sequence=x;return torch.tensor([[0.,1.]])
    temporal=TemporalFallClassifier.__new__(TemporalFallClassifier)
    temporal.model=Network();temporal.device='cpu';temporal.frames=deque(maxlen=64)
    temporal.last_sample=None;temporal.window_s=15/8
    temporal.mean=np.zeros((3,1,1),np.float32);temporal.std=np.ones((3,1,1),np.float32)
    for now,value in [(0,0),(1,127)]:
        assert temporal.predict(np.full((30,30,3),value,np.uint8),now) is None
    assert temporal.predict(np.full((30,30,3),255,np.uint8),2)>.5
    sequence=temporal.model.sequence
    assert tuple(sequence.shape)==(1,16,512)
    assert bool(torch.all(sequence[:,1:]>=sequence[:,:-1]))
    assert temporal.predict(np.zeros((30,30,3),np.uint8),10) is None
    assert temporal.predict(np.zeros((30,30,3),np.uint8),.1) is None


def test_ppe_refinement_skips_large_people_but_keeps_small_person_crops():
    from app.core import Person
    from app.scenario_models import ScenarioModels
    class Detector:
        name='ppe-head'
        def __init__(self):self.calls=0
        def predict(self,frame):self.calls+=1;return []
    pipeline=ScenarioModels.__new__(ScenarioModels);detector=Detector()
    pipeline.models=[detector];pipeline.classifiers=[];pipeline.allowed=set()
    pipeline.ppe_crops=True;pipeline.phone_crops=False;pipeline.max_people=8
    pipeline.ppe_labels={'head':{'hardhat','no_hardhat'},'torso':{'vest','no_vest'}}
    frame=np.zeros((200,100,3),np.uint8)
    pipeline.predict(frame,[Person((5,5,95,190),.9)])
    assert detector.calls==1
    detector.calls=0
    pipeline.predict(frame,[Person((5,5,35,80),.9)])
    assert detector.calls==2
