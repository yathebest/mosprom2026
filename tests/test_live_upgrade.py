import time
import numpy as np

from app.core import Person,RunConfig
from app.scenario_models import Observation
from app.scenarios import ScenarioRules


def worker(tid=1,box=(10,0,60,180)):
    return Person(box,.95,track_id=tid)


def test_preview_keeps_camera_rate_with_jitter_and_limits_fast_sources():
    from app.live import FramePacer
    pacer=FramePacer(25)
    times=[i*.04+(.008 if i%2 else 0) for i in range(150)]
    assert sum(pacer.accept(t) for t in times)>=140
    pacer=FramePacer(25)
    assert 74<=sum(pacer.accept(i/120) for i in range(360))<=77


def no_helmet():
    return Observation((15,2,50,35),.9,'no_hardhat','ppe-head')


def test_short_unknown_ppe_gap_bridges_but_never_confirms_itself():
    rules=ScenarioRules(RunConfig(helmet_detection=True,ppe_hold=.3))
    assert not rules.step([worker()],[no_helmet()],0,100,200)[0]
    assert not rules.step([worker()],[],.2,100,200)[0]
    assert not rules.step([worker()],[no_helmet()],.3,100,200)[0]
    # Unknown time contributes zero support: the former .4s assertion used
    # contiguous-hold semantics and is intentionally replaced by 75% time.
    assert not rules.step([worker()],[no_helmet()],.4,100,200)[0]
    assert not rules.step([worker()],[no_helmet()],.5,100,200)[0]
    events=rules.step([worker()],[no_helmet()],.6,100,200)[0]
    assert len(events)==1
    # Repeated missing evidence is not clearance and must not re-arm an episode.
    for t in [1.,2.,3.,4.]:assert not rules.step([worker()],[],t,100,200)[0]
    assert not rules.step([worker()],[no_helmet()],4.1,100,200)[0]


def test_single_ppe_sample_and_repeated_timestamp_cannot_confirm():
    rules=ScenarioRules(RunConfig(helmet_detection=True,ppe_hold=.3))
    for _ in range(10):assert not rules.step([worker()],[no_helmet()],0,100,200)[0]
    for t in [.2,.4,.6]:assert not rules.step([worker()],[],t,100,200)[0]


def test_reset_confirmation_requires_three_new_measurements():
    from app.scenarios import EvidenceGate
    gate=EvidenceGate()
    assert gate.update(0,True,.1,.5,1) is None
    assert gate.update(.2,True,.1,.5,1) is None
    # An explicit unknown observation, not mutation of a legacy private hold
    # field, interrupts time support and requires three fresh measurements.
    assert gate.update(.25,None,.1,.5,1) is None
    assert gate.update(.3,True,.1,.5,1) is None
    assert gate.update(.5,True,.1,.5,1) is None
    assert gate.update(.6,True,.1,.5,1) is not None


def test_specialist_agreement_resolves_legacy_model_conflict():
    rules=ScenarioRules(RunConfig(helmet_detection=True,ppe_hold=.3))
    objects=[no_helmet(),Observation((15,2,50,35),.99,'no_hardhat','ppe-head-classifier'),
             Observation((15,2,50,35),.93,'hardhat','ppe-safetyvision')]
    events=[]
    for t in [0,.2,.4]:events+=rules.step([worker()],objects,t,100,200)[0]
    assert len(events)==1 and rules.ppe_details[1]['helmet']['reason']=='specialist_consensus'
    # Independent specialists disagreeing must abstain.
    objects[1].label='hardhat'
    rules.step([worker()],objects,.5,100,200)
    assert rules.ppe_status[1]['helmet']=='unknown'


def test_weak_proposal_does_not_veto_confident_classifier():
    from app.scenarios import ppe_measurement
    weak=Observation((15,2,50,35),.28,'hardhat','ppe-head')
    verified=Observation((15,2,50,35),.98,'no_hardhat','ppe-head-classifier')
    assert ppe_measurement([weak,verified],{'hardhat'},{'no_hardhat'},.5,.15)[0]=='missing'
    assert ppe_measurement([weak],{'hardhat'},{'no_hardhat'},.5,.15)[0]=='unknown'
    weak.confidence=.8
    assert ppe_measurement([weak,verified],{'hardhat'},{'no_hardhat'},.5,.15)[0]=='unknown'


def test_phone_occlusion_does_not_repeat_an_alert_and_long_clearance_rearms():
    p=worker();p.keypoints=np.zeros((17,3),np.float32);p.keypoints[9]=[30,60,.9]
    phone=Observation((25,55,35,65),.9,'cell_phone','phone-objects')
    rules=ScenarioRules(RunConfig(phone_detection=True,object_hold=.3));events=[]
    for t in [0,.2,.4]:events+=rules.step([p],[phone],t,100,200)[0]
    assert len(events)==1
    for t in [.6,.8,1.]:events+=rules.step([p],[],t,100,200)[0]
    for t in [1.2,1.4,1.6]:events+=rules.step([p],[phone],t,100,200)[0]
    assert len(events)==1
    # New safe-state policy requires both hands actually visible before the
    # absence of a phone can close an interval. One occluded hand is unknown.
    p.keypoints[10]=[50,60,.9]
    for t in np.arange(2.,6.6,.5):events+=rules.step([p],[],float(t),100,200)[0]
    for t in [7.,7.2,7.4]:events+=rules.step([p],[phone],t,100,200)[0]
    assert len(events)==2


def test_id_switch_deduplicates_one_worker_but_not_visible_neighbours():
    rules=ScenarioRules(RunConfig(helmet_detection=True,ppe_hold=.3))
    events=[]
    for t in [0,.2,.4]:events+=rules.step([worker()],[no_helmet()],t,100,200)[0]
    for t in [.6,.8,1.]:events+=rules.step([worker(2)],[no_helmet()],t,100,200)[0]
    assert len(events)==1
    # A second clearly separate worker has independent incident ownership.
    obj=Observation((75,2,100,35),.9,'no_hardhat','ppe-head')
    for t in [1.2,1.4,1.6]:events+=rules.step([worker(2),worker(3,(70,0,110,180))],[no_helmet(),obj],t,120,200)[0]
    assert len(events)==2 and events[-1]['track_id']==3


def test_tiny_heads_are_unknown_even_for_confident_negative_proposals():
    rules=ScenarioRules(RunConfig(helmet_detection=True))
    small=worker(box=(10,0,20,40));obj=Observation((11,0,19,9),.99,'no_hardhat','ppe-head')
    for t in [0,.5,1.,1.5,2.]:assert not rules.step([small],[obj],t,100,100)[0]
    assert rules.ppe_details[1]['helmet']['reason']=='insufficient_pixels'


def test_slow_object_models_do_not_block_video_or_reuse_evidence(monkeypatch,tmp_path):
    from app import live
    class Backend:
        name='slow-test-model';scene_state=None
        def predict_people(self,frame):
            time.sleep(.01);return [worker()]
        def predict_scenarios(self,frame,people):
            time.sleep(.35);return []
    class Tracker:
        def __init__(self,*args,**kwargs):pass
        def update(self,people,*args):return people
    monkeypatch.setattr(live,'create_backend',lambda *a,**k:Backend())
    monkeypatch.setattr(live,'PeopleTracker',Tracker)
    monkeypatch.setattr(live.db,'DATA',tmp_path)
    monkeypatch.setattr(live.db,'add_job',lambda *a,**k:None)
    monkeypatch.setattr(live.db,'update_job',lambda *a,**k:None)
    session=live.LiveSession(RunConfig(fast_motion=False),'browser').start()
    try:
        for i in range(30):
            session.ingest(np.full((200,300,3),i*5,np.uint8));time.sleep(.05)
    finally:
        session.stop_event.set();session.worker.join(timeout=5)
    assert session.state=='stopped' and not session.worker.is_alive()
    assert session.display_count>=20 and session.processed>=8
    assert 2<=session.scenario_count<session.processed
    assert session.report()['display_fps']>12
    assert session.shape==(200,300,3)
    assert session.backend is None and session.report()['model']=='slow-test-model'
    assert all(q.empty() for q in (session.frames,session.display_frames,session.record_frames,session.scenario_frames))


def test_slow_live_relink_requires_close_spatial_continuity():
    from app.inference import StableTracks
    tracks=StableTracks(10,max_gap=3.)
    first=tracks.update([worker()],0)[0].track_id
    assert tracks.update([worker()],2)[0].track_id==first
    assert tracks.update([worker(box=(35,0,85,180))],4)[0].track_id!=first


def test_single_cuda_pipeline_uses_each_source_frame_once_without_object_queue(monkeypatch,tmp_path):
    from app import live
    class Backend:
        name='single-device';device='cuda:0';scene_state=None
        def __init__(self):self.people_frames=[];self.object_frames=[]
        def predict(self,frame):raise AssertionError('Use the explicit staged contract')
        def predict_people(self,frame):
            self.people_frames.append(int(frame[0,0,0]));return [worker()]
        def predict_scenarios(self,frame,people):
            self.object_frames.append(int(frame[0,0,0]));assert people[0].track_id==1
            return []
    class Tracker:
        def __init__(self,*args,**kwargs):pass
        def update(self,people,*args):return people
    backend=Backend()
    monkeypatch.delenv('SAFETY_LIVE_SPLIT_INFERENCE',raising=False)
    monkeypatch.setattr(live,'create_backend',lambda *a,**k:backend)
    monkeypatch.setattr(live,'PeopleTracker',Tracker)
    monkeypatch.setattr(live.db,'DATA',tmp_path)
    monkeypatch.setattr(live.db,'add_job',lambda *a,**k:None)
    monkeypatch.setattr(live.db,'update_job',lambda *a,**k:None)
    session=live.LiveSession(RunConfig(fast_motion=False),'browser').start()
    try:
        for i in range(18):
            session.ingest(np.full((120,160,3),i,np.uint8));time.sleep(.04)
    finally:
        session.stop_event.set();session.worker.join(timeout=5)
    assert session.state=='stopped' and session.pipeline_mode=='serial'
    assert len(backend.people_frames)>=4
    assert backend.people_frames==backend.object_frames
    assert len(set(backend.object_frames))==len(backend.object_frames)
    assert session.scenario_count==session.processed
    assert session.scenario_frames.empty() and session.display_count>=8


def test_pose_proposal_is_neutral_and_requires_visible_face():
    from app.scenario_models import pose_head_proposal
    p=worker();assert pose_head_proposal(p,100,200) is None
    p.keypoints=np.zeros((17,3),np.float32)
    p.keypoints[:3]=[[30,20,.9],[25,18,.9],[35,18,.9]]
    proposal=pose_head_proposal(p,100,200)
    assert proposal.label=='head_proposal' and proposal.owner_box==p.box
    assert proposal.box[1]<18 and proposal.box[3]>20
    p.keypoints[2,2]=.1
    assert pose_head_proposal(p,100,200) is None


def test_smoking_crop_preserves_owner_coordinates_and_skips_tiny_people():
    from app.scenario_models import ScenarioModels
    class Detector:
        name='smoking'
        def __init__(self):self.calls=0
        def predict(self,frame):
            self.calls+=1
            return [] if self.calls==1 else [Observation((15,15,25,25),.9,'cigarette',self.name)]
    model=Detector();pipeline=ScenarioModels.__new__(ScenarioModels)
    pipeline.models=[model];pipeline.classifiers=[];pipeline.allowed={'cigarette'}
    pipeline.ppe_crops=False;pipeline.phone_crops=False;pipeline.smoking_crops=True
    pipeline.smoking_min_height=120
    big=worker(box=(50,50,150,250));small=worker(2,(180,100,200,150))
    objects=pipeline.predict(np.zeros((300,300,3),np.uint8),[big,small])
    assert model.calls==2 and len(objects)==1
    assert objects[0].box==(45,45,55,55) and objects[0].owner_box==big.box


def test_additional_fire_model_does_not_change_smoke_evidence():
    from app.scenario_models import ScenarioModels
    class Detector:
        def __init__(self,name):self.name=name
        def predict(self,frame):
            return [Observation((10,10,40,40),.9,label,self.name) for label in ('fire','smoke')]
    pipeline=ScenarioModels.__new__(ScenarioModels)
    pipeline.models=[Detector('fire-smoke'),Detector('fire-dfire')];pipeline.classifiers=[]
    pipeline.allowed={'fire','smoke'};pipeline.ppe_crops=False;pipeline.phone_crops=False
    objects=pipeline.predict(np.zeros((100,100,3),np.uint8))
    assert {(o.model,o.label) for o in objects}=={('fire-smoke','fire'),('fire-smoke','smoke'),('fire-dfire','fire')}


def test_torso_proposal_requires_measured_shoulders_and_hips():
    from app.scenario_models import pose_torso_proposal
    p=worker();assert pose_torso_proposal(p,100,200) is None
    p.keypoints=np.zeros((17,3),np.float32)
    p.keypoints[[5,6,11,12]]=[[20,50,.9],[50,50,.9],[25,100,.9],[45,100,.9]]
    proposal=pose_torso_proposal(p,100,200)
    assert proposal.label=='torso_proposal' and proposal.owner_box==p.box
    assert proposal.box[0]<20 and proposal.box[3]>100
    p.keypoints[12,2]=.3
    assert pose_torso_proposal(p,100,200) is None


def test_direct_h264_clip_preserves_frame_order_and_timing(monkeypatch,tmp_path):
    import cv2
    from app import live
    if not live.ffmpeg_path():
        import pytest
        pytest.skip('FFmpeg is required for video export')
    session=live.LiveSession.__new__(live.LiveSession)
    session.folder=tmp_path;session.id='test'
    (tmp_path/'clips').mkdir()
    saved=[];payloads=[]
    monkeypatch.setattr(live.db,'set_clip',lambda *args:saved.append(args))
    monkeypatch.setattr(live.db,'patch_payload',lambda *args:payloads.append(args))
    frames=[(i*.04,cv2.imencode('.jpg',np.full((120,200,3),i*10,np.uint8))[1].tobytes()) for i in range(20)]
    session.finalize_clip({'id':'evidence','frames':frames,'until':.76,'alert_s':.4})
    assert saved and payloads[-1][1]['clip_state']=='ready'
    cap=cv2.VideoCapture(str(tmp_path/'clips/evidence.mp4'))
    assert cap.get(cv2.CAP_PROP_FPS)==25
    decoded=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        decoded.append(float(frame.mean()))
    cap.release()
    assert len(decoded)==20 and decoded[0]<5 and decoded[-1]>180
