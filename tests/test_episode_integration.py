"""Persistence and worker integration: an update closes a row, never an alert."""
from datetime import datetime,timezone
import threading
import time

import cv2
import numpy as np
import pytest

from app import storage as db,video,live
from app.access import Scope,current_scope
from app.core import RunConfig


def activation(episode='episode-a'):
    return dict(track_id=7,kind='no_helmet',zone='',sequence=1,start_s=0.,alert_s=0.,
                episode_id=episode,episode_state='open',end_s=None,
                state_from='unknown',state_to='violation',state_confirmation={'samples':3})


def closure(episode='episode-a'):
    return dict(episode_id=episode,episode_state='closed',end_s=.2,closed_at_s=.2,
                duration_s=.2,closure_reason='confirmed_clear')


@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3')
    db.init(False)
    # These tests cover lifecycle/worker persistence. Real codec export is
    # exercised separately; starting FFmpeg here makes shutdown depend on RAM.
    monkeypatch.setattr(live.LiveSession,'finalize_clip',lambda self,clip:None)
    return tmp_path


def job(key='job',camera='Camera'):
    db.add_job(key,'fixture.mp4',dict(fps=10,width=80,height=80,frames=3,duration_s=.3))
    db.update_job(key,config={'camera':camera})
    (db.DATA/'runs'/key).mkdir(parents=True)


def test_idempotent_episode_preserves_review_and_start_proofs(store):
    job();event=activation();eid=db.add_event('job','Camera',event,'start.jpg')
    db.review(eid,'confirmed');db.set_clip(eid,'start.mp4')
    assert db.add_event('job','Camera',{**event,'sequence':9},'replacement.jpg',return_created=True)==(eid,False)
    clear={**closure(),'state_from':'violation','state_to':'clear','state_confirmation':{'samples':5}}
    assert db.update_episode('job',clear)==eid
    assert db.update_episode('job',{**closure(),'end_s':50,'end_image':'wrong.jpg'}) is None
    db.patch_payload(eid,{'clip_state':'ready'})
    row=db.events('job')[0]
    assert len(db.events('job'))==1
    assert (row['status'],row['image'],row['clip'])==('confirmed','start.jpg','start.mp4')
    assert row['payload']['end_s']==.2 and row['payload']['clip_state']=='ready'
    proof=row['payload']
    assert (proof['start_state_from'],proof['start_state_to'],proof['start_state_confirmation'])==('unknown','violation',{'samples':3})
    assert (proof['state_from'],proof['state_to'],proof['state_confirmation'])==('violation','clear',{'samples':5})
    assert len(proof['state_transitions'])==2 and proof['state_transitions'][0]['confirmation']=={'samples':3}
    assert proof['state_transitions'][1]['confirmation']=={'samples':5}
    # Legacy sequence dedup must also return the actual stored ID.
    legacy={k:v for k,v in event.items() if k!='episode_id'};legacy['sequence']=2
    legacy_id=db.add_event('job','Camera',legacy,'legacy.jpg')
    assert db.add_event('job','Camera',legacy,'ignored.jpg')==legacy_id
    # New episode after a reset may reuse a legacy sequence; it is independent.
    next_id,created=db.add_event('job','Camera',activation('episode-b'),'next.jpg',return_created=True)
    assert created and next_id!=eid and len(db.events('job'))==3


@pytest.mark.parametrize('patch',[{'end_s':float('nan')},{'end_s':float('inf')},{'closed_at_s':.1},
                                  {'end_s':-.1},{'duration_s':float('inf')}])
def test_invalid_closure_cannot_close_an_open_incident(store,patch):
    job();db.add_event('job','Camera',activation(),'start.jpg')
    assert db.update_episode('job',{**closure(),**patch}) is None
    assert db.events('job')[0]['payload']['episode_state']=='open'


def test_episode_updates_are_job_tenant_camera_scoped_and_interruption_is_not_clear(store):
    token=current_scope.set(Scope('org-a'))
    try:job('a');eid=db.add_event('a','Camera',activation(),'start.jpg')
    finally:current_scope.reset(token)
    job('b');db.add_event('b','Camera',activation(),'b.jpg')
    for scope in (Scope('org-b'),Scope('org-a',('Other camera',))):
        token=current_scope.set(scope)
        try:assert db.update_episode('a',closure()) is None
        finally:current_scope.reset(token)
    update={**closure(),'episode_state':'interrupted','closure_reason':'track_lost','observed_until_s':.2}
    assert db.update_episode('a',update)==eid
    a=db.events('a')[0]['payload'];b=db.events('b')[0]['payload']
    assert a['end_s'] is None and 'duration_s' not in a and a['observed_until_s']==.2
    assert b['episode_state']=='open'


def test_server_restart_marks_unfinished_episode_interrupted_without_fabricating_end(store):
    job();eid=db.add_event('job','Camera',activation(),'start.jpg')
    db.review(eid,'confirmed');db.update_job('job',status='running')
    db.init(True)
    row=db.events('job')[0]
    assert db.job('job')['status']=='interrupted'
    assert row['payload']['episode_state']=='interrupted' and row['payload']['end_s'] is None
    assert row['payload']['closure_reason']=='server_restarted' and row['status']=='confirmed'


def test_real_temporal_state_persists_one_interval_across_unknown_and_clear(store):
    from app.temporal_state import PatternStates
    from app.core import Person
    job();states=PatternStates(RunConfig());person=Person((1,1,60,79),.95,track_id=7)
    ids=[];frame=np.full((80,80,3),90,np.uint8)
    for now,condition in [(0.,True),(.2,True),(.4,True),(.6,None),(.8,None),(1.,None),
                          (1.2,False),(1.4,False),(1.6,False),(1.8,False),(2.,False)]:
        states.begin([person],now)
        gate,fired=states.observe(7,'no_helmet','',condition,.4,.6,.3)
        if fired:
            start,sequence=fired
            event={**activation(gate.episode_id),**gate.last_transition,'start_s':start,'alert_s':now,'sequence':sequence}
            ids.append(db.add_event('job','Camera',event,'start.jpg'))
        states.end(now)
        video.persist_episode_updates('job',states.event_updates,frame)
        if .6<=now<=1.:assert db.events('job')[0]['payload']['episode_state']=='open'
    assert len(ids)==1
    row=db.events('job')[0];payload=row['payload']
    assert row['id']==ids[0] and payload['episode_state']=='closed'
    assert payload['end_s']==1.2 and payload['closed_at_s']>=1.8
    assert payload['duration_s']==1.2 and cv2.imread(str(store/payload['end_image'])).mean()==90


@pytest.mark.parametrize('engine_kind',['people','ppe'])
def test_actual_people_and_ppe_rules_contract_closes_same_stored_event(store,engine_kind):
    from app.core import Person,Rules,Zone
    from app.scenarios import ScenarioRules
    from app.scenario_models import Observation
    job();config=RunConfig(fast_motion=False,zone_hold=.4,release_hold=.4,ppe_hold=.4,
        helmet_detection=engine_kind=='ppe',zones=[Zone(name='Zone',points=[(0,0),(.5,0),(.5,1),(0,1)])] if engine_kind=='people' else [])
    rules=Rules(config) if engine_kind=='people' else ScenarioRules(config)
    for now,condition in [(0.,True),(.2,True),(.4,True),(.6,None),(.8,None),
                          (1.,False),(1.2,False),(1.4,False),(1.6,False)]:
        person=Person((10,10,40,90) if condition is not False or engine_kind=='ppe' else (60,10,90,90),
                      .95,track_id=7,observed=condition is not None)
        if engine_kind=='people':events,_=rules.step([person],now,100,100)
        else:
            objects=[] if condition is None else [Observation((12,12,38,28),.95,'no_hardhat' if condition else 'hardhat','ppe-head')]
            events,_=rules.step([person],objects,now,100,100)
        for event in events:db.add_event('job','Camera',event,'start.jpg')
        video.persist_episode_updates('job',rules.event_updates,np.zeros((100,100,3),np.uint8))
        if condition is None:assert db.events('job')[0]['payload']['episode_state']=='open'
    rows=db.events('job');assert len(rows)==1
    payload=rows[0]['payload']
    assert payload['episode_state']=='closed' and payload['end_s']==1.
    assert payload['state_from']=='violation' and payload['state_to']=='clear'
    assert len(payload['state_transitions'])==2 and payload['state_confirmation']['samples']>=3


class FakeRules:
    def __init__(self,*args,**kwargs):
        self.gates={};self.event_updates=[];self.state_rows={};self.steps=0
        self.ppe_details={};self.ppe_status={}
    def step(self,*args):
        self.steps+=1;self.event_updates=[]
        state='violation' if self.steps==1 else 'unknown' if self.steps==2 else 'normal'
        self.state_rows={7:{'helmet':{'state':state,'episode_id':'episode-a'}}}
        if self.steps==3:self.event_updates=[closure()]
        return ([activation()] if self.steps==1 else []),{}
    def finish(self,now,reason):
        if self.steps<3:
            return [dict(episode_id='episode-a',episode_state='interrupted',end_s=None,
                         observed_until_s=now,closure_reason=reason)]
        return []


class EmptyRules(FakeRules):
    def step(self,*args):return [],{}
    def finish(self,*args):return []


class Tracker:
    def __init__(self,*args,**kwargs):pass
    def update(self,*args):return []


@pytest.mark.parametrize('preexisting',[False,True])
@pytest.mark.parametrize('failure',[False,True])
def test_offline_applies_unknown_then_closure_with_same_id_and_end_frame(store,monkeypatch,preexisting,failure):
    job()
    if preexisting:
        start='runs/job/existing-proof.jpg'
        cv2.imwrite(str(store/start),np.full((80,80,3),5,np.uint8))
        saved=db.add_event('job','Camera',{**activation(),'clip_scope':'incident_excerpt'},start)
        db.review(saved,'confirmed')
    class Capture:
        def __init__(self,*args):self.index=0
        def read(self):
            self.index+=1
            return (True,np.full((80,80,3),self.index*20,np.uint8)) if self.index<=3 else (False,None)
        def set(self,*args):self.index=0
        def release(self):pass
    class Writer:
        def __init__(self,*args):pass
        def isOpened(self):return True
        def write(self,*args):pass
        def release(self):pass
    class Backend:
        name='fake';objects_state=[];scene_state=None
        calls=0
        def predict(self,*args):
            self.calls+=1
            if failure and self.calls==4:raise RuntimeError('test inference failure')
            return []
    monkeypatch.setattr(video,'Rules',FakeRules);monkeypatch.setattr(video,'ScenarioRules',EmptyRules)
    monkeypatch.setattr(video,'PeopleTracker',Tracker);monkeypatch.setattr(video,'create_backend',lambda *a,**k:Backend())
    monkeypatch.setattr(video.cv2,'VideoCapture',Capture);monkeypatch.setattr(video.cv2,'VideoWriter',Writer)
    monkeypatch.setattr(video,'draw',lambda f,*a,**k:f.copy());monkeypatch.setattr(video,'annotate_objects',lambda f,*a:f)
    monkeypatch.setattr(video,'assessment_rows',lambda *a:[])
    monkeypatch.setattr(video,'transcode',lambda *a:False);monkeypatch.setattr(video,'ffmpeg_path',lambda:None)
    video.run('job',RunConfig(fast_motion=False),threading.Event())
    rows=db.events('job');assert len(rows)==1
    payload=rows[0]['payload'];stats=db.job('job')['stats']
    if failure:
        assert db.job('job')['status']=='failed'
        assert payload['episode_state']=='interrupted' and payload['end_s'] is None
        assert payload['observed_until_s']==.1 and payload['closure_reason']=='processing_failed'
        assert 'end_image' not in payload
    else:
        assert payload['episode_state']=='closed' and payload['end_s']==.2
        assert cv2.imread(str(store/payload['end_image'])).mean()==60
    assert payload['clip_scope']=='incident_excerpt'
    assert cv2.imread(str(store/rows[0]['image'])).mean()==(5 if preexisting else 20)
    assert stats['events']==(0 if preexisting else 1)
    assert stats['current_states']['7']['helmet']['state']==('unknown' if failure else 'normal')
    if preexisting:assert rows[0]['id']==saved and rows[0]['status']=='confirmed' and rows[0]['image']==start


@pytest.mark.parametrize('separated',[False,True])
@pytest.mark.parametrize('frame_count',[2,3])
def test_live_people_and_separate_scenario_workers_apply_lifecycle_once(store,monkeypatch,separated,frame_count):
    monkeypatch.setattr(live,'Rules',EmptyRules if separated else FakeRules)
    monkeypatch.setattr(live,'ScenarioRules',FakeRules if separated else EmptyRules)
    monkeypatch.setattr(live,'PeopleTracker',Tracker)
    monkeypatch.setattr(live,'draw',lambda f,*a,**k:f.copy());monkeypatch.setattr(live,'annotate_objects',lambda f,*a:f)
    monkeypatch.setattr(live,'assessment_rows',lambda *a:[])
    class Backend:
        name='fake';scene_state=None;objects_state=[]
        def predict(self,frame):return []
    if separated:
        Backend.predict_people=lambda self,frame:[]
        Backend.predict_scenarios=lambda self,frame,people:[]
    monkeypatch.setattr(live,'create_backend',lambda *a,**k:Backend())
    session=live.LiveSession(RunConfig(fast_motion=False),'browser').start()
    # Wait for each source frame to be consumed, not just queued. This prevents
    # latest-only buffering from making a test accidentally skip its unknown.
    try:
        for i in range(frame_count):
            session.ingest(np.full((80,80,3),(i+1)*20,np.uint8))
            deadline=time.monotonic()+3
            while (session.scenario_count if separated else session.processed)<i+1 and time.monotonic()<deadline:time.sleep(.01)
            assert (session.scenario_count if separated else session.processed)>=i+1
            if i==1:
                assert db.events(session.id)[0]['payload']['episode_state']=='open'
            time.sleep(.12)
    finally:
        session.stop_event.set();session.worker.join(5)
    assert not session.worker.is_alive(),session.error
    rows=db.events(session.id);assert len(rows)==1 and session.events==1
    payload=rows[0]['payload']
    if frame_count==3:
        assert payload['episode_state']=='closed'
        assert cv2.imread(str(store/payload['end_image'])).mean()==60
        assert session.report()['current_states'][7]['helmet']['state']=='normal'
    else:
        assert payload['episode_state']=='interrupted' and payload['end_s'] is None
        assert 'end_image' not in payload and payload['closure_reason']=='stream_stopped'


@pytest.mark.parametrize('observed',[True,False])
def test_real_live_main_accepts_one_fps_fresh_measurements_but_never_predicted_votes(store,monkeypatch,observed):
    from app.core import Person,Zone
    class Backend:
        name='one-fps-fixture';scene_state=None;objects_state=[];calls=0
        def predict(self,frame):
            self.calls+=1
            return [Person((10,10,40,90),.95,track_id=7,observed=observed or self.calls!=2)]
    class IdentityTracker:
        def __init__(self,*a,**k):pass
        def update(self,people,*a):return people
    monkeypatch.setattr(live,'create_backend',lambda *a,**k:Backend())
    monkeypatch.setattr(live,'PeopleTracker',IdentityTracker)
    config=RunConfig(fast_motion=False,zone_hold=1.,zones=[Zone(points=[(0,0),(1,0),(1,1),(0,1)])])
    session=live.LiveSession(config,'browser')
    session.first_mono=time.monotonic();session.frame_origin=datetime.now(timezone.utc)
    session.start()
    try:
        for i in range(3):
            session.frames.put((session.first_mono+i,np.zeros((100,100,3),np.uint8)),timeout=2)
            deadline=time.monotonic()+3
            while session.processed<i+1 and time.monotonic()<deadline:time.sleep(.01)
            assert session.processed==i+1
    finally:
        session.stop_event.set();session.worker.join(5)
    assert not session.worker.is_alive() and not session.error
    rows=db.events(session.id)
    assert len(rows)==(1 if observed else 0)
    assert session.report()['effective_people_observation_gap_s']==2.5
    assert config.observation_gap==.8
    if observed:
        assert rows[0]['payload']['start_state_confirmation']['samples']==3
        assert rows[0]['payload']['start_state_confirmation']['window_s']==2.
