import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app import storage as db
from app.server import app
from app import auth
from app.access import Scope,current_scope


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path)
    monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3')
    with TestClient(app) as client:
        with db.connection() as c:
            c.execute("INSERT INTO users(id,login,name,password_hash,organization_id,role) VALUES('test-admin','test-admin','Test administrator',?,'legacy','enterprise_admin')",(auth.password_hash('TestPassword123!'),))
        assert client.post('/api/auth/login',json={'login':'test-admin','password':'TestPassword123!'}).status_code==200
        client.headers['X-CSRF-Token']=client.get('/api/auth/me').json()['csrf']
        token=current_scope.set(Scope('legacy'))
        try:yield client
        finally:current_scope.reset(token)


def test_reject_corrupt_video_and_bad_extension(client):
    r=client.post('/api/jobs',files={'file':('x.mp4',b'not a video','video/mp4')})
    assert r.status_code==400
    assert client.get('/api/jobs').json()==[]
    assert client.post('/api/jobs',files={'file':('x.exe',b'x')}).status_code==400


def test_valid_upload_and_preview(client,tmp_path):
    path=tmp_path/'fixture.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120))
    for _ in range(20):
        writer.write(np.zeros((120,160,3),np.uint8))
    writer.release()
    r=client.post('/api/jobs',files={'file':('camera.mp4',path.read_bytes(),'video/mp4')})
    assert r.status_code==200,r.text
    item=r.json()
    assert item['meta']['frames']==20
    assert client.get(f"/media/runs/{item['id']}/preview.jpg").status_code==200
    assert client.get(f"/media/runs/{item['id']}/source.video").status_code==404


def test_event_persistence_review_filters_and_csv(client):
    db.add_job('test','test.mp4',{})
    event={'track_id':1,'kind':'danger_zone','zone':'z','start_s':0,'alert_s':1,'sequence':1}
    eid=db.add_event('test','=camera',event,'runs/test/evidence.jpg')
    db.add_event('test','=camera',event,'runs/test/evidence.jpg')
    assert len(client.get('/api/events').json())==1
    assert client.patch(f'/api/events/{eid}',json={'status':'confirmed'}).status_code==200
    assert client.get('/api/events?status=new').json()==[]
    assert len(client.get('/api/events?camera=%3Dcamera&start=0.5&end=2').json())==1
    assert "'=camera" in client.get('/api/export').text
    assert client.patch(f'/api/events/{eid}',json={'status':'invalid'}).status_code==422


def test_private_files_are_not_served(client):
    assert client.get('/media/events.sqlite3').status_code==404
    assert client.get('/media/%2e%2e/requirements.txt').status_code==404


def test_restart_marks_incomplete_job(client):
    db.add_job('test','test.mp4',{})
    db.update_job('test',status='running')
    db.init()
    assert db.job('test')['status']=='interrupted'


def test_camera_templates_summary_and_job_report(client):
    configuration={'camera':'Склад 2','fast_motion':False,'fall_detection':True,
                   'zones':[{'name':'Проход','points':[[0,0],[1,0],[1,1],[0,1]]}]}
    assert client.post('/api/cameras',json=configuration).status_code==200
    template=client.get('/api/cameras').json()[0]
    assert template['name']=='Склад 2'
    assert template['config']['zones'][0]['name']=='Проход'
    db.add_job('report','x.mp4',{})
    db.add_event('report','Склад 2',{'track_id':1,'kind':'danger_zone','zone':'Проход',
                 'start_s':0,'alert_s':1,'sequence':1},'runs/report/evidence.jpg')
    assert client.get('/api/summary').json()['by_camera']=={'Склад 2':1}
    report=client.get('/api/jobs/report/report').json()
    assert report['job']['id']=='report' and len(report['events'])==1


def test_start_is_single_use_and_cancel_signals_worker(client,monkeypatch):
    from app import server,video
    calls=[]
    monkeypatch.setattr(server.EXECUTOR,'submit',lambda *args:calls.append(args))
    monkeypatch.setattr(video,'CANCEL',{})
    db.add_job('single','x.mp4',{})
    body={'fast_motion':False,'fall_detection':True}
    assert client.post('/api/jobs/single/start',json=body).status_code==200
    assert client.post('/api/jobs/single/start',json=body).status_code==409
    assert len(calls)==1 and not video.CANCEL['single'].is_set()
    assert client.post('/api/jobs/single/cancel').status_code==200
    assert video.CANCEL['single'].is_set()


def test_unknown_demo_cannot_access_files(client):
    assert client.post('/api/samples/other/import').status_code==404


def test_browser_live_ingest_keeps_latest_and_rejects_other_source(client,monkeypatch):
    from app import live
    from app.core import RunConfig
    session=live.LiveSession(RunConfig(fast_motion=False,fall_detection=True),'browser')
    session.state='running'
    monkeypatch.setattr(live,'SESSIONS',{session.id:session})
    first=cv2.imencode('.jpg',np.zeros((120,160,3),np.uint8))[1].tobytes()
    second=cv2.imencode('.jpg',np.full((120,160,3),200,np.uint8))[1].tobytes()
    url=f'/api/live/{session.id}/frame'
    assert client.post(url,content=first,headers={'content-type':'image/jpeg'}).status_code==200
    assert client.post(url,content=second,headers={'content-type':'image/jpeg'}).status_code==200
    assert session.frames.qsize()==1 and session.dropped==1
    assert session.frames.get()[1].mean()>190
    # Evidence preserves the already validated source JPEG byte-for-byte.
    assert session.record_frames.get()[2]==second
    assert client.post(url,content=b'not an image').status_code==400
    session.source_type='network'
    assert client.post(url,content=first).status_code==409
    assert client.post(f'/api/live/{session.id}/stop').status_code==200
    assert session.stop_event.is_set()


def test_live_url_not_in_public_session_or_job(client,monkeypatch):
    from app import live
    from app.core import RunConfig
    session=live.LiveSession(RunConfig(),'network','rtsp://user:secret@camera/stream')
    monkeypatch.setattr(live,'SESSIONS',{session.id:session})
    assert 'secret' not in client.get('/api/live').text
    assert 'secret' not in client.get('/api/jobs').text
    assert client.post('/api/live',json={'source_type':'network','url':'file:///private.mp4'}).status_code==422


def test_payload_clip_completion_preserves_other_event_fields(client):
    db.add_job('clip','x.mp4',{})
    eid=db.add_event('clip','camera',{'track_id':1,'kind':'possible_fall','zone':'',
        'start_s':1,'alert_s':2,'sequence':1,'clip_state':'capturing','occurred_at':'2026-10-23T15:00:00Z'},'runs/clip/frame.jpg')
    db.patch_payload(eid,{'clip_state':'ready','clip_alert_offset_s':3})
    event=client.get('/api/events').json()[0]
    assert event['payload']['clip_state']=='ready' and event['payload']['occurred_at']=='2026-10-23T15:00:00Z'


def test_scenario_preflight_requires_height_context_and_installed_weights(client):
    missing=client.post('/api/live',json={'source_type':'browser','config':{'helmet_detection':True}})
    assert missing.status_code==409 and 'ppe-safetyvision' in missing.json()['detail']
    context=client.post('/api/live',json={'source_type':'browser','config':{'harness_detection':True}})
    assert context.status_code==422 and 'высоте' in context.json()['detail']
    assert not any(s['ready'] for s in client.get('/api/scenarios').json()['scenarios'])


def test_typed_zone_and_scenario_camera_template_roundtrip(client):
    body={'camera':'Высота','harness_detection':True,'phone_detection':True,
          'zones':[{'name':'Площадка','kind':'height','points':[[0,0],[1,0],[1,1],[0,1]]}]}
    assert client.post('/api/cameras',json=body).status_code==200
    config=client.get('/api/cameras').json()[0]['config']
    assert config['harness_detection'] and config['phone_detection']
    assert config['zones'][0]['kind']=='height'
