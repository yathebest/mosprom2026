"""Real authorization/storage and synthetic capture; never constructs a NN."""
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import auth,live,storage as db
from app.access import Scope,current_scope
from app.core import RunConfig
from test_access import system,sign_in,enterprise,activate,PASS


def test_create_is_atomic_duplicate_names_are_tenant_scoped_and_global_capacity_is_bounded(system,monkeypatch):
    client=system;organization=enterprise(client,'wall-admin');activate(client,organization);sign_in(client,'wall-admin')
    monkeypatch.setattr(live.LiveSession,'start',lambda self:self)
    payload={'source_type':'network','url':'http://127.0.0.1:8766/cam-a.mjpg','config':{'camera':'Line A','fast_motion':True}}
    first=client.post('/api/live',json=payload);assert first.status_code==200,first.text
    assert client.post('/api/live',json={**payload,'url':'http://127.0.0.1:8766/other.mjpg'}).status_code==409
    assert client.post('/api/live',json={**payload,'config':{'camera':'Line B'}}).status_code==409
    second=client.post('/api/live',json={**payload,'url':'http://127.0.0.1:8766/cam-b.mjpg','config':{'camera':'Line B'}});assert second.status_code==200
    full=client.post('/api/live',json={**payload,'url':'http://127.0.0.1:8766/cam-c.mjpg','config':{'camera':'Line C'}})
    assert full.status_code==409 and 'оборудования' in full.text
    a=live.SESSIONS[first.json()['id']];b=live.SESSIONS[second.json()['id']]
    assert client.post('/api/live/'+a.id+'/stop').status_code==200
    assert a.state=='stopping' and a.stop_event.is_set() and not b.stop_event.is_set()


def test_camera_wall_events_snapshots_and_mutations_follow_operator_scope_and_tenant(system,monkeypatch):
    client=system;organization=enterprise(client,'scoped-wall');activate(client,organization);sign_in(client,'scoped-wall')
    monkeypatch.setattr(live.LiveSession,'start',lambda self:self)
    sessions=[]
    for camera in ['Allowed','Hidden']:
        response=client.post('/api/live',json={'source_type':'browser','config':{'camera':camera}});assert response.status_code==200
        session=live.SESSIONS[response.json()['id']];session.jpeg=b'jpeg-'+camera.encode();session.version=1;sessions.append(session)
        event={'track_id':1,'kind':'danger_zone','zone':'z','sequence':1,'start_s':0,'alert_s':1,'live':True}
        token=current_scope.set(Scope(organization))
        try:db.add_event(session.id,camera,event,'fixture.jpg')
        finally:current_scope.reset(token)
    with db.connection() as connection:
        connection.execute("INSERT INTO users(id,login,name,password_hash,organization_id,role,camera_scope) VALUES('viewer','wall-viewer','Viewer',?,?,'operator',?)",(auth.password_hash(PASS),organization,'["Allowed"]'))
    sign_in(client,'wall-viewer');allowed,hidden=sessions
    assert [row['id'] for row in client.get('/api/live').json()]==[allowed.id]
    assert {row['camera'] for row in client.get('/api/live-events').json()}=={'Allowed'}
    assert client.get('/api/live/'+allowed.id+'/snapshot').content==b'jpeg-Allowed'
    assert client.get('/api/live/'+hidden.id+'/snapshot').status_code==404
    assert client.get('/api/live/'+hidden.id).status_code==404
    assert client.post('/api/live/'+allowed.id+'/stop').status_code==403
    assert client.post('/api/live/'+allowed.id+'/frame',content=b'jpeg').status_code==403
    assert client.post('/api/live',json={'source_type':'browser'}).status_code==403
    other=enterprise(client,'other-wall');activate(client,other);sign_in(client,'other-wall')
    assert client.get('/api/live').json()==[] and client.get('/api/live-events').json()==[]
    assert client.get('/api/live/'+allowed.id+'/snapshot').status_code==404


def test_concurrent_creation_cannot_overbook_or_duplicate_same_camera(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3');db.init()
    monkeypatch.setattr(live,'SESSIONS',{});monkeypatch.setattr(live.LiveSession,'start',lambda self:self)
    def create(name):
        try:return live.create(RunConfig(camera=name),'browser',None).id
        except ValueError:return None
    with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(create,['A','A','B','C','D','E']))
    assert len([result for result in results if result])==2
    assert len({session.config.camera for session in live.SESSIONS.values()})==2


def test_ingest_after_stop_does_not_accept_new_frames(tmp_path,monkeypatch):
    import numpy as np
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'add_job',lambda *a,**k:None);monkeypatch.setattr(db,'update_job',lambda *a,**k:None)
    session=live.LiveSession(RunConfig(),'browser');frame=np.zeros((32,32,3),np.uint8)
    assert session.ingest(frame) is True;assert session.capture_count==1
    session.stop();assert session.ingest(frame) is False;assert session.capture_count==1


def test_scoped_manager_cannot_create_unassigned_camera_but_other_tenant_can_reuse_name(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3');db.init()
    monkeypatch.setattr(live,'SESSIONS',{});monkeypatch.setattr(live.LiveSession,'start',lambda self:self)
    token=current_scope.set(Scope('one',('Assigned',)))
    try:
        with pytest.raises(PermissionError):live.create(RunConfig(camera='Hidden'),'browser',None)
        first=live.create(RunConfig(camera='Assigned'),'browser',None)
    finally:current_scope.reset(token)
    token=current_scope.set(Scope('two'))
    try:second=live.create(RunConfig(camera='Assigned'),'browser',None)
    finally:current_scope.reset(token)
    assert first.id!=second.id and first.organization_id!=second.organization_id


def test_ui_staging_has_new_db_and_reuse_does_not_reset_passwords_or_touch_production(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/create_camera_wall_staging.py'
    spec=importlib.util.spec_from_file_location('stage_fixture_test',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    production=tmp_path/'data';(production/'models').mkdir(parents=True);(production/'models/fake.pt').write_bytes(b'weights-read-only')
    original=production/'events.sqlite3';original.write_bytes(b'not copied or opened')
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(db,'DATA',production);monkeypatch.setattr(db,'DB',original)
    stage=production/'cache/qa';fixture=module.prepare(stage);again=module.prepare(stage)
    assert fixture==again and original.read_bytes()==b'not copied or opened'
    assert (stage/'models/fake.pt').read_bytes()==b'weights-read-only'
    assert fixture['production_database_copied'] is False
    with db.connection() as connection:
        users=connection.execute('SELECT role,camera_scope,must_change FROM users ORDER BY role').fetchall()
    assert len(users)==2 and all(row['must_change']==0 for row in users)
    operator=next(row for row in users if row['role']=='operator')
    import json
    assert json.loads(operator['camera_scope'])==['QA Камера A','QA Камера B']
