import io
import json
import zipfile
import pytest
from fastapi.testclient import TestClient
from app import auth,storage as db,live
from app.server import app
from app.access import Scope,current_scope
from app.core import RunConfig

PASS='StrongTestPassword!42'


@pytest.fixture
def system(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3')
    monkeypatch.setattr(live,'SESSIONS',{})
    with TestClient(app) as client:
        with db.connection() as c:
            c.execute("INSERT INTO users(id,login,name,password_hash,platform_admin,role) VALUES('root','root','Platform',?,1,'enterprise_admin')",(auth.password_hash(PASS),))
        yield client


def sign_in(client,login):
    client.cookies.clear();client.headers.pop('X-CSRF-Token',None)
    result=client.post('/api/auth/login',json={'login':login,'password':PASS})
    assert result.status_code==200,result.text
    user=client.get('/api/auth/me').json();client.headers['X-CSRF-Token']=user['csrf'];return user


def enterprise(client,name):
    client.cookies.clear();client.headers.pop('X-CSRF-Token',None)
    r=client.post('/api/auth/register',json={'login':name,'name':name,'password':PASS,'enterprise':'Enterprise '+name})
    assert r.status_code==200,r.text
    user=client.get('/api/auth/me').json();client.headers['X-CSRF-Token']=user['csrf'];return user['organization_id']


def activate(client,org):
    sign_in(client,'root');assert client.patch('/api/platform/organizations/'+org,json={'status':'active'}).status_code==200


def seed(org,key,camera):
    token=current_scope.set(Scope(org))
    try:
        db.save_camera({'camera':camera});db.add_job(key,'fixture.mp4',{})
        db.update_job(key,config={'camera':camera})
        folder=db.DATA/'runs'/key;folder.mkdir(parents=True);(folder/'evidence.jpg').write_bytes(b'evidence')
        eid=db.add_event(key,camera,{'track_id':1,'kind':'danger_zone','zone':'z','start_s':0,'alert_s':1,'sequence':1},f'runs/{key}/evidence.jpg')
        db.review(eid,'confirmed');return eid
    finally:current_scope.reset(token)


def test_enrollment_approval_rejection_and_csrf(system):
    c=system
    assert c.get('/api/events').status_code==401
    org=enterprise(c,'enterprise-a')
    assert c.get('/api/events').status_code==403
    assert c.get('/api/platform/organizations').status_code==403
    sign_in(c,'root');csrf=c.headers.pop('X-CSRF-Token')
    assert c.patch('/api/platform/organizations/'+org,json={'status':'active'}).status_code==403
    c.headers['X-CSRF-Token']=csrf
    assert c.patch('/api/platform/organizations/'+org,json={'status':'rejected','reason':''}).status_code==422
    assert c.patch('/api/platform/organizations/'+org,json={'status':'rejected','reason':'Please clarify contact'}).status_code==200
    sign_in(c,'enterprise-a')
    assert c.get('/api/auth/me').json()['organization']['reason']=='Please clarify contact'
    assert c.post('/api/account/reapply').status_code==200
    activate(c,org);sign_in(c,'enterprise-a')
    assert c.get('/api/events').status_code==200
    assert c.post('/api/cameras',json={'camera':'x'},headers={'origin':'https://evil.example'}).status_code==403


def test_enterprise_isolation_media_live_exports_and_camera_names(system):
    c=system;a=enterprise(c,'enterprise-a');activate(c,a);b=enterprise(c,'enterprise-b');activate(c,b)
    ea=seed(a,'job-a','same camera');eb=seed(b,'job-b','same camera')
    token=current_scope.set(Scope(b))
    try:s=live.LiveSession(RunConfig(camera='same camera'),'browser');live.SESSIONS[s.id]=s
    finally:current_scope.reset(token)
    sign_in(c,'enterprise-a')
    assert [j['id'] for j in c.get('/api/jobs').json()]==['job-a']
    assert [e['id'] for e in c.get('/api/events').json()]==[ea]
    assert c.get('/api/jobs/job-b').status_code==404
    assert c.get('/media/runs/job-b/evidence.jpg').status_code==404
    assert c.get('/media/runs/job-a/evidence.jpg').status_code==200
    assert c.post('/api/jobs/job-b/cancel').status_code==404
    assert c.post('/api/jobs/job-b/clone').status_code==404
    assert c.patch('/api/events/'+eb,json={'status':'false'}).status_code==404
    assert c.get('/api/live').json()==[]
    assert c.get('/api/live/'+s.id).status_code==404
    assert c.post('/api/live/'+s.id+'/stop').status_code==404
    assert eb not in c.get('/api/export').text
    with zipfile.ZipFile(io.BytesIO(c.get('/api/review-dataset').content)) as z:
        assert [e['id'] for e in json.loads(z.read('annotations.json'))]==[ea]
    assert len(c.get('/api/cameras').json())==1
    assert c.post('/api/cameras',json={'camera':'same camera','fall_detection':True}).status_code==200
    sign_in(c,'enterprise-b');assert c.get('/api/cameras').json()[0]['config']=={'camera':'same camera'}


def test_local_roles_scopes_revocation_and_last_admin(system):
    c=system;org=enterprise(c,'enterprise-a');activate(c,org);first=seed(org,'one','Line 1');seed(org,'two','Line 2');user=sign_in(c,'enterprise-a')
    def add(login,role,cameras=None):
        r=c.post('/api/members',json={'login':login,'name':login,'password':PASS,'role':role,'cameras':cameras});assert r.status_code==200,r.text
        # Test role behavior independently from mandatory temporary-password flow (covered below).
        with db.connection() as conn:conn.execute('UPDATE users SET must_change=0 WHERE id=?',(r.json()['id'],))
        return r.json()['id']
    foreman=add('foreman','foreman',['Line 1']);operator=add('operator','operator');add('manager','management');add('worker','worker');add('legal','safety_legal')
    bad=c.post('/api/members',json={'login':'bad','name':'Bad','password':PASS,'role':'platform_admin'})
    assert bad.status_code==422
    assert c.patch('/api/members/'+user['id'],json={'name':'Admin','role':'operator','active':True}).status_code==409
    sign_in(c,'foreman')
    assert [e['id'] for e in c.get('/api/events').json()]==[first]
    assert c.get('/media/runs/two/evidence.jpg').status_code==404
    assert c.get('/api/jobs/two').status_code==404
    assert c.post('/api/cameras',json={'camera':'Line 1'}).status_code==403
    assert c.get('/api/members').status_code==403
    assert c.patch('/api/events/'+first,json={'status':'forwarded'}).status_code==200
    sign_in(c,'manager');assert c.get('/api/summary').json()['events']==2
    for url in ['/api/events','/api/jobs','/api/live','/media/runs/one/evidence.jpg','/api/export']:
        assert c.get(url).status_code==403
    sign_in(c,'worker');assert c.get('/api/auth/me').status_code==200;assert c.get('/api/events').status_code==403
    sign_in(c,'legal');assert c.get('/api/events').status_code==200;assert c.get('/api/export').status_code==200
    assert c.patch('/api/events/'+first,json={'status':'false'}).status_code==403
    sign_in(c,'operator');stolen=dict(c.cookies)
    assert c.get('/api/live').status_code==200;assert c.post('/api/live',json={'source_type':'browser'}).status_code==403
    sign_in(c,'enterprise-a')
    assert c.patch('/api/members/'+operator,json={'name':'Operator','role':'operator','active':False}).status_code==200
    c.cookies.clear();c.cookies.update(stolen);assert c.get('/api/auth/me').status_code==401


def test_temporary_password_and_logout(system):
    c=system;org=enterprise(c,'enterprise-a');activate(c,org);sign_in(c,'enterprise-a')
    r=c.post('/api/members',json={'login':'temp-user','name':'New operator','role':'operator','password':PASS})
    assert r.status_code==200
    user=sign_in(c,'temp-user');assert user['must_change']==1
    assert c.get('/api/events').status_code==403
    assert c.post('/api/auth/password',json={'current':'wrong','password':'NewPassword12345!'}).status_code==400
    assert c.post('/api/auth/password',json={'current':PASS,'password':'NewPassword12345!'}).status_code==200
    c.headers['X-CSRF-Token']=c.get('/api/auth/me').json()['csrf']
    assert c.get('/api/events').status_code==200
    assert c.post('/api/auth/logout').status_code==200;assert c.get('/api/events').status_code==401


def test_platform_context_and_suspend(system):
    c=system;org=enterprise(c,'enterprise-a');activate(c,org);seed(org,'one','Line 1')
    assert c.get('/api/events').status_code==403
    assert c.post('/api/platform/context',json={'organization_id':org}).status_code==200
    assert len(c.get('/api/events').json())==1
    assert c.get('/api/audit').status_code==200
    assert c.patch('/api/platform/organizations/'+org,json={'status':'suspended','reason':'Access paused'}).status_code==200
    sign_in(c,'enterprise-a');assert c.get('/api/events').status_code==403
    assert c.get('/api/auth/me').json()['organization']['status']=='suspended'
