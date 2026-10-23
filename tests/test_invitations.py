from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app import auth,storage as db,invitations,mailing,live
from app.server import app

PASSWORD='ValidTestPassword123!'


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3')
    monkeypatch.setattr(live,'SESSIONS',{})
    monkeypatch.setattr(mailing,'settings',lambda:{'mode':'preview','public_url':'http://127.0.0.1:8765'})
    with TestClient(app) as c:
        with db.connection() as conn:
            conn.execute("INSERT INTO users(id,login,name,password_hash,platform_admin,role) VALUES('root','root','Platform',?,1,'enterprise_admin')",(auth.password_hash(PASSWORD),))
        assert c.post('/api/auth/login',json={'login':'root','password':PASSWORD}).status_code==200
        c.headers['X-CSRF-Token']=c.get('/api/auth/me').json()['csrf']
        yield c


def create(client,email='admin@example.org'):
    result=client.post('/api/platform/organizations',json={'name':'Invited factory','email':email,'note':'By platform administrator'})
    assert result.status_code==200,result.text
    info=result.json();message=client.get('/api/platform/mail-preview/'+info['preview_id']).json()
    return info,message['link'].split('#invite=')[1]


def test_invitation_email_role_and_enterprise_are_bound_and_one_use(client):
    info,token=create(client)
    assert info['delivery']=='preview' and info['lifetime_s']==60
    with db.connection() as c:
        row=c.execute('SELECT * FROM admin_invitations WHERE id=?',(info['id'],)).fetchone()
        assert row['token_hash']==hashlib.sha256(token.encode()).hexdigest()
        assert token not in str(dict(row))
    check=client.post('/api/invitations/inspect',json={'token':token})
    assert check.status_code==200 and check.json()['email']=='admin@example.org'
    for tamper in [{'role':'platform_admin'},{'email':'attacker@example.org'},{'organization_id':'legacy'}]:
        assert client.post('/api/invitations/accept',json={'token':token,'name':'Local administrator','password':PASSWORD,**tamper}).status_code==422
    result=client.post('/api/invitations/accept',json={'token':token,'name':'Local administrator','password':PASSWORD})
    assert result.status_code==200,result.text
    user=client.get('/api/auth/me').json()
    assert user['organization_id']==info['organization_id'] and user['role']=='enterprise_admin' and user['platform_admin']==0
    assert user['login']=='admin@example.org' and user['must_change']==0
    client.headers['X-CSRF-Token']=user['csrf']
    assert client.get('/api/members').status_code==200
    assert client.get('/api/platform/organizations').status_code==403
    assert client.get('/api/platform/mail-preview/'+info['preview_id']).status_code==403
    assert client.post('/api/platform/organizations',json={'name':'Forbidden factory','email':'other@example.org'}).status_code==403
    assert client.post('/api/invitations/accept',json={'token':token,'name':'Replay','password':PASSWORD}).status_code==410
    assert client.post('/api/invitations/inspect',json={'token':token}).status_code==410
    with db.connection() as c:
        assert c.execute('SELECT count(*) FROM users WHERE organization_id=?',(info['organization_id'],)).fetchone()[0]==1
        assert token not in ' '.join(r['details'] for r in c.execute('SELECT details FROM audit'))


def test_resend_changes_token_and_revokes_previous_link(client):
    info,old=create(client)
    response=client.post(f"/api/platform/organizations/{info['organization_id']}/invitations",json={'email':'admin@example.org'})
    assert response.status_code==200
    new=client.get('/api/platform/mail-preview/'+response.json()['preview_id']).json()['link'].split('#invite=')[1]
    assert old!=new
    assert client.post('/api/invitations/inspect',json={'token':old}).status_code==410
    assert client.post('/api/invitations/inspect',json={'token':new}).status_code==200
    rows=client.get(f"/api/platform/organizations/{info['organization_id']}/invitations").json()
    assert {r['status'] for r in rows}=={'sent','revoked'}


def test_expiration_at_exact_deadline_and_during_password_hash(client,monkeypatch):
    clock=[invitations.time.time()]
    monkeypatch.setattr(invitations.time,'time',lambda:clock[0])
    info,token=create(client)
    clock[0]+=59.9
    assert client.post('/api/invitations/inspect',json={'token':token}).status_code==200
    original=auth.password_hash
    def hash_after_expiration(value):
        result=original(value);clock[0]+=.1;return result
    monkeypatch.setattr(auth,'password_hash',hash_after_expiration)
    assert client.post('/api/invitations/accept',json={'token':token,'name':'Expired administrator','password':PASSWORD}).status_code==410
    assert client.post('/api/invitations/inspect',json={'token':token}).status_code==410
    with db.connection() as c:assert c.execute('SELECT count(*) FROM users WHERE organization_id=?',(info['organization_id'],)).fetchone()[0]==0


def test_concurrent_registration_consumes_invitation_only_once(client):
    info,token=create(client)
    def accept(i):
        with TestClient(app) as visitor:
            return visitor.post('/api/invitations/accept',json={'token':token,'name':'Concurrent '+str(i),'password':PASSWORD}).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:statuses=list(pool.map(accept,[1,2]))
    assert sorted(statuses)==[200,410]
    with db.connection() as c:assert c.execute('SELECT count(*) FROM users WHERE organization_id=?',(info['organization_id'],)).fetchone()[0]==1


def test_unconfigured_smtp_creates_no_enterprise_and_preview_is_private(client,monkeypatch):
    monkeypatch.setattr(mailing,'settings',lambda:{'mode':'smtp'})
    before=len(client.get('/api/platform/organizations').json())
    result=client.post('/api/platform/organizations',json={'name':'No SMTP','email':'new@example.org'})
    assert result.status_code==503
    assert len(client.get('/api/platform/organizations').json())==before
    client.cookies.clear()
    assert client.get('/api/platform/mail-preview/'+'a'*32).status_code==401


def test_smtp_failure_revokes_link_and_suspended_org_cannot_be_joined(client,monkeypatch):
    real=mailing.Delivery.send
    def failure(*args):raise HTTPException(502,'Simulated SMTP failure')
    monkeypatch.setattr(mailing.Delivery,'send',failure)
    result=client.post('/api/platform/organizations',json={'name':'Delivery failure','email':'new@example.org'})
    assert result.status_code==502
    with db.connection() as c:assert c.execute('SELECT status FROM admin_invitations').fetchone()['status']=='failed'
    monkeypatch.setattr(mailing.Delivery,'send',real)
    info,token=create(client)
    assert client.patch('/api/platform/organizations/'+info['organization_id'],json={'status':'suspended','reason':'Pause'}).status_code==200
    assert client.post('/api/invitations/inspect',json={'token':token}).status_code==410


def test_existing_email_is_not_promoted_to_new_enterprise(client):
    response=client.post('/api/platform/organizations',json={'name':'Conflict','email':'reserved@example.org'})
    assert response.status_code==200
    # A pre-existing account must never be granted a different role by invitation.
    with db.connection() as c:
        c.execute("INSERT INTO users(id,login,name,password_hash,role,organization_id) VALUES('reserved','reserved@example.org','Worker',?,'worker','legacy')",(auth.password_hash(PASSWORD),))
    old_count=len(client.get('/api/platform/organizations').json())
    response=client.post('/api/platform/organizations',json={'name':'Another factory','email':'reserved@example.org'})
    assert response.status_code==409
    assert len(client.get('/api/platform/organizations').json())==old_count


def test_smtp_transport_uses_tls_and_correct_recipient_without_network(tmp_path,monkeypatch):
    calls={}
    class FakeSMTP:
        def __init__(self,*args,**kwargs):calls['connection']=(args,kwargs)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def ehlo(self):pass
        def starttls(self,context):calls['tls']=context
        def login(self,username,password):calls['login']=(username,password)
        def send_message(self,message):calls['message']=message;return {}
    monkeypatch.setattr(mailing.smtplib,'SMTP',FakeSMTP)
    config={'mode':'smtp','host':'smtp.example.org','port':587,'security':'starttls','username':'sender@example.org','password':'smtp-test-secret','from_address':'sender@example.org','public_url':'https://safety.example.org'}
    monkeypatch.setattr(mailing,'settings',lambda:config)
    result=mailing.prepare().send('recipient@example.org','Enterprise','https://safety.example.org/#invite=test','2026-10-23T12:00:00Z')
    assert result=={'delivery':'smtp'} and calls['tls']
    assert calls['message']['To']=='recipient@example.org'
    assert 'smtp-test-secret' not in calls['message'].get_content()
    assert 'https://safety.example.org/#invite=test' in calls['message'].get_content()
