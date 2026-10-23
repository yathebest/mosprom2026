"""Email-bound, single-use, 60-second enterprise administrator invitations."""
import hashlib
import json
import re
import secrets
import sqlite3
import time
import uuid
from datetime import datetime,timezone

from fastapi import APIRouter,HTTPException,Request
from fastapi.responses import JSONResponse
from pydantic import Field
from . import auth,mailing,storage as db

router=APIRouter()
LIFETIME_SECONDS=60


class Enterprise(auth.StrictModel):
    name: str=Field(min_length=2,max_length=180)
    email: str=Field(min_length=5,max_length=120)
    note: str=Field(default='',max_length=2000)


class Invitation(auth.StrictModel):
    email: str=Field(min_length=5,max_length=120)


class Token(auth.StrictModel):
    token: str=Field(min_length=40,max_length=100)


class Accept(Token):
    name: str=Field(min_length=2,max_length=120)
    password: str=Field(min_length=12,max_length=256)


def timestamp(value):return datetime.fromtimestamp(value,timezone.utc).isoformat()


def issue(user,org_id,email,delivery,new_enterprise=None):
    key=uuid.uuid4().hex;raw=secrets.token_urlsafe(32);expires=time.time()+LIFETIME_SECONDS
    with db.connection() as c:
        c.execute('BEGIN IMMEDIATE')
        if c.execute('SELECT 1 FROM users WHERE login=?',(email,)).fetchone():raise HTTPException(409,'Этот email уже используется аккаунтом. Укажите другой адрес; существующие аккаунты не переносятся автоматически.')
        if new_enterprise:
            c.execute("INSERT INTO organizations(id,name,status,note) VALUES(?,?,'active',?)",(org_id,new_enterprise.name.strip(),new_enterprise.note))
            auth.audit(user,'organization.created',org_id,{'name':new_enterprise.name.strip()},org_id,c)
        org=c.execute("SELECT * FROM organizations WHERE id=? AND status='active'",(org_id,)).fetchone()
        if not org:raise HTTPException(409,'Приглашать администратора можно только в активное предприятие')
        c.execute("UPDATE admin_invitations SET status='revoked' WHERE organization_id=? AND email=? AND status IN ('sending','sent')",(org_id,email))
        c.execute("INSERT INTO admin_invitations(id,organization_id,email,token_hash,expires,status,created_by) VALUES(?,?,?,?,?,'sending',?)",(key,org_id,email,hashlib.sha256(raw.encode()).hexdigest(),expires,user['id']))
        auth.audit(user,'invitation.created',key,{'email':email,'expires_at':timestamp(expires)},org_id,c)
    link=delivery.public_url+'/#invite='+raw
    try:result=delivery.send(email,org['name'],link,timestamp(expires))
    except HTTPException as exc:
        with db.connection() as c:
            c.execute("UPDATE admin_invitations SET status='failed' WHERE id=? AND status='sending'",(key,))
            auth.audit(user,'invitation.failed',key,organization_id=org_id,conn=c)
        if new_enterprise:
            exc.detail+=' Предприятие уже создано; отправьте новое приглашение из его карточки.'
        raise
    with db.connection() as c:
        updated=c.execute("UPDATE admin_invitations SET status='sent' WHERE id=? AND status='sending'",(key,))
        auth.audit(user,'invitation.sent',key,{'delivery':result['delivery']},org_id,c)
    if not updated.rowcount:raise HTTPException(409,'Приглашение уже заменено более новым. Используйте последнее письмо.')
    if expires<=time.time():raise HTTPException(409,'Письмо отправлено, но срок ссылки уже истёк. Отправьте новое приглашение.')
    return {'id':key,'organization_id':org_id,'email':email,'expires_at':timestamp(expires),
            'remaining_s':max(0,expires-time.time()),'lifetime_s':LIFETIME_SECONDS,**result}


@router.get('/api/platform/mail-status')
def mail_status(request: Request):
    auth.require(request,platform=True);return mailing.public_status()


@router.post('/api/platform/organizations')
def create_enterprise(body: Enterprise,request: Request):
    user=auth.require(request,platform=True)
    auth.rate_limit('invite-admin:'+user['id'],30)
    if len(body.name.strip())<2:raise HTTPException(422,'Укажите название предприятия')
    delivery=mailing.prepare();email=mailing.email_address(body.email)
    return issue(user,uuid.uuid4().hex,email,delivery,new_enterprise=body)


@router.post('/api/platform/organizations/{org_id}/invitations')
def invite(org_id: str,body: Invitation,request: Request):
    user=auth.require(request,platform=True);auth.rate_limit('invite-admin:'+user['id'],30)
    return issue(user,org_id,mailing.email_address(body.email),mailing.prepare())


@router.get('/api/platform/organizations/{org_id}/invitations')
def list_invitations(org_id: str,request: Request):
    auth.require(request,platform=True)
    with db.connection() as c:
        rows=[dict(row) for row in c.execute('SELECT id,email,status,expires,created,consumed_by FROM admin_invitations WHERE organization_id=? ORDER BY created DESC,rowid DESC LIMIT 50',(org_id,))]
    for row in rows:
        if row['status'] in ('sent','sending') and row['expires']<=time.time():row['status']='expired'
        row['expires_at']=timestamp(row.pop('expires'))
    return rows


@router.get('/api/platform/mail-preview/{message_id}')
def preview(message_id: str,request: Request):
    auth.require(request,platform=True)
    if mailing.settings().get('mode','smtp')!='preview' or not re.fullmatch('[0-9a-f]{32}',message_id):raise HTTPException(404,'Письмо не найдено')
    path=db.DATA/'mail-preview'/(message_id+'.json')
    if not path.is_file():raise HTTPException(404,'Письмо не найдено')
    return json.loads(path.read_text(encoding='utf-8'))


def find_invitation(c,raw):
    row=c.execute('''SELECT admin_invitations.*,organizations.name AS enterprise_name,organizations.status AS organization_status
        FROM admin_invitations JOIN organizations ON organizations.id=admin_invitations.organization_id
        WHERE token_hash=?''',(hashlib.sha256(raw.encode()).hexdigest(),)).fetchone()
    if not row or row['status']!='sent' or row['expires']<=time.time() or row['organization_status']!='active':
        raise HTTPException(410,'Приглашение недействительно: срок истёк, ссылка уже использована или заменена. Попросите администратора платформы отправить новую.')
    return row


@router.post('/api/invitations/inspect')
def inspect_invitation(body: Token,request: Request):
    auth.rate_limit('invite-inspect:'+request.client.host,100)
    with db.connection() as c:row=find_invitation(c,body.token)
    return {'email':row['email'],'enterprise':row['enterprise_name'],'role':'enterprise_admin',
            'expires_at':timestamp(row['expires']),'remaining_s':max(0,row['expires']-time.time())}


@router.post('/api/invitations/accept')
def accept(body: Accept,request: Request):
    auth.rate_limit('invite-accept:'+request.client.host,15)
    if len(body.name.strip())<2:raise HTTPException(422,'Укажите имя')
    # Check twice: password hashing must not extend a token that expires in flight.
    with db.connection() as c:find_invitation(c,body.token)
    encoded=auth.password_hash(body.password);key=uuid.uuid4().hex
    try:
        with db.connection() as c:
            c.execute('BEGIN IMMEDIATE');row=find_invitation(c,body.token)
            c.execute("INSERT INTO users(id,login,name,password_hash,organization_id,role) VALUES(?,?,?,?,?,'enterprise_admin')",(key,row['email'],body.name.strip(),encoded,row['organization_id']))
            c.execute("UPDATE admin_invitations SET status='consumed',consumed_by=? WHERE id=?",(key,row['id']))
            # All other outstanding grants for this email become unusable as well.
            c.execute("UPDATE admin_invitations SET status='revoked' WHERE email=? AND id<>? AND status IN ('sending','sent')",(row['email'],row['id']))
            auth.audit({'id':key,'name':body.name.strip(),'organization_id':row['organization_id']},'invitation.accepted',row['id'],conn=c)
    except sqlite3.IntegrityError:raise HTTPException(409,'Этот email уже используется. Обратитесь к администратору платформы.') from None
    response=JSONResponse({'ok':True});auth.new_session(key,response,request);return response
