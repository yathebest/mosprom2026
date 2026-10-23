"""Two-level RBAC, opaque cookie sessions and enterprise enrollment."""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from . import storage as db
from .access import Scope, current_scope

router = APIRouter()
COOKIE = 'safety_session'
SESSION_SECONDS = 12 * 3600
READ = {'events.read','live.read','jobs.read','analytics.read','evidence.read','export'}
ROLES = {
    'enterprise_admin': ('Администратор предприятия', READ | {'members.manage','rules.manage','analysis.run','live.manage','events.review','audit.read'}),
    'safety_specialist': ('Специалист по охране труда', READ | {'rules.manage','analysis.run','live.manage','events.review','audit.read'}),
    'foreman': ('Мастер / начальник смены', {'events.read','live.read','jobs.read','evidence.read','events.review'}),
    'operator': ('Оператор / диспетчер', {'events.read','live.read','jobs.read','evidence.read','events.review'}),
    'worker': ('Работник / водитель техники', set()),
    'safety_legal': ('Служба охраны труда / юрист', {'events.read','jobs.read','evidence.read','export','audit.read'}),
    'management': ('Менеджмент предприятия', {'analytics.read'}),
}


def init():
    with db.connection() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS organizations (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
          note TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT '',
          created TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')));
        CREATE TABLE IF NOT EXISTS users (
          id TEXT PRIMARY KEY, login TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
          password_hash TEXT NOT NULL, platform_admin INTEGER NOT NULL DEFAULT 0,
          organization_id TEXT REFERENCES organizations(id), role TEXT NOT NULL,
          camera_scope TEXT, active INTEGER NOT NULL DEFAULT 1, must_change INTEGER NOT NULL DEFAULT 0,
          created TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')));
        CREATE TABLE IF NOT EXISTS sessions (
          token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
          csrf TEXT NOT NULL, active_organization TEXT, expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS audit (
          id INTEGER PRIMARY KEY AUTOINCREMENT, organization_id TEXT, actor_id TEXT,
          actor_name TEXT, action TEXT NOT NULL, target TEXT, details TEXT NOT NULL DEFAULT '{}',
          created TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')));
        CREATE TABLE IF NOT EXISTS auth_attempts (key TEXT PRIMARY KEY, count INTEGER NOT NULL, until REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS audit_org ON audit(organization_id,id);
        CREATE TABLE IF NOT EXISTS admin_invitations (
          id TEXT PRIMARY KEY, organization_id TEXT NOT NULL REFERENCES organizations(id),
          email TEXT NOT NULL, token_hash TEXT NOT NULL UNIQUE, expires REAL NOT NULL,
          status TEXT NOT NULL, created_by TEXT NOT NULL, consumed_by TEXT,
          created TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')));
        CREATE INDEX IF NOT EXISTS invitations_target ON admin_invitations(organization_id,email);
        ''')
        c.execute("INSERT OR IGNORE INTO organizations(id,name,status,note) VALUES('legacy','Локальный проект','active','Данные до включения авторизации')")
        c.execute('DELETE FROM sessions WHERE expires<?',(time.time(),))
        c.execute('DELETE FROM auth_attempts WHERE until<?',(time.time(),))


def password_hash(password):
    if not 12 <= len(password) <= 256:
        raise HTTPException(422,'Пароль: от 12 до 256 символов')
    salt=secrets.token_bytes(16)
    value=hashlib.scrypt(password.encode(),salt=salt,n=32768,r=8,p=3,maxmem=64*1024*1024).hex()
    return f'scrypt${salt.hex()}${value}'


def verify(password, encoded):
    try:
        kind,salt,expected=encoded.split('$')
        if kind!='scrypt' or len(password)>256:return False
        actual=hashlib.scrypt(password.encode(),salt=bytes.fromhex(salt),n=32768,r=8,p=3,maxmem=64*1024*1024).hex()
        return hmac.compare_digest(actual,expected)
    except (ValueError,TypeError):return False


def normalize(login):
    value=login.strip().lower()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.@+\-]{2,119}',value):
        raise HTTPException(422,'Логин: 3–120 латинских букв, цифр или символов . _ @ + -')
    return value


def audit(actor,action,target='',details=None,organization_id=None,conn=None):
    values=(organization_id if organization_id is not None else (actor or {}).get('organization_id'),
            (actor or {}).get('id'),(actor or {}).get('name'),action,target,json.dumps(details or {},ensure_ascii=False))
    sql='INSERT INTO audit(organization_id,actor_id,actor_name,action,target,details) VALUES(?,?,?,?,?,?)'
    if conn:conn.execute(sql,values)
    else:
        with db.connection() as c:c.execute(sql,values)


def resolve(token):
    if not token or len(token)>128:return None
    with db.connection() as c:
        row=c.execute('''SELECT users.*, sessions.csrf,sessions.active_organization,sessions.token_hash
            FROM sessions JOIN users ON users.id=sessions.user_id
            WHERE token_hash=? AND expires>? AND users.active=1''',
            (hashlib.sha256(token.encode()).hexdigest(),time.time())).fetchone()
        if not row:return None
        user=dict(row)
        org_id=user['active_organization'] if user['platform_admin'] else user['organization_id']
        org=c.execute('SELECT * FROM organizations WHERE id=?',(org_id,)).fetchone() if org_id else None
        user['organization']=dict(org) if org else None
        user['organization_id']=org_id
        user['permissions']=sorted(ROLES['enterprise_admin' if user['platform_admin'] else user['role']][1]) if org and org['status']=='active' else []
        user['cameras']=json.loads(user['camera_scope']) if user['camera_scope'] is not None else None
        if user['role']=='foreman' and user['cameras'] is None:user['cameras']=[]
        return user


def public_user(user):
    return {k:user[k] for k in ('id','login','name','platform_admin','organization_id','organization','role','permissions','cameras','must_change','csrf')}


def require(request,permission=None,platform=False):
    user=getattr(request.state,'user',None)
    if not user:raise HTTPException(401,'Войдите в аккаунт')
    if platform and not user['platform_admin']:raise HTTPException(403,'Требуются права администратора платформы')
    if permission and permission not in user['permissions']:raise HTTPException(403,'Недостаточно прав для этого действия')
    return user


def rate_limit(key,maximum=10):
    with db.connection() as c:
        c.execute('BEGIN IMMEDIATE')
        row=c.execute('SELECT * FROM auth_attempts WHERE key=?',(key,)).fetchone()
        now=time.time()
        if row and row['until']>now and row['count']>=maximum:raise HTTPException(429,'Слишком много попыток. Повторите через 15 минут.')
        c.execute('INSERT INTO auth_attempts(key,count,until) VALUES(?,1,?) ON CONFLICT(key) DO UPDATE SET count=CASE WHEN until<? THEN 1 ELSE count+1 END, until=CASE WHEN until<? THEN excluded.until ELSE until END',(key,now+900,now,now))


def new_session(user_id,response,request):
    token=secrets.token_urlsafe(32);csrf=secrets.token_urlsafe(32)
    with db.connection() as c:
        old=request.cookies.get(COOKIE,'')
        c.execute('DELETE FROM sessions WHERE token_hash=?',(hashlib.sha256(old.encode()).hexdigest(),))
        c.execute('INSERT INTO sessions VALUES(?,?,?,?,?)',(hashlib.sha256(token.encode()).hexdigest(),user_id,csrf,None,time.time()+SESSION_SECONDS))
    response.set_cookie(COOKIE,token,httponly=True,samesite='strict',secure=request.url.scheme=='https' or os.getenv('SAFETY_SECURE_COOKIE')=='1',max_age=SESSION_SECONDS,path='/')
    return token


class StrictModel(BaseModel):
    model_config=ConfigDict(extra='forbid')


class Login(StrictModel):
    login: str=Field(min_length=3,max_length=120)
    password: str=Field(min_length=1,max_length=256)


class Enrollment(Login):
    name: str=Field(min_length=2,max_length=120)
    enterprise: str=Field(min_length=2,max_length=180)
    note: str=Field(default='',max_length=2000)


@router.get('/api/auth/status')
def status():
    with db.connection() as c:
        return {'initialized':bool(c.execute('SELECT 1 FROM users WHERE platform_admin=1 AND active=1').fetchone())}


@router.post('/api/auth/login')
def login(body: Login,request: Request):
    login=normalize(body.login)
    rate_limit('login-ip:'+request.client.host,30);rate_limit('login:'+hashlib.sha256(login.encode()).hexdigest())
    with db.connection() as c:row=c.execute('SELECT * FROM users WHERE login=?',(login,)).fetchone()
    # Equal-cost password work for an unknown login.
    encoded=row['password_hash'] if row else 'scrypt$'+'00'*16+'$'+'00'*64
    valid=verify(body.password,encoded)
    if not row or not valid or not row['active']:raise HTTPException(401,'Неверный логин или пароль')
    response=JSONResponse({'ok':True});token=new_session(row['id'],response,request)
    audit(dict(row),'auth.login')
    with db.connection() as c:c.execute('DELETE FROM auth_attempts WHERE key=?',('login:'+hashlib.sha256(login.encode()).hexdigest(),))
    return response


@router.post('/api/auth/register')
def enroll(body: Enrollment,request: Request):
    rate_limit('register:'+request.client.host,5)
    login=normalize(body.login);encoded=password_hash(body.password)
    org_id=uuid.uuid4().hex;user_id=uuid.uuid4().hex
    try:
        with db.connection() as c:
            c.execute('INSERT INTO organizations(id,name,status,note) VALUES(?,?,?,?)',(org_id,body.enterprise,'pending',body.note))
            c.execute('INSERT INTO users(id,login,name,password_hash,organization_id,role) VALUES(?,?,?,?,?,?)',(user_id,login,body.name,encoded,org_id,'enterprise_admin'))
            audit({'id':user_id,'name':body.name,'organization_id':org_id},'organization.request',org_id,conn=c)
    except sqlite3.IntegrityError:raise HTTPException(409,'Этот логин уже занят') from None
    response=JSONResponse({'ok':True});new_session(user_id,response,request);return response


@router.get('/api/auth/me')
def me(request: Request):return public_user(require(request))


@router.post('/api/auth/logout')
def logout(request: Request):
    user=require(request)
    with db.connection() as c:c.execute('DELETE FROM sessions WHERE token_hash=?',(user['token_hash'],))
    response=JSONResponse({'ok':True});response.delete_cookie(COOKIE,path='/');return response


class Password(StrictModel):
    current: str=Field(min_length=1,max_length=256)
    password: str=Field(min_length=12,max_length=256)


@router.post('/api/auth/password')
def change_password(body: Password,request: Request):
    user=require(request);rate_limit('password:'+user['id'])
    if not verify(body.current,user['password_hash']):raise HTTPException(400,'Текущий пароль неверен')
    if body.current==body.password:raise HTTPException(422,'Выберите новый пароль')
    encoded=password_hash(body.password)
    with db.connection() as c:
        c.execute('UPDATE users SET password_hash=?,must_change=0 WHERE id=?',(encoded,user['id']))
        c.execute('DELETE FROM sessions WHERE user_id=?',(user['id'],))
        audit(user,'auth.password_changed',user['id'],conn=c)
    response=JSONResponse({'ok':True});new_session(user['id'],response,request);return response


@router.get('/api/roles')
def roles(request: Request):
    require(request)
    return [{'key':key,'name':name,'permissions':sorted(perms)} for key,(name,perms) in ROLES.items()]


@router.get('/api/platform/organizations')
def organizations(request: Request):
    require(request,platform=True)
    with db.connection() as c:
        return [dict(r) for r in c.execute('''SELECT organizations.*,
            (SELECT count(*) FROM users WHERE users.organization_id=organizations.id AND active=1) AS members
            FROM organizations ORDER BY created DESC,rowid DESC''')]


class Decision(StrictModel):
    status: str=Field(pattern='^(active|rejected|suspended|pending)$')
    reason: str=Field(default='',max_length=1000)


@router.patch('/api/platform/organizations/{org_id}')
def decide(org_id: str,body: Decision,request: Request):
    user=require(request,platform=True)
    if body.status in ('rejected','suspended') and not body.reason.strip():raise HTTPException(422,'Укажите причину решения')
    with db.connection() as c:
        c.execute('BEGIN IMMEDIATE')
        row=c.execute('SELECT status FROM organizations WHERE id=?',(org_id,)).fetchone()
        if not row:raise HTTPException(404,'Предприятие не найдено')
        allowed={'pending':{'active','rejected'},'active':{'suspended'},'suspended':{'active'},'rejected':set()}
        if body.status not in allowed[row['status']]:raise HTTPException(409,'Заявка уже рассмотрена или переход недопустим. Обновите список.')
        c.execute('UPDATE organizations SET status=?,reason=? WHERE id=?',(body.status,body.reason,org_id))
        audit(user,'organization.'+body.status,org_id,{'from':row['status'],'reason':body.reason},org_id,c)
    if body.status=='suspended':
        from . import live,video
        with db.connection() as c:ids={r['id'] for r in c.execute('SELECT id FROM jobs WHERE organization_id=?',(org_id,))}
        for key in ids:
            if key in live.SESSIONS:live.SESSIONS[key].stop_event.set()
            if key in video.CANCEL:video.CANCEL[key].set()
    return {'ok':True}


class Context(StrictModel):
    organization_id: str | None=Field(default=None,max_length=64)


@router.post('/api/platform/context')
def context(body: Context,request: Request):
    user=require(request,platform=True)
    with db.connection() as c:
        if body.organization_id:
            row=c.execute("SELECT 1 FROM organizations WHERE id=? AND status='active'",(body.organization_id,)).fetchone()
            if not row:raise HTTPException(409,'Рабочее пространство доступно только активному предприятию')
        c.execute('UPDATE sessions SET active_organization=? WHERE token_hash=?',(body.organization_id,user['token_hash']))
        audit(user,'platform.context',body.organization_id or '',organization_id=body.organization_id,conn=c)
    return {'ok':True}


@router.post('/api/account/reapply')
def reapply(request: Request):
    user=require(request)
    if user['platform_admin'] or user['role']!='enterprise_admin':raise HTTPException(403,'Недостаточно прав')
    with db.connection() as c:
        result=c.execute("UPDATE organizations SET status='pending',reason='' WHERE id=? AND status='rejected'",(user['organization_id'],))
        if not result.rowcount:raise HTTPException(409,'Повторная заявка доступна после отклонения')
        audit(user,'organization.reapply',user['organization_id'],conn=c)
    return {'ok':True}


@router.get('/api/members')
def members(request: Request):
    user=require(request,'members.manage')
    with db.connection() as c:
        return [dict(r) for r in c.execute('SELECT id,login,name,role,camera_scope,active,must_change,created FROM users WHERE organization_id=? AND platform_admin=0 ORDER BY active DESC,name',(user['organization_id'],))]


class Member(StrictModel):
    login: str=Field(min_length=3,max_length=120)
    name: str=Field(min_length=2,max_length=120)
    role: str
    cameras: list[str] | None=Field(default=None,max_length=100)
    password: str=Field(min_length=12,max_length=256)


class MemberUpdate(StrictModel):
    name: str=Field(min_length=2,max_length=120)
    role: str
    cameras: list[str] | None=Field(default=None,max_length=100)
    active: bool=True


def validate_member(body,c):
    if body.role not in ROLES:raise HTTPException(422,'Неизвестная локальная роль')
    if body.role=='foreman' and not body.cameras:raise HTTPException(422,'Назначьте мастеру хотя бы одну камеру участка')
    if body.role in ('enterprise_admin','safety_specialist') and body.cameras is not None:raise HTTPException(422,'Администратор и специалист по охране труда управляют всеми камерами предприятия')
    if body.cameras is not None:
        known={r['name'] for r in db.cameras()}
        if not set(body.cameras)<=known:raise HTTPException(422,'Выбрана камера другого предприятия или несуществующая камера')
    return json.dumps(list(dict.fromkeys(body.cameras)),ensure_ascii=False) if body.cameras is not None else None


@router.post('/api/members')
def add_member(body: Member,request: Request):
    user=require(request,'members.manage');login=normalize(body.login);encoded=password_hash(body.password);key=uuid.uuid4().hex
    try:
        with db.connection() as c:
            cameras=validate_member(body,c)
            c.execute('INSERT INTO users(id,login,name,password_hash,organization_id,role,camera_scope,must_change) VALUES(?,?,?,?,?,?,?,1)',(key,login,body.name,encoded,user['organization_id'],body.role,cameras))
            audit(user,'member.created',key,{'role':body.role,'cameras':body.cameras},conn=c)
    except sqlite3.IntegrityError:raise HTTPException(409,'Этот логин уже занят') from None
    return {'id':key}


@router.patch('/api/members/{member_id}')
def update_member(member_id: str,body: MemberUpdate,request: Request):
    user=require(request,'members.manage')
    with db.connection() as c:
        c.execute('BEGIN IMMEDIATE');cameras=validate_member(body,c)
        target=c.execute('SELECT * FROM users WHERE id=? AND organization_id=? AND platform_admin=0',(member_id,user['organization_id'])).fetchone()
        if not target:raise HTTPException(404,'Сотрудник не найден')
        if target['role']=='enterprise_admin' and target['active'] and (body.role!='enterprise_admin' or not body.active):
            count=c.execute("SELECT count(*) FROM users WHERE organization_id=? AND role='enterprise_admin' AND active=1",(user['organization_id'],)).fetchone()[0]
            if count<=1:raise HTTPException(409,'Нельзя удалить или понизить последнего администратора предприятия')
        c.execute('UPDATE users SET name=?,role=?,camera_scope=?,active=? WHERE id=?',(body.name,body.role,cameras,int(body.active),member_id))
        c.execute('DELETE FROM sessions WHERE user_id=?',(member_id,))
        audit(user,'member.updated',member_id,{'role':body.role,'cameras':body.cameras,'active':body.active},conn=c)
    return {'ok':True}


class Reset(StrictModel):
    password: str=Field(min_length=12,max_length=256)


@router.post('/api/members/{member_id}/password')
def reset_password(member_id: str,body: Reset,request: Request):
    user=require(request,'members.manage')
    if member_id==user['id']:raise HTTPException(409,'Свой пароль изменяйте в разделе «Аккаунт»')
    encoded=password_hash(body.password)
    with db.connection() as c:
        result=c.execute('UPDATE users SET password_hash=?,must_change=1 WHERE id=? AND organization_id=? AND platform_admin=0',(encoded,member_id,user['organization_id']))
        if not result.rowcount:raise HTTPException(404,'Сотрудник не найден')
        c.execute('DELETE FROM sessions WHERE user_id=?',(member_id,))
        audit(user,'member.password_reset',member_id,conn=c)
    return {'ok':True}


@router.get('/api/audit')
def audit_log(request: Request):
    user=require(request)
    if not user['platform_admin']:require(request,'audit.read')
    with db.connection() as c:
        query='SELECT * FROM audit';args=[]
        if not user['platform_admin'] or user['organization_id']:
            query+=' WHERE organization_id=?';args.append(user['organization_id'])
        return [dict(r) for r in c.execute(query+' ORDER BY id DESC LIMIT 300',args)]


def endpoint_permission(path,method):
    if path.startswith('/media/'):return 'evidence.read'
    if path in ('/api/health','/api/quality','/api/scenarios','/api/defaults','/api/samples'):return None
    if path=='/api/summary':return 'analytics.read'
    if path in ('/api/export','/api/review-dataset') or path.endswith('/report'):return 'export'
    if path=='/api/cameras':return 'rules.manage' if method!='GET' else 'jobs.read'
    if path.startswith('/api/live'):
        return 'live.read' if method=='GET' else 'live.manage'
    if path=='/api/streams':return 'analysis.run'
    if path.startswith('/api/jobs') or (path.startswith('/api/samples/') and path.endswith('/import')):
        return 'jobs.read' if method=='GET' else 'analysis.run'
    if path.startswith('/api/events') or path=='/api/triage':return 'events.read' if method=='GET' else 'events.review'
    return None


def install(app):
    app.include_router(router)
    from . import invitations
    app.include_router(invitations.router)

    @app.middleware('http')
    async def access_middleware(request,call_next):
        path=request.url.path;protected=path.startswith(('/api/','/media/'))
        user=None;token=None
        try:
            if protected:
                if request.method not in ('GET','HEAD','OPTIONS'):
                    origin=request.headers.get('origin')
                    if origin and origin!=str(request.base_url).rstrip('/'):
                        raise HTTPException(403,'Запрос с другого сайта запрещён')
                    if request.headers.get('sec-fetch-site')=='cross-site':raise HTTPException(403,'Запрос с другого сайта запрещён')
                public=path in ('/api/auth/login','/api/auth/register','/api/auth/status',
                               '/api/invitations/inspect','/api/invitations/accept')
                if not public:
                    user=resolve(request.cookies.get(COOKIE));request.state.user=user
                    if not user:raise HTTPException(401,'Войдите в аккаунт')
                    if request.method not in ('GET','HEAD','OPTIONS') and not hmac.compare_digest(request.headers.get('x-csrf-token','').encode(),user['csrf'].encode()):
                        raise HTTPException(403,'Недействительный защитный токен. Обновите страницу.')
                    account=path.startswith(('/api/auth/','/api/account/')) or path=='/api/roles'
                    platform=path.startswith('/api/platform/')
                    if user['must_change'] and path not in ('/api/auth/me','/api/auth/password','/api/auth/logout'):
                        raise HTTPException(403,'Сначала смените временный пароль')
                    if not account and not platform and not (path=='/api/audit' and user['platform_admin']):
                        if not user['organization'] or user['organization']['status']!='active':raise HTTPException(403,'Рабочее пространство предприятия недоступно')
                        permission=endpoint_permission(path,request.method)
                        if permission:require(request,permission)
                    token=current_scope.set(Scope(user['organization_id'] or '__no_enterprise__',tuple(user['cameras']) if user['cameras'] is not None else None))
            response=await call_next(request)
            if protected:response.headers['Cache-Control']='no-store'
            response.headers['X-Content-Type-Options']='nosniff'
            response.headers['Referrer-Policy']='same-origin'
            response.headers['X-Frame-Options']='DENY'
            return response
        except HTTPException as exc:return JSONResponse({'detail':exc.detail},status_code=exc.status_code,headers={'Cache-Control':'no-store'})
        finally:
            if token is not None:current_scope.reset(token)
