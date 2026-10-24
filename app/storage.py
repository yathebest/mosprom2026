from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import uuid
from .access import current_scope, sql_scope

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "events.sqlite3"


@contextmanager
def connection():
    DATA.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB,timeout=20)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init(mark_interrupted=True):
    with connection() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, filename TEXT, status TEXT, progress REAL DEFAULT 0,
          meta TEXT, config TEXT, stats TEXT, error TEXT,
          created TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')));
        CREATE TABLE IF NOT EXISTS events (
          id TEXT PRIMARY KEY, job_id TEXT, track_id INTEGER, kind TEXT, zone TEXT,
          camera TEXT, start_s REAL, alert_s REAL, sequence INTEGER, payload TEXT,
          status TEXT DEFAULT 'new', image TEXT, clip TEXT,
          created TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
          UNIQUE(job_id, track_id, kind, zone, sequence));
        CREATE TABLE IF NOT EXISTS cameras (
          name TEXT PRIMARY KEY, config TEXT NOT NULL,
          updated TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')));
        """)
        if 'organization_id' not in {r['name'] for r in c.execute('PRAGMA table_info(jobs)')}:
            c.execute("ALTER TABLE jobs ADD COLUMN organization_id TEXT NOT NULL DEFAULT 'legacy'")
        if 'organization_id' not in {r['name'] for r in c.execute('PRAGMA table_info(cameras)')}:
            c.executescript("""
            ALTER TABLE cameras RENAME TO cameras_legacy;
            CREATE TABLE cameras (organization_id TEXT NOT NULL, name TEXT NOT NULL,
                config TEXT NOT NULL, updated TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                PRIMARY KEY(organization_id,name));
            INSERT INTO cameras SELECT 'legacy',name,config,updated FROM cameras_legacy;
            DROP TABLE cameras_legacy;
            """)
        c.execute('CREATE INDEX IF NOT EXISTS jobs_org ON jobs(organization_id)')
        c.execute('CREATE INDEX IF NOT EXISTS events_job ON events(job_id)')
        if 'episode_id' not in {r['name'] for r in c.execute('PRAGMA table_info(events)')}:
            c.execute('ALTER TABLE events ADD COLUMN episode_id TEXT')
            c.execute("UPDATE events SET episode_id=json_extract(payload,'$.episode_id') WHERE json_valid(payload)")
        c.execute('CREATE UNIQUE INDEX IF NOT EXISTS events_episode ON events(job_id,episode_id) WHERE episode_id IS NOT NULL')
        if mark_interrupted:
            c.execute("""UPDATE events SET payload=json_set(payload,'$.episode_state','interrupted',
                      '$.end_s',NULL,'$.closure_reason','server_restarted','$.state_to','unknown')
                      WHERE episode_id IS NOT NULL AND json_extract(payload,'$.episode_state')='open'
                      AND job_id IN (SELECT id FROM jobs WHERE status IN ('queued','running','exporting'))""")
            c.execute("UPDATE jobs SET status='interrupted',error='Сервер перезапущен во время обработки' WHERE status IN ('queued','running','exporting')")


def decode(row):
    d = dict(row)
    for key in ("meta","config","stats","payload"):
        if key in d:
            d[key] = json.loads(d[key]) if d[key] else None
    return d


def jobs():
    clause,args=sql_scope()
    with connection() as c:
        return [decode(r) for r in c.execute("SELECT * FROM jobs WHERE 1=1"+clause+" ORDER BY created DESC,rowid DESC LIMIT 100",args)]


def job(job_id):
    clause,args=sql_scope()
    with connection() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?"+clause,(job_id,*args)).fetchone()
        return decode(row) if row else None


def add_job(job_id, filename, meta):
    scope=current_scope.get()
    with connection() as c:
        c.execute("INSERT INTO jobs(id,filename,status,meta,organization_id) VALUES(?,?,'ready',?,?)",(job_id,filename,json.dumps(meta),scope.organization_id if scope else 'legacy'))


def update_job(job_id, **values):
    allowed = {"status","progress","config","stats","error","meta"}
    if not set(values)<=allowed:
        raise ValueError("Invalid job field")
    values = {k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in values.items()}
    with connection() as c:
        c.execute(f"UPDATE jobs SET {','.join(k+'=?' for k in values)} WHERE id=?",(*values.values(),job_id))


def events(job_id=None, kind=None, camera=None, zone=None, status=None, start=None, end=None):
    query, params = "SELECT * FROM events WHERE 1=1",[]
    scope=current_scope.get()
    if scope:
        query+=' AND job_id IN (SELECT id FROM jobs WHERE organization_id=?)'
        params.append(scope.organization_id)
        if scope.cameras is not None:
            query+=f" AND camera IN ({','.join('?' for _ in scope.cameras) or 'NULL'})"
            params.extend(scope.cameras)
    for key,value in (("job_id",job_id),("kind",kind),("camera",camera),("zone",zone),("status",status)):
        if value is not None:
            query += f" AND {key}=?"
            params.append(value)
    for operator,value in ((">=",start),("<=",end)):
        if value is not None:
            query += f" AND alert_s{operator}?"
            params.append(value)
    with connection() as c:
        return [decode(r) for r in c.execute(query+" ORDER BY created DESC,alert_s DESC",params)]


def patch_payload(event_id,updates):
    with connection() as c:
        c.execute('BEGIN IMMEDIATE')
        row=c.execute('SELECT payload FROM events WHERE id=?',(event_id,)).fetchone()
        if row:
            value=json.loads(row['payload']);value.update(updates)
            c.execute('UPDATE events SET payload=? WHERE id=?',(json.dumps(value,ensure_ascii=False),event_id))


def add_event(job_id, camera, event, image, *, return_created=False):
    event_id = uuid.uuid4().hex
    with connection() as c:
        c.execute('BEGIN IMMEDIATE')
        # Modern episode identity supersedes the legacy gate sequence. A gate
        # recreated after track loss may restart sequence=1 for a new episode.
        if event.get('episode_id'):
            existing=c.execute('SELECT id FROM events WHERE job_id=? AND episode_id=?',(job_id,event['episode_id'])).fetchone()
            if existing:return (existing['id'],False) if return_created else existing['id']
            collision=c.execute('SELECT MAX(sequence) AS sequence FROM events WHERE job_id=? AND track_id=? AND kind=? AND zone IS ?',
                                (job_id,event['track_id'],event['kind'],event['zone'])).fetchone()['sequence']
            if collision is not None and collision>=event['sequence']:
                event={**event,'sequence':collision+1}
            event={**event,'start_state_from':event.get('state_from'),'start_state_to':event.get('state_to'),
                   'start_state_confirmation':event.get('state_confirmation'),
                   'state_transitions':[{'from':event.get('state_from'),'to':event.get('state_to'),
                       'at_s':event.get('alert_s'),'onset_s':event.get('start_s'),
                       'confirmation':event.get('state_confirmation')}]}
        result=c.execute("""INSERT OR IGNORE INTO events
          (id,job_id,track_id,kind,zone,camera,start_s,alert_s,sequence,payload,image,episode_id)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
          (event_id,job_id,event["track_id"],event["kind"],event["zone"],camera,
           event["start_s"],event["alert_s"],event["sequence"],json.dumps(event),image,event.get('episode_id')))
        created=result.rowcount>0
        if not created:
            if event.get('episode_id'):
                row=c.execute('SELECT id FROM events WHERE job_id=? AND episode_id=?',(job_id,event['episode_id'])).fetchone()
            else:row=None
            if row is None:
                row=c.execute('SELECT id FROM events WHERE job_id=? AND track_id=? AND kind=? AND zone IS ? AND sequence=?',
                              (job_id,event['track_id'],event['kind'],event['zone'],event['sequence'])).fetchone()
            if row is None:raise RuntimeError('Ignored incident insert has no existing incident')
            event_id=row['id']
    return (event_id,created) if return_created else event_id


def update_episode(job_id, update):
    """Patch one lifecycle transition without altering review/start evidence.

    Workers use their job ID. HTTP callers additionally retain request tenant
    and camera scope; an episode ID alone never identifies an accessible event.
    Returns the changed event ID, or None for stale/duplicate/inaccessible input.
    """
    if not update.get('episode_id') or update.get('episode_state') not in ('closed','interrupted'):return None
    clause,args=sql_scope('jobs','events.camera')
    with connection() as c:
        # Serialize read/merge/write with concurrent clip exporters and workers.
        c.execute('BEGIN IMMEDIATE')
        row=c.execute('SELECT events.* FROM events JOIN jobs ON jobs.id=events.job_id WHERE events.job_id=? AND events.episode_id=?'+clause,
                      (job_id,update['episode_id'],*args)).fetchone()
        if row is None:return None
        payload=json.loads(row['payload'])
        if payload.get('episode_state') in ('closed','interrupted'):return None
        keys=('episode_state','end_s','closed_at_s','duration_s','closure_reason','observed_until_s','end_image','state_from','state_to','state_confirmation')
        patch={key:update[key] for key in keys if key in update}
        def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
        for key in ('observed_until_s','closed_at_s','duration_s'):
            if key in patch and patch[key] is not None and not finite(patch[key]):return None
        if patch['episode_state']=='interrupted':
            patch['end_s']=None
            patch['state_to']='unknown'
            patch.pop('duration_s',None);patch.pop('end_image',None)
        else:
            end,confirmed=patch.get('end_s'),patch.get('closed_at_s')
            if not finite(end) or not finite(confirmed) or not confirmed>=end>=row['start_s']:return None
            patch['duration_s']=round(end-row['start_s'],3)
            patch.setdefault('state_to','clear')
        for name in ('state_from','state_to','state_confirmation'):
            payload.setdefault('start_'+name,payload.get(name))
        patch.setdefault('state_from',payload.get('state_to'))
        start_transition={'from':payload['start_state_from'],'to':payload['start_state_to'],
            'at_s':row['alert_s'],'onset_s':row['start_s'],'confirmation':payload['start_state_confirmation']}
        payload['state_transitions']=[start_transition,{'from':patch.get('state_from'),'to':patch.get('state_to'),
            'at_s':patch.get('closed_at_s',patch.get('observed_until_s')),'onset_s':patch.get('end_s'),
            'confirmation':patch.get('state_confirmation'),'reason':patch.get('closure_reason')}]
        payload.update(patch)
        c.execute('UPDATE events SET payload=? WHERE id=?',(json.dumps(payload,ensure_ascii=False),row['id']))
        return row['id']


def set_clip(event_id,path):
    with connection() as c:
        c.execute("UPDATE events SET clip=? WHERE id=?",(path,event_id))


def review(event_id,status):
    with connection() as c:
        scope=current_scope.get()
        if scope:
            query='SELECT events.* FROM events JOIN jobs ON jobs.id=events.job_id WHERE events.id=? AND jobs.organization_id=?'
            args=[event_id,scope.organization_id]
            if scope.cameras is not None:
                query+=f" AND events.camera IN ({','.join('?' for _ in scope.cameras) or 'NULL'})";args.extend(scope.cameras)
            if not c.execute(query,args).fetchone():return False
        result = c.execute("UPDATE events SET status=? WHERE id=?",(status,event_id))
        return result.rowcount > 0


def cameras():
    clause,args=sql_scope('cameras','cameras.name')
    with connection() as c:
        return [decode(row) for row in c.execute('SELECT * FROM cameras WHERE 1=1'+clause+' ORDER BY name',args)]


def save_camera(config):
    scope=current_scope.get()
    with connection() as c:
        c.execute("INSERT INTO cameras(organization_id,name,config) VALUES(?,?,?) ON CONFLICT(organization_id,name) DO UPDATE SET config=excluded.config,updated=strftime('%Y-%m-%dT%H:%M:%SZ','now')",
                  (scope.organization_id if scope else 'legacy',config['camera'],json.dumps(config,ensure_ascii=False)))
