"""Conservative shared-reference migration. Dry-run default; never deletes media."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.clip_metadata import probe_clip,safe_path
from app.clip_groups import episode_identity,member_counts


def number(value):
    import math
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else None


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def matching_overlap(a,b):
    """Conservative same-source visual check for slightly shifted encodes, CPU only."""
    import cv2
    import numpy as np
    start=max(a['start'],b['start']);end=min(a['end'],b['end'])
    if end-start<.4:return False
    caps=[cv2.VideoCapture(str(item['path'])) for item in (a,b)]
    try:
        for when in (start+.15,(start+end)/2,end-.15):
            frames=[]
            for cap,item in zip(caps,(a,b)):
                cap.set(cv2.CAP_PROP_POS_MSEC,max(0,when-item['start'])*1000);ok,frame=cap.read()
                if not ok:return False
                frames.append(cv2.resize(frame,(64,36),interpolation=cv2.INTER_AREA).astype(np.float32))
            difference=np.abs(frames[0]-frames[1])
            if difference.mean()>8 or np.percentile(difference,95)>25:return False
        return True
    finally:
        for cap in caps:cap.release()


def compatible(canonical,member,tolerance):
    if max(abs(canonical['start']-member['start']),abs(canonical['end']-member['end']))>tolerance:return False
    if abs(canonical['alert']-member['alert'])>2:return False
    # A canonical file must contain BOTH alert positions, not an imaginary union.
    for alert in (canonical['alert'],member['alert']):
        if not canonical['start']<=alert<=min(canonical['end'],canonical['start']+canonical['media']['clip_duration_s'])+.001:return False
    exact=max(abs(canonical['start']-member['start']),abs(canonical['end']-member['end']))<=.001
    if exact:return canonical['sha']==member['sha']
    return matching_overlap(canonical,member)


def coalesce(database,data_dir,*,apply=False,tolerance=.25):
    database=Path(database).resolve();data_dir=Path(data_dir).resolve()
    if not database.is_file() or not 0<=tolerance<=.5:raise ValueError('Existing DB and tolerance in [0,0.5] required')
    stats=Counter();partitions=defaultdict(list);plan=[]
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=20) as conn:
        conn.row_factory=sqlite3.Row
        rows=conn.execute("SELECT events.*,jobs.organization_id FROM events JOIN jobs ON jobs.id=events.job_id WHERE clip IS NOT NULL AND clip<>''").fetchall()
        hash_cache={}
        for row in rows:
            stats['scanned']+=1
            try:
                payload=json.loads(row['payload'] or '{}')
                if not isinstance(payload,dict):raise ValueError()
                if payload.get('clip_group_id'):stats['skipped_grouped']+=1;continue
                if payload.get('clip_state') not in (None,'ready'):stats['skipped_not_ready']+=1;continue
                start,end,alert=number(payload.get('clip_start_s')),number(payload.get('clip_end_s')),number(row['alert_s'])
                if start is None or end is None or alert is None or not 0<=start<=alert<=end or not 0<end-start<=10:
                    stats['skipped_bounds']+=1;continue
                path=safe_path(row['clip'],data_dir)
                if not path.is_relative_to((data_dir/'runs'/row['job_id']).resolve()):raise ValueError()
                media=probe_clip(path,data_dir=data_dir)
                if not media or alert-start>media['clip_duration_s']+.001:stats['skipped_unreadable']+=1;continue
                if path not in hash_cache:hash_cache[path]=digest(path)
                item={**dict(row),'payload':payload,'path':path,'start':start,'end':end,'alert':alert,'media':media,'sha':hash_cache[path]}
                partitions[(row['organization_id'],row['job_id'],row['camera'])].append(item)
            except (OSError,ValueError,TypeError):stats['skipped_invalid']+=1
        for scope,values in partitions.items():
            remaining=sorted(values,key=lambda row:(-(row['end']-row['start']),row['start'],row['clip'],row['id']))
            while remaining:
                canonical=remaining.pop(0);members=[canonical]
                for candidate in list(remaining):
                    anchors=[row['alert'] for row in members]+[candidate['alert']]
                    if max(anchors)-min(anchors)<=2 and compatible(canonical,candidate,tolerance):
                        members.append(candidate);remaining.remove(candidate)
                if len(members)<2:continue
                ids=sorted(row['id'] for row in members)
                group_id=uuid.uuid5(uuid.NAMESPACE_URL,'linq-legacy-clip:'+json.dumps([scope,canonical['clip'],ids])).hex
                identity_members=[{'identity':episode_identity(row['id'],{**row['payload'],'kind':row['kind'],
                    'track_id':row['track_id'],'sequence':row['sequence'],'zone':row['zone']}),
                    'kind':row['kind'],'track_id':row['track_id']} for row in members]
                counts=member_counts(identity_members)
                common={'clip_group_id':group_id,'clip_group_event_ids':ids,'clip_group_counts':counts['counts'],
                    'clip_group_episode_count':counts['episode_count'],'clip_group_people_count':counts['people_count'],
                    'clip_group_scene_events':counts['scene_events'],'clip_scope':'incident_excerpt',
                    'clip_group_migration':'verified_legacy_reference_v1',**canonical['media'],
                    'clip_start_s':canonical['start'],'clip_end_s':canonical['end']}
                updates={row['id']:{**common,'clip_alert_offset_s':round(row['alert']-canonical['start'],3),
                    'clip_legacy_original':{'clip':row['clip'],'metadata':{key:value for key,value in row['payload'].items() if key.startswith('clip_')}}}
                    for row in members}
                plan.append({'scope':scope,'clip':canonical['clip'],'updates':updates,
                    'members':{row['id']:{'clip':row['clip'],'start':row['start'],'end':row['end'],'sha':row['sha']} for row in members}})
        backup=None
        if apply and plan:
            folder=data_dir/'cache/clip-group-backups';folder.mkdir(parents=True,exist_ok=True)
            backup=folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.sqlite3')
            with sqlite3.connect(backup) as destination:conn.backup(destination)
    updated=0
    if apply and plan:
        with sqlite3.connect(database,timeout=20) as conn:
            conn.execute('BEGIN IMMEDIATE')
            for group in plan:
                organization,job,camera=group['scope'];fresh=[]
                for event_id,original in group['members'].items():
                    row=conn.execute('SELECT events.payload,events.clip FROM events JOIN jobs ON jobs.id=events.job_id '
                        'WHERE events.id=? AND events.job_id=? AND events.camera=? AND jobs.organization_id IS ?',
                        (event_id,job,camera,organization)).fetchone()
                    if not row:break
                    try:
                        payload=json.loads(row[0]);path=safe_path(row[1],data_dir)
                        if (row[1]!=original['clip'] or payload.get('clip_group_id') or payload.get('clip_state') not in (None,'ready')
                                or payload.get('clip_start_s')!=original['start'] or payload.get('clip_end_s')!=original['end']
                                or digest(path)!=original['sha']):break
                    except (OSError,ValueError,TypeError):break
                    fresh.append((event_id,payload))
                if len(fresh)!=len(group['members']):stats['skipped_changed_groups']+=1;continue
                for event_id,payload in fresh:
                    payload.update(group['updates'][event_id])
                    conn.execute('UPDATE events SET clip=?,payload=? WHERE id=?',
                        (group['clip'],json.dumps(payload,ensure_ascii=False),event_id));updated+=1
    return {'mode':'apply' if apply else 'dry_run','counts':dict(stats),'groups':len(plan),
        'would_update_events':sum(len(group['members']) for group in plan),'updated_events':updated,
        'backup':str(backup) if backup else None,'files_deleted':0,
        'policy':'Same job/organization/camera only. Exact bounds require identical bytes; near bounds require shared visual overlap and one existing clip covering every alert. Original references are archived; no incident/review fields or files are deleted.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--database',type=Path);parser.add_argument('--apply',action='store_true')
    parser.add_argument('--tolerance-s',type=float,default=.25);parser.add_argument('--report',type=Path)
    args=parser.parse_args();result=coalesce(args.database or args.data_dir/'events.sqlite3',args.data_dir,apply=args.apply,tolerance=args.tolerance_s)
    text=json.dumps(result,indent=2);print(text)
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(text,encoding='utf-8')
