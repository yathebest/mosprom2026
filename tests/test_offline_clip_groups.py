import json
from pathlib import Path
import shutil
import sqlite3
import threading

import cv2
import numpy as np
import pytest

from app import storage as db,video
from scripts.coalesce_legacy_clips import coalesce


def media(path,frames=75,color=80):
    path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(96,64));assert writer.isOpened()
    for _ in range(frames):writer.write(np.full((64,96,3),color,np.uint8))
    writer.release();return path


@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3');db.init(False)
    db.add_job('j','source.mp4',{});db.update_job('j',config={'camera':'Camera'},stats={'kept':True})
    return tmp_path


def add(tid,*,job='j',camera='Camera',alert=1,clip=None,start=0,end=3):
    event={'track_id':tid,'kind':'no_helmet','zone':'z','start_s':.1,'alert_s':alert,'sequence':1,
        'episode_id':job+'-episode-'+str(tid),'episode_state':'interrupted','end_s':None,'observed_until_s':99,
        'clip_state':'ready','clip_scope':'incident_excerpt','clip_start_s':start,'clip_end_s':end,
        'clip_alert_offset_s':alert-start,'reviewed_by':'Operator'}
    event_id=db.add_event(job,camera,event,f'runs/{job}/proof-{tid}.jpg')
    db.review(event_id,'confirmed' if tid%2 else 'false')
    if clip:db.set_clip(event_id,clip)
    return event_id,event


def test_offline_two_person_events_use_one_real_encoder_and_keep_separate_reviews(store,monkeypatch):
    if not video.ffmpeg_path():pytest.skip('H264 encoder unavailable')
    source=media(store/'runs/j/source.mp4',frames=200)
    first=add(1,alert=3);second=add(2,alert=3.5)
    real=video.subprocess.run;calls=[]
    def encoder(*args,**kwargs):calls.append(args[0]);return real(*args,**kwargs)
    monkeypatch.setattr(video.subprocess,'run',encoder)
    result=video.export_event_clips('j','Camera',[second,first],source,store/'runs/j',8,threading.Event())
    assert result=={'clip_failures':0,'clip_groups':1,'clip_files':1} and len(calls)==1
    rows=db.events();assert len(rows)==2 and len({row['clip'] for row in rows})==1
    for row in rows:
        payload=row['payload']
        assert payload['clip_group_counts']=={'no_helmet':2}
        assert set(payload['clip_group_event_ids'])=={first[0],second[0]}
        assert payload['clip_group_people_count']==2 and payload['clip_state']=='ready'
        assert payload['clip_alert_offset_s']==row['alert_s']-payload['clip_start_s']
        assert payload['clip_duration_s']>7 and payload['end_s'] is None and payload['episode_state']=='interrupted'
    assert {row['status'] for row in rows}=={'confirmed','false'}
    assert db.job('j')['stats']=={'kept':True}


def test_legacy_exact_twins_are_referenced_once_backed_up_idempotent_and_leave_files_endpoints_and_reviews(store):
    a=media(store/'runs/j/a.mp4');b=store/'runs/j/b.mp4';shutil.copyfile(a,b)
    first=add(1,clip='runs/j/a.mp4');second=add(2,alert=1.2,clip='runs/j/b.mp4')
    before={row['id']:row for row in db.events()}
    dry=coalesce(db.DB,store);assert dry['groups']==1 and dry['would_update_events']==2 and dry['updated_events']==0
    assert {row['id']:row for row in db.events()}==before
    result=coalesce(db.DB,store,apply=True)
    assert result['updated_events']==2 and Path(result['backup']).is_file() and result['files_deleted']==0
    assert a.is_file() and b.is_file()
    rows=db.events();assert len({row['clip'] for row in rows})==1
    for row in rows:
        old=before[row['id']]
        assert row['status']==old['status'] and row['image']==old['image']
        assert row['payload']['end_s'] is None and row['payload']['episode_state']=='interrupted'
        assert row['payload']['clip_start_s']==0 and row['payload']['clip_end_s']==3
        assert row['payload']['clip_duration_s']==3
        assert row['payload']['clip_legacy_original']['clip']==old['clip']
        assert row['payload']['clip_group_counts']=={'no_helmet':2}
    again=coalesce(db.DB,store,apply=True)
    assert again['updated_events']==0 and again['groups']==0 and again['backup'] is None


def test_legacy_near_clip_uses_existing_covering_media_not_imaginary_union(store):
    media(store/'runs/j/a.mp4',frames=80);media(store/'runs/j/b.mp4')
    add(1,alert=1,clip='runs/j/a.mp4',start=0,end=3.2)
    add(2,alert=1.2,clip='runs/j/b.mp4',start=.1,end=3.1)
    result=coalesce(db.DB,store,apply=True)
    assert result['updated_events']==2
    for row in db.events():
        assert row['clip']=='runs/j/a.mp4'
        assert row['payload']['clip_start_s']==0 and row['payload']['clip_end_s']==3.2
        assert row['payload']['clip_duration_s']==3.2
        assert row['payload']['clip_alert_offset_s']==row['alert_s']


def test_legacy_different_visual_content_and_disjoint_windows_are_not_coalesced(store):
    media(store/'runs/j/a.mp4',frames=80,color=10);media(store/'runs/j/b.mp4',color=220)
    add(1,alert=1,clip='runs/j/a.mp4',start=0,end=3.2)
    add(2,alert=1.2,clip='runs/j/b.mp4',start=.1,end=3.1)
    assert coalesce(db.DB,store)['groups']==0
    third=store/'runs/j/c.mp4';shutil.copyfile(store/'runs/j/a.mp4',third)
    add(3,alert=21,clip='runs/j/c.mp4',start=20,end=23.2)
    assert coalesce(db.DB,store)['groups']==0


def test_legacy_never_merges_other_job_or_camera(store):
    a=media(store/'runs/j/a.mp4');add(1,clip='runs/j/a.mp4')
    b=store/'runs/j/b.mp4';shutil.copyfile(a,b);add(2,camera='Other camera',clip='runs/j/b.mp4')
    db.add_job('other','other.mp4',{});c=store/'runs/other/c.mp4';c.parent.mkdir();shutil.copyfile(a,c)
    add(3,job='other',clip='runs/other/c.mp4')
    assert coalesce(db.DB,store,apply=True)['updated_events']==0


def test_legacy_nullable_organization_partition_is_applied_with_null_safe_equality(tmp_path):
    database=tmp_path/'legacy.sqlite3';a=media(tmp_path/'runs/j/a.mp4');b=a.with_name('b.mp4');shutil.copyfile(a,b)
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY, organization_id TEXT)');conn.execute("INSERT INTO jobs VALUES('j',NULL)")
        conn.execute('CREATE TABLE events (id TEXT,job_id TEXT,camera TEXT,clip TEXT,payload TEXT,alert_s REAL,kind TEXT,track_id INTEGER,sequence INTEGER,zone TEXT,status TEXT)')
        for i,clip in enumerate(('runs/j/a.mp4','runs/j/b.mp4'),1):
            payload={'episode_state':'interrupted','end_s':None,'clip_state':'ready','clip_start_s':0,'clip_end_s':3}
            conn.execute("INSERT INTO events VALUES(?,'j','Camera',?,?,1,'no_helmet',?,1,'','confirmed')",(str(i),clip,json.dumps(payload),i))
    result=coalesce(database,tmp_path,apply=True)
    assert result['groups']==1 and result['updated_events']==2
    with sqlite3.connect(database) as conn:assert len({row[0] for row in conn.execute('SELECT clip FROM events')})==1
