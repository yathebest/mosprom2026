"""Shared physical media, independent person episodes; CPU/codec only, no NN."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import threading
import time

import cv2
import numpy as np
import pytest

from app.clip_groups import GroupPool
from app import live, storage as db
from app.access import Scope, current_scope
from app.core import RunConfig


def event(tid=1, kind='no_helmet', alert=10., episode='episode-1'):
    return dict(track_id=tid, kind=kind, zone='', alert_s=alert, start_s=max(0,alert-1),
                episode_id=episode, sequence=1, episode_state='open')


def test_three_people_two_patterns_share_one_bounded_group_and_individual_offsets():
    pool=GroupPool('job','Camera','org')
    a=pool.add('a',event())
    b=pool.add('b',event(tid=2,episode='episode-2',alert=10.4))
    c=pool.add('c',event(tid=3,kind='possible_phone_use',episode='episode-3',alert=10.7))
    assert a is b is c and len(pool.groups)==1
    assert (a.start_s,a.end_s)==(7.,14.7)
    meta=a.metadata()
    assert meta['clip_group_counts']=={'no_helmet':2,'possible_phone_use':1}
    assert meta['clip_group_episode_count']==3 and meta['clip_group_people_count']==3
    assert set(meta['clip_group_event_ids'])=={'a','b','c'}
    assert a.member_metadata('c')['clip_alert_offset_s']==3.7


def test_same_episode_handoff_and_retries_never_expand_window_or_double_count_actor():
    pool=GroupPool('job','Camera');group=pool.add('a',event())
    pool.add('a',event(tid=9,alert=100))
    pool.add('alias-row',event(tid=9,alert=100))
    pool.add('phone',event(tid=9,kind='possible_phone_use',episode='phone',alert=10.1))
    assert group.end_s==14.1 and group.start_s==7
    assert group.metadata()['clip_group_counts']=={'no_helmet':1,'possible_phone_use':1}
    assert group.metadata()['clip_group_people_count']==1
    with pytest.raises(ValueError):pool.add('a',event(episode='other'))


def test_anchor_span_bounds_stop_overlap_chains_and_late_new_episodes_stay_separate():
    pool=GroupPool('job','Camera')
    first=pool.add('a',event(alert=10))
    assert pool.add('b',event(tid=2,episode='b',alert=11.8)) is first
    later=pool.add('c',event(tid=3,episode='c',alert=13.6))
    assert later is not first and later.start_s<first.end_s
    assert all(group.end_s-group.start_s<=10 for group in pool.groups)
    assert pool.reserve(first.id) and not pool.reserve(first.id)
    late=pool.add('late',event(tid=4,episode='late',alert=10.1))
    assert late is not first
    assert pool.add('retry',event(alert=99)) is first
    assert first.end_s==15.8


def test_sources_cameras_and_organizations_cannot_be_joined_and_scene_is_one_event():
    pool=GroupPool('job','Camera','org')
    for conflict in [{'job_id':'another-job'},{'camera':'another-camera'},{'organization_id':'another-org'}]:
        with pytest.raises(ValueError):pool.add('foreign',{**event(),**conflict})
    one=pool.add('scene',event(tid=0,kind='fire',episode='fire'))
    pool.add('scene-alias',event(tid=0,kind='fire',episode='fire'))
    assert one.metadata()['clip_group_counts']=={'fire':1}
    assert one.metadata()['clip_group_scene_events']==1 and one.metadata()['clip_group_people_count']==0
    assert GroupPool('job2','Camera','org').add('scene',event()).id!=one.id


def test_concurrent_workers_join_one_group_and_only_one_encoder_reserves_it():
    pool=GroupPool('job','Camera')
    with ThreadPoolExecutor(max_workers=8) as executor:
        groups=list(executor.map(lambda i:pool.add(str(i),event(tid=i+1,alert=10+i%4*.2,episode=f'ep-{i}')),range(40)))
        reservations=list(executor.map(lambda _:pool.reserve(groups[0].id),range(40)))
    assert len({group.id for group in groups})==1 and sum(reservations)==1
    assert groups[0].metadata()['clip_group_counts']=={'no_helmet':40}
    assert groups[0].metadata()['clip_group_people_count']==40
    assert pool.finish(groups[0].id,'ready',{'clip_path':'media'})
    assert not pool.finish(groups[0].id,'failed')


@pytest.mark.parametrize('alert',[-1,float('nan'),float('inf'),True,None])
def test_invalid_source_times_cannot_create_groups(alert):
    with pytest.raises(ValueError):GroupPool('job','Camera').add('bad',event(alert=0)|{'alert_s':alert})


def test_source_end_clamps_group_excerpt_without_faking_coverage():
    group=GroupPool('job','Camera').add('end',event(alert=1),duration_s=2)
    assert group.start_s==0 and group.end_s==2


@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3')
    db.init(False)
    return tmp_path


def test_atomic_group_storage_preserves_review_and_rejects_cross_source_or_path_escape(store):
    db.add_job('a','a.mp4',{});db.add_job('b','b.mp4',{})
    a=db.add_event('a','Camera',event(),'proof-a.jpg');b=db.add_event('b','Camera',event(),'proof-b.jpg')
    db.review(a,'confirmed')
    with pytest.raises(ValueError):db.set_group_clip('a','Camera',{a:{'clip_group_id':'x'},b:{'clip_group_id':'x'}},clip='runs/a/shared.mp4')
    assert not db.events()[0]['payload'].get('clip_group_id')
    for path in ['runs/a/../../b/shared.mp4','runs/b/shared.mp4']:
        with pytest.raises(ValueError):db.set_group_clip('a','Camera',{a:{}},clip=path)
    with pytest.raises(ValueError):db.set_group_clip('a','Camera',{a:{}},organization_id='foreign')
    db.set_group_clip('a','Camera',{a:{'clip_group_id':'valid','clip_state':'ready'}},clip='runs/a/shared.mp4',organization_id='legacy')
    row=next(row for row in db.events() if row['id']==a)
    assert row['status']=='confirmed' and row['clip']=='runs/a/shared.mp4' and row['image']=='proof-a.jpg'
    token=current_scope.set(Scope('another-organization'))
    try:
        with pytest.raises(ValueError):db.set_group_clip('a','Camera',{a:{'clip_state':'failed'}})
    finally:current_scope.reset(token)


def test_live_simultaneous_workers_export_one_real_clip_and_keep_independent_rows(store,monkeypatch):
    if not live.ffmpeg_path():pytest.skip('FFmpeg required for codec test')
    session=live.LiveSession(RunConfig(camera='Shared camera'),'browser')
    session.first_mono=time.monotonic()-16;session.frame_origin=datetime.now(timezone.utc);session.elapsed=10
    image=np.full((64,96,3),90,np.uint8)
    jpeg=cv2.imencode('.jpg',image)[1].tobytes()
    session.ring.extend((round(i*.04,5),jpeg) for i in range(150,401))
    class Pending:
        def done(self):return False
    submissions=[]
    class Executor:
        def submit(self,fn,clip):submissions.append((fn,clip));return Pending()
    monkeypatch.setattr(live,'CLIPS',Executor())
    events=[event(),event(tid=2,episode='p2',alert=10.4),event(tid=3,kind='possible_phone_use',episode='p3',alert=10.7)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda e:session.record_events([e],image,e['alert_s']),events))
    assert session.events==3 and len(session.pending)==1
    rows=db.events();assert len(rows)==3 and len({row['payload']['clip_group_id'] for row in rows})==1
    db.review(rows[0]['id'],'confirmed')
    session.queue_finished_clips(16);session.queue_finished_clips(16)
    assert len(submissions)==1 and session.pending==[]
    submissions[0][0](submissions[0][1])
    rows=db.events()
    assert len({row['clip'] for row in rows})==1 and all(row['payload']['clip_state']=='ready' for row in rows)
    assert len(list((session.folder/'clips').glob('*.mp4')))==1
    assert sorted(row['payload']['clip_alert_offset_s'] for row in rows)==[3.,3.4,3.7]
    assert all(row['payload']['clip_group_counts']=={'no_helmet':2,'possible_phone_use':1} for row in rows)
    assert all(7<=row['payload']['clip_duration_s']<=10 for row in rows)
    assert sum(row['status']=='confirmed' for row in rows)==1
    # Duplicate worker delivery never creates another event or encoder job.
    session.record_events([events[0]],image,10)
    session.queue_finished_clips(20)
    assert len(db.events())==3 and len(submissions)==1
