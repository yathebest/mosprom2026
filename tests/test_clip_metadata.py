import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app import clip_metadata as media, storage as db,live
from scripts.backfill_clip_metadata import backfill


def make_video(path,frames=30,fps=25):
    path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(96,64))
    assert writer.isOpened()
    for i in range(frames):writer.write(np.full((64,96,3),i%255,np.uint8))
    writer.release()
    return path


def test_actual_mp4_duration_and_changed_file_cache(tmp_path):
    path=make_video(tmp_path/'runs/j/clip.mp4')
    result=media.probe_clip(path,data_dir=tmp_path)
    assert result['clip_duration_s']==pytest.approx(1.2,abs=.001)
    assert result['clip_duration_source']=='mp4_mvhd'
    assert result['clip_file_bytes']==path.stat().st_size
    assert media.cached_clip_metadata(result,path,data_dir=tmp_path)
    make_video(path,frames=50)
    assert not media.cached_clip_metadata(result,path,data_dir=tmp_path)
    assert media.probe_clip(path,data_dir=tmp_path)['clip_duration_s']==pytest.approx(2,abs=.001)


def test_missing_corrupt_outside_and_permission_denied_media_have_no_fabricated_duration(tmp_path,monkeypatch):
    path=tmp_path/'runs/j/clip.mp4';path.parent.mkdir(parents=True)
    assert media.probe_clip(path,data_dir=tmp_path) is None
    path.write_bytes(b'not a valid MP4 file')
    monkeypatch.setattr(media.shutil,'which',lambda *args:None)
    assert media.probe_clip(path,data_dir=tmp_path) is None
    outside=make_video(tmp_path/'outside.mp4')
    assert media.probe_clip(outside,data_dir=tmp_path) is None
    def denied(*args):raise PermissionError('test denial')
    monkeypatch.setattr(media,'movie_duration',denied)
    assert media.probe_clip(path,data_dir=tmp_path) is None


@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3');db.init(False)
    db.add_job('j','fixture.mp4',{});db.update_job('j',config={'camera':'Camera'},stats={'kept':True})
    payload={'track_id':1,'kind':'no_helmet','zone':'z','start_s':3,'alert_s':5,'sequence':1,
        'episode_id':'ep','episode_state':'interrupted','end_s':None,'observed_until_s':99,'closure_reason':'video_ended',
        'clip_state':'ready','clip_start_s':100,'clip_end_s':107,'clip_alert_offset_s':3,'reviewed_by':'Operator'}
    eid=db.add_event('j','Camera',payload,'runs/j/evidence.jpg');db.review(eid,'confirmed')
    make_video(tmp_path/'runs/j/clip.mp4');db.set_clip(eid,'runs/j/clip.mp4')
    with db.connection() as conn:
        conn.execute('CREATE TABLE audit (id INTEGER PRIMARY KEY, action TEXT, details TEXT)')
        conn.execute("INSERT INTO audit VALUES(1,'event.reviewed','unchanged')")
    return tmp_path,eid


def test_backfill_is_dry_by_default_online_backed_up_idempotent_and_preserves_incident_review_and_audit(store):
    root,eid=store;before=db.events()[0];job_before=db.job('j')
    dry=backfill(db.DB,root)
    assert dry['would_update']==1 and dry['updated']==0 and dry['backup'] is None
    assert db.events()[0]==before
    result=backfill(db.DB,root,apply=True)
    assert result['updated']==1 and Path(result['backup']).is_file()
    after=db.events()[0]
    assert {k:v for k,v in before.items() if k!='payload'}=={k:v for k,v in after.items() if k!='payload'}
    assert all(after['payload'][key]==value for key,value in before['payload'].items())
    assert set(after['payload'])-set(before['payload'])==set(media.FIELDS)
    assert after['payload']['clip_duration_s']==pytest.approx(1.2,abs=.001)
    assert after['payload']['clip_start_s']==100 and after['payload']['clip_end_s']==107
    assert after['payload']['end_s'] is None and after['status']=='confirmed'
    assert db.job('j')==job_before
    with db.connection() as conn:assert tuple(conn.execute('SELECT * FROM audit').fetchone())==(1,'event.reviewed','unchanged')
    with sqlite3.connect(result['backup']) as conn:
        assert 'clip_duration_s' not in json.loads(conn.execute('SELECT payload FROM events WHERE id=?',(eid,)).fetchone()[0])
    again=backfill(db.DB,root,apply=True)
    assert again['updated']==0 and again['would_update']==0 and again['backup'] is None
    assert again['counts']['skipped_cached']==1


def test_backfill_skips_unreadable_and_traversal_paths_without_mutating_rows(store):
    root,eid=store;db.set_clip(eid,'../../outside.mp4');before=db.events()[0]
    result=backfill(db.DB,root,apply=True)
    assert result['counts']['skipped_invalid_path']==1 and not result['updated']
    assert db.events()[0]==before
    db.set_clip(eid,'runs/j/missing.mp4');before=db.events()[0]
    result=backfill(db.DB,root,apply=True)
    assert result['counts']['skipped_unreadable']==1 and db.events()[0]==before


def test_real_live_clip_export_saves_actual_duration_without_closing_episode(store):
    root,eid=store
    from app.video import ffmpeg_path
    if not ffmpeg_path():pytest.skip('H264 encoder unavailable')
    folder=root/'runs/j';(folder/'clips').mkdir()
    image=cv2.imencode('.jpg',np.full((64,96,3),100,np.uint8))[1].tobytes()
    clip={'id':eid,'frames':[(i/10,image) for i in range(5)],'until':.4,'alert_s':.1}
    live.LiveSession.finalize_clip(SimpleNamespace(id='j',folder=folder),clip)
    event=db.events()[0];payload=event['payload']
    assert payload['clip_state']=='ready' and payload['clip_duration_s']>0
    assert payload['clip_end_s']==.4 and payload['clip_start_s']==0
    assert payload['clip_duration_s']!=payload['clip_end_s']-payload['clip_start_s']
    assert payload['episode_state']=='interrupted' and payload['end_s'] is None
    assert event['status']=='confirmed'
