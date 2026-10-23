from datetime import datetime,timezone
import io
import json
import zipfile

from app.operations import triage,review_archive
from app import storage as db
from app.streams import StreamRequest
import pytest
from pydantic import ValidationError


def test_triage_prioritizes_urgent_and_uses_waiting_age():
    base={'created':'2026-10-23T10:00:00Z','status':'new','payload':{}}
    rows=triage([{**base,'kind':'danger_zone'},
                 {**base,'kind':'possible_fall'},
                 {**base,'kind':'possible_fall','status':'confirmed'}],
                datetime(2026,10,23,10,2,tzinfo=timezone.utc))
    assert len(rows)==2
    assert rows[0]['priority']=='high' and rows[0]['overdue']
    assert rows[1]['priority']=='medium' and not rows[1]['overdue']


def test_review_export_keeps_review_label_and_requires_evidence_inside_runs(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path)
    monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3')
    db.init();db.add_job('a','test.mp4',{})
    event={'track_id':1,'kind':'danger_zone','zone':'x','start_s':0,'alert_s':1,'sequence':1}
    eid=db.add_event('a','cam',event,'../private.jpg')
    db.review(eid,'false')
    with zipfile.ZipFile(io.BytesIO(review_archive())) as archive:
        records=json.loads(archive.read('annotations.json'))
        assert records[0]['supervision']=='operator_rejected_alert'
        assert 'evidence' not in records[0]
        assert archive.namelist()==['annotations.json','README.txt']


@pytest.mark.parametrize('url',['file:///private/video.mp4','ftp://x/video','garbage'])
def test_live_source_rejects_non_video_protocols(url):
    with pytest.raises(ValidationError):StreamRequest(url=url)
