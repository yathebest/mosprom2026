import csv
from datetime import datetime,timezone
import io
import json
import xml.etree.ElementTree as ET
import zipfile

import pytest
from fastapi.testclient import TestClient
from app import analytics,analytics_export as export,auth,storage as db
from app.server import app

NOW=datetime(2026,10,25,10,tzinfo=timezone.utc)


def report(allow=True):
    jobs=[{'id':'job','filename':'shift.mp4','config':{'camera':'Линия 1'},'meta':{},'stats':{},'created':'2026-10-24T21:00:00Z'}]
    events=[{'id':'event','job_id':'job','track_id':7,'kind':'no_helmet','camera':'Линия 1','zone':'=SUM(1,2)',
             'status':'new','created':'2026-10-24T21:30:00Z','start_s':0.5,'alert_s':2.,'image':'proof.jpg','clip':'proof.mp4',
             'payload':{'episode_state':'interrupted','observed_until_s':5,'occurred_at':'2026-10-24T21:29:00Z'}}]
    return analytics.aggregate(events,jobs,days=None,date_from='2026-10-25',date_to='2026-10-25',now=NOW,allow_events=allow)


def cells(blob,sheet=1):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        root=ET.fromstring(z.read(f'xl/worksheets/sheet{sheet}.xml'))
        return {c.attrib['r']:c for c in root.iter(export.Q('c'))}


def text(cell):
    return ''.join(cell.itertext())


def test_xlsx_actual_dates_typed_counts_unresolved_and_literal_input():
    r=report(); r['events'][0]['track_id']=7
    blob=export.xlsx_bytes(r); assert blob[:2]==b'PK'
    summary=cells(blob); detail=cells(blob,2)
    assert summary['C10'].get('t') is None and text(summary['C10'])=='1'
    assert detail['A10'].get('t') is None
    assert float(text(detail['A10']))==pytest.approx(export.excel_date('2026-10-24T21:30:00Z'))
    assert text(detail['E10'])=='=SUM(1,2)' and detail['E10'].get('t')=='inlineStr'
    assert detail['E10'].find(export.Q('f')) is None
    assert not text(detail['J10']) and not text(detail['K10'])
    assert text(detail['L10'])=='Наблюдение прервано'
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        style=z.read('xl/styles.xml').decode()
        assert 'dd.mm.yyyy' in style and '[h]:mm:ss.0' in style
        table=ET.fromstring(z.read('xl/tables/table2.xml'))
        assert table.get('ref')=='A9:N10'


def test_xlsx_empty_and_aggregate_only_do_not_leak_detail():
    r=report(False); blob=export.xlsx_bytes(r); detail=cells(blob,2)
    assert not text(detail['A10']) and not text(detail['C10'])
    assert 'недоступны' in text(detail['B7'])
    assert b'proof.jpg' not in blob
    r=report();r['events']=[]
    assert 'событий нет' in text(cells(export.xlsx_bytes(r),2)['B7'])


def test_xlsx_short_proof_links_keep_full_urls_without_formulas():
    r=report();r['events'][0].update(image='http://127.0.0.1:8765/media/runs/id/evidence.jpg',clip='http://127.0.0.1:8765/media/runs/id/clip.mp4')
    blob=export.xlsx_bytes(r);detail=cells(blob,2)
    assert text(detail['M10'])=='Открыть кадр' and text(detail['N10'])=='Открыть клип'
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        relations=ET.fromstring(z.read('xl/worksheets/_rels/sheet2.xml.rels'))
        links=[e for e in relations if e.get('Type').endswith('/hyperlink')]
        assert len(links)==2 and links[0].get('Target')==r['events'][0]['image']
        assert all(e.get('TargetMode')=='External' for e in links)
        sheet=ET.fromstring(z.read('xl/worksheets/sheet2.xml'))
        assert len(sheet.find(export.Q('hyperlinks')))==2


def test_csv_moscow_human_time_formula_injection_and_unknown_end():
    rows=list(csv.reader(io.StringIO(export.csv_bytes(report()).decode('utf-8-sig')),delimiter=';'))
    detail=rows[-1]
    assert detail[0]=='25.10.2026, 00:30:00'
    assert detail[4]=="'=SUM(1,2)"
    assert detail[7:9]==['0:00.5','0:02.0']
    assert detail[9:11]==['Неизвестно','Неизвестно']
    assert export.duration(135)=='2 мин 15 с'


@pytest.mark.parametrize('seconds,expected',[(0,'0:00.0'),(.005,'0:00.005'),(.523,'0:00.523'),
    (59.9996,'1:00.0'),(3723.125,'1:02:03.125'),(None,'Неизвестно')])
def test_source_clock_preserves_fraction_without_zero_for_unknown(seconds,expected):
    assert export.video_clock(seconds)==expected


def test_period_metadata_preserves_rolling_instants_custom_days_and_unknown_source_purpose():
    r=report(); meta=dict(export.metadata(r))
    assert meta['Период']=='25.10.2026 — 25.10.2026 включительно · МСК (UTC+3)'
    assert 'Без отметки назначения: 1' in meta['Отбор']
    rolling=analytics.aggregate([],[],days=7,now=NOW)
    assert dict(export.metadata(rolling))['Период']=='18.10.2026 13:00:00 — 25.10.2026 13:00:00 (конец не включён; МСК UTC+3)'
    explicit=analytics.aggregate([],[],date_from='2026-10-24T14:15:16+03:00',date_to='2026-10-24T15:16:17+03:00',now=NOW)
    assert '24.10.2026 14:15:16 — 24.10.2026 15:16:17' in dict(export.metadata(explicit))['Период']
    all_time=analytics.aggregate([],[],days=None,now=NOW)
    assert dict(export.metadata(all_time))['Период']=='Всё время · по состоянию на 25.10.2026 13:00:00 · МСК (UTC+3)'


@pytest.mark.parametrize('value',['=1+1','+SUM(1,2)','-1+2','@SUM(1,2)','\t=1','\r@SUM(1,2)','\n+1','\x00=1',' \x00=1'])
def test_csv_formula_prefixes_after_control_character_sanitization(value):
    r=report();r['events'][0]['zone']=value
    rows=list(csv.reader(io.StringIO(export.csv_bytes(r).decode('utf-8-sig')),delimiter=';'))
    assert rows[-1][4].startswith("'")
    assert '\x00' not in rows[-1][4]


@pytest.fixture
def system(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3')
    with TestClient(app) as client:
        encoded=auth.password_hash('TestAnalyticsPassword!42')
        with db.connection() as c:
            for role in ('enterprise_admin','management','operator'):
                c.execute('INSERT INTO users(id,login,name,password_hash,organization_id,role) VALUES(?,?,?,?,?,?)',
                          (role,role,role,encoded,'legacy',role))
            c.execute("INSERT INTO jobs(id,filename,status,config,meta,organization_id) VALUES('j','shift.mp4','completed',?, '{}','legacy')",(json.dumps({'camera':'Line'}),))
            c.execute("INSERT INTO events(id,job_id,track_id,kind,camera,start_s,alert_s,payload,status) VALUES('e','j',7,'no_helmet','Line',0,1,?,'new')",(json.dumps({'episode_state':'interrupted'}),))
        yield client


def login(c,role):
    c.cookies.clear()
    assert c.post('/api/auth/login',json={'login':role,'password':'TestAnalyticsPassword!42'}).status_code==200


def test_download_roles_dates_and_aggregate_management(system):
    c=system; assert c.get('/api/analytics/export').status_code==401
    login(c,'operator'); assert c.get('/api/analytics/export').status_code==403
    login(c,'management'); result=c.get('/api/analytics/export?days=all')
    assert result.status_code==200 and 'attachment' in result.headers['content-disposition']
    assert not text(cells(result.content,2)['C10'])
    login(c,'enterprise_admin'); result=c.get('/api/analytics/export?days=all')
    assert result.status_code==200 and text(cells(result.content,2)['C10'])=='Line'
    assert c.get('/api/analytics/export?from=2026-10-25&to=2026-10-24').status_code==422
    assert c.get('/api/analytics/export?format=exe').status_code==422
    assert c.get('/api/analytics/export?from=2026-10-25').status_code==422


def test_export_preserves_org_and_camera_scope_for_aggregate_and_evidence(system):
    c=system
    with db.connection() as conn:
        conn.execute("INSERT INTO jobs(id,filename,status,config,meta,organization_id) VALUES('other','shift.mp4','completed',?, '{}','another-org')",(json.dumps({'camera':'OtherTenant'}),))
        conn.execute("INSERT INTO jobs(id,filename,status,config,meta,organization_id) VALUES('hidden','shift.mp4','completed',?, '{}','legacy')",(json.dumps({'camera':'HiddenCamera'}),))
        for key,camera in [('other','OtherTenant'),('hidden','HiddenCamera')]:
            conn.execute("INSERT INTO events(id,job_id,track_id,kind,camera,start_s,alert_s,payload,status,image) VALUES(?,?,7,'no_helmet',?,0,1,?,'new',?)",(key,key,camera,'{}','runs/'+key+'/secret.jpg'))
        conn.execute("UPDATE users SET camera_scope=? WHERE id IN ('enterprise_admin','management')",(json.dumps(['Line']),))
    for role in ('management','enterprise_admin'):
        login(c,role);response=c.get('/api/analytics/export?days=all&format=csv')
        assert response.status_code==200
        text_value=response.content.decode('utf-8-sig')
        assert 'OtherTenant' not in text_value and 'HiddenCamera' not in text_value and 'secret.jpg' not in text_value
        assert 'Line' in text_value
        if role=='management':assert 'Человек в сеансе' not in text_value
