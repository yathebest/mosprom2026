from datetime import datetime, timezone
import json

import pytest

from app import analytics, storage as db
from app.access import Scope, current_scope

NOW=datetime(2026,10,25,12,tzinfo=timezone.utc)


def job(key='j',camera='Press',**values):
    return {'id':key,'created':'2026-10-25T00:00:00Z','filename':'production.mp4',
        'config':{'camera':camera},'meta':{'duration_s':3600},'stats':{},**values}


def event(key='e',job_id='j',camera='Press',kind='no_helmet',status='new',**values):
    return {'id':key,'job_id':job_id,'camera':camera,'kind':kind,'status':status,'zone':'A',
        'created':'2026-10-25T00:00:00Z','start_s':1,'alert_s':3,'payload':{},'image':'proof.jpg','clip':None,**values}


def test_review_outcomes_are_separate_and_unknown_ends_do_not_become_durations():
    records=[event('closed',status='confirmed',payload={'episode_state':'closed','end_s':7}),
        event('open',payload={'episode_state':'open','observed_until_s':999}),
        event('interrupted',status='forwarded',payload={'episode_state':'interrupted','observed_until_s':999}),
        event('rejected',status='false'),event('invalid',payload={'episode_state':'closed','end_s':-2})]
    report=analytics.aggregate(records,[job()],now=NOW)
    assert report['kpis']=={'automatic_events':5,'pending':3,'confirmed':1,'unresolved_episodes':4,'rejected':1}
    assert report['episodes']['closed_interval_total_s']==6
    assert report['episodes']['closed_count']==1
    assert report['episodes']['censored_count']==4
    assert all(row['duration_s'] is None for row in report['events'] if row['id']!='closed')
    assert report['coverage']['camera_hours'] is None
    assert report['review']['median_first_decision_s'] is None


def test_real_audit_decision_times_only_no_status_based_fabrication():
    records=[event('a',status='confirmed'),event('b',status='false'),event('c',status='confirmed')]
    audits=[{'target':'a','created':'2026-10-25T00:01:00Z','details':'{"status":"forwarded"}'},
        {'target':'a','created':'2026-10-25T00:02:00Z','details':'{"status":"confirmed"}'},
        {'target':'a','created':'2026-10-25T00:09:00Z','details':'{"status":"false"}'},
        {'target':'b','created':'2026-10-25T00:04:00Z','details':'{"status":"false"}'},
        {'target':'c','created':'2026-10-24T00:00:00Z','details':'{"status":"confirmed"}'}]
    report=analytics.aggregate(records,[job()],audit=audits,now=NOW)
    assert report['review']['median_first_decision_s']==180
    assert report['review']['timed_decisions']==2
    assert report['review']['reviewed_without_audit_time']==1


def test_moscow_inclusive_calendar_period_and_processing_vs_source_dates():
    records=[event('before',created='2026-10-23T20:59:59Z'),event('start',created='2026-10-23T21:00:00Z'),
        event('last',created='2026-10-24T20:59:59Z'),event('next',created='2026-10-24T21:00:00Z'),
        event('film',created='2026-10-25T00:00:00Z',payload={'occurred_at':'2026-10-24T10:00:00+03:00'})]
    report=analytics.aggregate(records,[job()],date_from='2026-10-24',date_to='2026-10-24',now=NOW)
    assert {row['id'] for row in report['events']}=={'start','last'}
    assert report['period_start']=='2026-10-23T21:00:00+00:00'
    assert report['period_end']=='2026-10-24T21:00:00+00:00'
    source=analytics.aggregate(records,[job()],date_from='2026-10-24',date_to='2026-10-24',time_basis='recording',now=NOW)
    assert [row['id'] for row in source['events']]==['film']
    assert source['excluded']['unknown_time']==4


@pytest.mark.parametrize('start,end', [('2026-10-26','2026-10-24'),('2026-02-30','2026-10-24'),
    ('2026-10-24T00:00:00','2026-10-25T00:00:00'),('2026-10-24',None)])
def test_invalid_custom_periods_rejected(start,end):
    with pytest.raises(ValueError): analytics.aggregate([],[],date_from=start,date_to=end,now=NOW)


def test_explicit_examples_test_clips_and_experimental_patterns_excluded_without_deletion():
    jobs=[job(),job('sample',meta={'example':True}),job('test',filename='6_te9.mp4')]
    records=[event(),event('demo',job_id='sample'),event('factory-test',job_id='test'),event('phone',kind='possible_phone_use')]
    report=analytics.aggregate(records,jobs,now=NOW)
    assert report['kpis']['automatic_events']==1
    assert report['excluded']=={'examples':2,'experimental':1}
    expanded=analytics.aggregate(records,jobs,include_examples=True,include_experimental=True,now=NOW)
    assert expanded['kpis']['automatic_events']==4
    assert sum(row['educational'] for row in expanded['events'])==2


def test_known_legacy_fixtures_excluded_but_live_purpose_not_guessed_from_camera_name():
    jobs=[job('fixture',filename='no_hardhat-demo.mp4'),job('urfd',filename='fall-02-cam0-rgb.mp4'),
        job('live','LIVE · Demo',filename='Live stream',meta={'mode':'live'}),
        job('declared','Line 1',meta={'source_purpose':'production'})]
    records=[event('a',job_id='fixture'),event('b',job_id='urfd'),
        event('c',job_id='live',camera='LIVE · Demo'),event('d',job_id='declared',camera='Line 1')]
    result=analytics.aggregate(records,jobs,now=NOW)
    assert result['kpis']['automatic_events']==2
    assert result['excluded']['examples']==2
    assert result['data_quality']['source_purpose_events']=={'educational':0,'declared_production':1,'unclassified':1}
    assert result['data_quality']['source_purpose_jobs']['unclassified']==1


def test_only_auditable_coverage_counts_union_overlapping_runs_per_camera():
    def covered(key,camera,start,end):
        return job(key,camera,stats={'analytics_coverage':{'basis':'continuous_observed_source_time',
            'intervals':[{'start_at':start,'end_at':end}]}})
    jobs=[covered('a','A','2026-10-25T00:00:00Z','2026-10-25T01:00:00Z'),
        covered('b','A','2026-10-25T00:30:00Z','2026-10-25T01:30:00Z'),
        covered('c','B','2026-10-25T00:00:00Z','2026-10-25T01:00:00Z')]
    report=analytics.aggregate([],jobs,now=NOW)
    assert report['coverage']['available']
    assert report['coverage']['camera_hours']==2.5
    assert report['coverage']['per_camera_hours']=={'A':1.5,'B':1.0}
    jobs.append(job('missing','B',stats={'current_states_updated_s':3600,'pipeline_wall_s':3600}))
    incomplete=analytics.aggregate([],jobs,now=NOW)
    assert not incomplete['coverage']['available']
    assert incomplete['coverage']['camera_hours'] is None
    assert incomplete['coverage']['per_camera_hours']=={}


def test_aggregate_only_role_has_no_individual_identifiers_or_evidence():
    result=analytics.aggregate([event()],[job()],now=NOW,allow_events=False)
    assert not result['can_drilldown']
    assert result['events']==[]
    assert result['priorities'][0]['event_ids']==[]
    assert 'proof.jpg' not in json.dumps(result)


def test_recurrence_is_per_place_pattern_and_separates_jobs_from_people():
    records=[event(str(i),status='confirmed') for i in range(3)]+[event('next',job_id='j2',status='confirmed')]
    result=analytics.aggregate(records,[job(),job('j2')],now=NOW)
    row=result['priorities'][0]
    assert row['confirmed']==4 and row['distinct_jobs']==2
    assert result['cameras'][0]['confirmed']==4


def test_scoped_history_has_no_100_job_truncation_or_cross_org_camera_leak(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3');db.init(False)
    for organization,camera,key in [('org-a','allowed','a'),('org-a','hidden','b'),('org-b','allowed','c')]:
        token=current_scope.set(Scope(organization))
        try:
            db.save_camera({'camera':camera});db.add_job(key,'production.mp4',{})
            db.update_job(key,config={'camera':camera})
            db.add_event(key,camera,{'track_id':1,'kind':'no_helmet','zone':'z','start_s':0,'alert_s':1,'sequence':1},'proof.jpg')
        finally:current_scope.reset(token)
    token=current_scope.set(Scope('org-a',('allowed',)))
    try:
        for i in range(105):
            db.add_job('extra'+str(i),'production.mp4',{});db.update_job('extra'+str(i),config={'camera':'allowed'})
        events,jobs,cameras,audit=analytics.load_scoped()
        assert len(jobs)==106
        assert [row['id'] for row in jobs if row['id'] in ('a','b','c')]==['a']
        assert len(events)==1 and events[0]['job_id']=='a'
        result=analytics.build_analytics(days=None,allow_events=False)
        assert result['available_cameras']==['allowed']
        assert result['kpis']['automatic_events']==1
    finally:current_scope.reset(token)
