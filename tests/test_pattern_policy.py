"""Production boundary checks with synthetic policy; no inference or weights."""
import copy
import json

import pytest

from app import pattern_policy as policy, server, storage as db, live, video
from app.core import RunConfig
from test_api import client


def test_checked_in_production_policy_removes_only_requested_four_patterns_and_phone_context_option():
    # No temporary permissive manifest: this verifies the actual deployed file.
    selected=policy.catalog()
    assert len(selected['patterns'])==16
    removed={'proximity','harness','load_posture','mask'}
    assert {row['id'] for row in selected['patterns'] if row['enabled']}=={row[0] for row in policy._SPECS}-removed
    assert set(selected['enabled_zone_kinds'])=={'danger_zone','barrier','under_load','under_conveyor',
        'smoking_allowed','fire_allowed','reflection'}
    assert {row['id'] for row in selected['patterns'] if not row['enabled']}==removed
    assert selected['disabled_options']==['phone_work_context_only']
    assert 'phone_work_context_only' in selected['disabled_flags']
    assert 'phone_detection' in selected['enabled_flags']
    assert not policy.sanitize_config({'phone_detection':True,'phone_work_context_only':True})[0]['phone_work_context_only']
    assert not RunConfig().fast_motion


def test_checked_in_remaining_choices_reach_boundary_but_removed_choices_cannot(client,monkeypatch):
    selected=policy.catalog()
    assert sum(row['enabled'] for row in client.get('/api/patterns').json()['patterns'])==12
    calls=[]
    class FakeSession:
        def report(self):return {'id':'availability-only'}
    def create(config,*args):calls.append(config);return FakeSession()
    monkeypatch.setattr(live,'create',create)
    monkeypatch.setattr('app.scenario_models.required_models',lambda config:[])
    for row in selected['patterns']:
        config={'camera':'Restored availability test'}
        if row['config_flag']:config[row['config_flag']]=True
        if row['zone_kinds']:
            config['zones']=[{'name':'Context area','kind':row['zone_kinds'][0],'points':[[0,0],[1,0],[1,1]]}]
        result=client.post('/api/live',json={'source_type':'browser','config':config})
        assert result.status_code==(200 if row['enabled'] else 422),(row['id'],result.text)
    assert len(calls)==12


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    path = tmp_path/'pattern-policy.json'
    monkeypatch.setattr(policy, 'MANIFEST_PATH', path)
    def write(enabled=('helmet', 'danger_zone'), **changes):
        data = {'version': 'qa-policy', 'enabled_patterns': list(enabled),
                'decisions': {'fire': {'reason': 'Rejected by video review', 'evidence_status': 'failed'}}}
        data.update(changes)
        path.write_text(json.dumps(data), encoding='utf-8')
        return path
    return write


def test_scaffold_and_explicit_allowlist_include_both_fall_kinds_and_contexts(manifest):
    manifest(tuple(row[0] for row in policy._SPECS))
    initial = policy.catalog()
    assert len(initial['patterns']) == 16 and all(row['enabled'] for row in initial['patterns'])
    manifest(('fall', 'helmet', 'danger_zone'))
    selected = policy.catalog()
    fall = next(row for row in selected['patterns'] if row['id'] == 'fall')
    assert fall['event_kinds'] == ['possible_fall', 'person_down']
    assert selected['enabled_zone_kinds'] == ['danger_zone', 'reflection']
    assert selected['enabled_flags'] == ['helmet_detection', 'fall_detection']
    manifest(('phone', 'smoking', 'smoke', 'harness'))
    assert set(policy.catalog()['enabled_zone_kinds']) == {'reflection', 'equipment', 'smoking_allowed', 'fire_allowed', 'height'}


def test_missing_manifest_fails_closed_and_motion_requires_explicit_opt_in(client, manifest):
    assert not RunConfig().fast_motion
    for endpoint in ['/api/patterns','/api/defaults','/api/cameras','/api/scenarios','/api/samples']:
        assert client.get(endpoint).status_code==503
    assert client.post('/api/cameras',json={}).status_code==503
    assert client.post('/api/live',json={'source_type':'browser','config':{}}).status_code==503
    db.add_job('missing-policy','source.mp4',{})
    assert client.post('/api/jobs/missing-policy/start',json={}).status_code==503
    manifest(('helmet',))
    assert client.post('/api/cameras',json={'helmet_detection':True}).status_code==200
    assert client.post('/api/cameras',json={'fast_motion':True}).status_code==422


@pytest.mark.parametrize('changes', [
    {'enabled_patterns':['typo']}, {'enabled_patterns':['helmet','helmet']},
    {'enabled_patterns':'helmet'}, {'enabled_patterns':[1]}, {'version':''},
    {'decisions':{'typo':{}}}, {'decisions':{'helmet':False}},
    {'disabled_options':['typo']}, {'disabled_options':'phone_work_context_only'},
    {'disabled_options':[True]}, {'disabled_options':['phone_work_context_only','phone_work_context_only']},
])
def test_invalid_manifest_fails_closed(manifest, changes):
    manifest(**changes)
    with pytest.raises(policy.PolicyError):policy.catalog()


def test_sanitize_preserves_original_and_never_implies_historical_reclassification(manifest):
    manifest()
    config = RunConfig(fast_motion=True, fire_detection=True, fall_detection=True, zones=[
        {'name':'Forbidden','kind':'under_load','points':[[0,0],[1,0],[1,1]]},
        {'name':'Mirror','kind':'reflection','points':[[0,0],[1,0],[1,1]]}],
        state_windows={'fire':2,'possible_fall':3,'no_helmet':4}).model_dump()
    original=copy.deepcopy(config)
    sanitized,warnings=policy.sanitize_config(config)
    assert config == original
    assert not sanitized['fast_motion'] and not sanitized['fire_detection'] and not sanitized['fall_detection']
    assert [zone['kind'] for zone in sanitized['zones']] == ['reflection']
    assert sanitized['state_windows'] == {'no_helmet':4}
    assert {row['setting'] for row in warnings} == {'fast_motion','fire_detection','fall_detection','zones'}
    assert next(row for row in warnings if row['pattern_id']=='fire')['reason']=='Rejected by video review'
    # The model stays usable for historical decoding / independent research.
    assert RunConfig(**original).fall_detection


def test_disabled_phone_context_option_is_sanitized_in_old_templates_and_rejected_on_every_new_boundary(client,manifest,monkeypatch):
    manifest(('phone',),disabled_options=['phone_work_context_only'])
    raw={'camera':'Old phone context','phone_detection':True,'phone_work_context_only':True}
    db.save_camera(raw)
    cleaned=client.get('/api/cameras').json()[0]
    assert cleaned['config']['phone_detection'] and not cleaned['config']['phone_work_context_only']
    assert cleaned['policy_warnings']==[{'pattern_id':'phone_context','setting':'phone_work_context_only',
        'reason':policy._OPTIONS['phone_work_context_only']['reason']}]
    assert db.cameras()[0]['config']==raw
    assert len(client.get('/api/patterns').json()['patterns'])==16
    assert 'equipment' not in client.get('/api/patterns').json()['enabled_zone_kinds']
    monkeypatch.setattr(live,'create',lambda *args:pytest.fail('Disabled option reached worker'))
    db.add_job('phone-context-rejected','source.mp4',{})
    for path,body in [('/api/cameras',raw),('/api/live',{'source_type':'browser','config':raw}),
                      ('/api/jobs/phone-context-rejected/start',raw)]:
        result=client.post(path,json=body)
        assert result.status_code==422,(path,result.text)
        assert result.json()['detail']['disabled_settings'][0]['setting']=='phone_work_context_only'
    # Historical/research interpretation remains possible; ordinary phone is not disabled.
    assert RunConfig(**raw).phone_work_context_only
    monkeypatch.setattr('app.scenario_models.required_models',lambda config:[])
    class FakeSession:
        def report(self):return {'id':'ordinary-phone-allowed'}
    monkeypatch.setattr(live,'create',lambda *args:FakeSession())
    assert client.post('/api/live',json={'source_type':'browser','config':{'phone_detection':True}}).status_code==200


def test_new_activation_and_save_reject_all_disabled_flags_and_zone_types(client, manifest, monkeypatch):
    manifest()
    monkeypatch.setattr(live, 'create', lambda *args: pytest.fail('Disabled request reached live worker'))
    monkeypatch.setattr(server.EXECUTOR, 'submit', lambda *args: pytest.fail('Disabled request reached video worker'))
    db.add_job('never-run', 'source.mp4', {})
    disabled = [row for row in policy.catalog()['patterns'] if not row['enabled']]
    for row in disabled:
        config={'camera':'Policy QA','fast_motion':False}
        if row['config_flag']:
            config[row['config_flag']]=True
        else:
            config['zones']=[{'name':'Disabled','kind':row['zone_kinds'][0],'points':[[0,0],[1,0],[1,1]]}]
        for path, body in [('/api/cameras',config),('/api/jobs/never-run/start',config),
                           ('/api/live',{'source_type':'browser','config':config})]:
            response=client.post(path,json=body)
            assert response.status_code==422,(row['id'],path,response.text)
            assert response.json()['detail']['code']=='disabled_patterns'
    assert db.cameras()==[] and db.job('never-run')['status']=='ready'


def test_readable_old_templates_are_sanitized_but_database_and_history_stay_original(client, manifest):
    manifest()
    raw={'camera':'Legacy','fast_motion':True,'fire_detection':True,'fall_detection':True,
         'zones':[{'name':'Safe mirror','kind':'reflection','points':[[0,0],[1,0],[1,1]]}]}
    db.save_camera(raw)
    db.add_job('old-run','source.mp4',{})
    db.update_job('old-run',config=raw,status='completed')
    db.add_event('old-run','Legacy',{'track_id':1,'kind':'fire','zone':'','start_s':0,'alert_s':1,'sequence':1},'proof.jpg')
    editable=client.get('/api/cameras').json()[0]
    assert editable['policy_version']=='qa-policy' and len(editable['policy_warnings'])==3
    assert not editable['config']['fire_detection'] and not editable['config']['fall_detection']
    assert editable['config']['zones'][0]['kind']=='reflection'
    assert db.cameras()[0]['config']==raw
    assert client.get('/api/jobs').json()[0]['config']==raw
    assert client.get('/api/events').json()[0]['kind']=='fire'
    assert not client.get('/api/defaults').json()['fast_motion']
    exposed=client.get('/api/patterns').json()
    assert exposed['enabled_flags']==['helmet_detection']
    assert [row['flag'] for row in client.get('/api/scenarios').json()['scenarios']]==['helmet_detection']
    assert all(sample['pattern_ids']==['helmet'] for sample in client.get('/api/samples').json())
    assert client.post('/api/samples/fire-demo/import').status_code==422
    assert client.post('/api/samples/fall-01/import').status_code==422


def test_enabled_boundaries_still_work_without_constructing_nn(client, manifest, monkeypatch, tmp_path):
    manifest(('helmet','danger_zone'))
    config={'camera':'Allowed','fast_motion':False,'helmet_detection':True,
            'zones':[{'name':'Area','points':[[0,0],[1,0],[1,1]]}]}
    monkeypatch.setattr('app.scenario_models.required_models',lambda config:[])
    class FakeSession:
        def report(self):return {'id':'fake-allowed'}
    monkeypatch.setattr(live,'create',lambda *args:FakeSession())
    monkeypatch.setattr(video,'MODEL',tmp_path/'model.pt');video.MODEL.write_bytes(b'not loaded')
    monkeypatch.setattr(video,'CANCEL',{})
    scheduled=[];monkeypatch.setattr(server.EXECUTOR,'submit',lambda *args:scheduled.append(args))
    assert client.post('/api/cameras',json=config).status_code==200
    assert client.post('/api/live',json={'source_type':'browser','config':config}).status_code==200
    db.add_job('allowed-run','source.mp4',{})
    assert client.post('/api/jobs/allowed-run/start',json=config).status_code==200
    assert len(scheduled)==1


def test_invalid_policy_returns_service_unavailable_without_fallback_enabling(client, manifest):
    path=manifest();path.write_text('{broken',encoding='utf-8')
    for endpoint in ['/api/patterns','/api/defaults','/api/cameras','/api/scenarios','/api/samples']:
        assert client.get(endpoint).status_code==503
    assert client.post('/api/cameras',json={'fast_motion':False}).status_code==503
