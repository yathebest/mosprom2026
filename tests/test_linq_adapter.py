"""Transport/decoder/host-pose regressions, without VPN or remote hardware."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import my_linq
from scripts.repro_linq_adapter import frozen_reply,frozen_pose


@pytest.fixture(autouse=True)
def clean_status(monkeypatch):
    monkeypatch.setattr(my_linq,'_STATUS',{'verified':False,'last_reply_at':None})


class Channel:
    def __init__(self,bad_id=False,timeout=False):self.bad_id=bad_id;self.timeout=timeout
    def queue_declare(self,**kwargs):self.queue=kwargs
    def queue_bind(self,**kwargs):self.bound=kwargs
    def basic_publish(self,**kwargs):self.published=kwargs
    def consume(self,*args,**kwargs):
        while self.timeout:yield None,None,None
        headers=self.published['properties'].headers.copy()
        if self.bad_id:headers['message_id']='another-request'
        yield object(),SimpleNamespace(headers=headers),b'reply'
    def cancel(self):pass


class Connection:
    def __init__(self,channel):self.is_open=True;self.ch=channel
    def channel(self):return self.ch
    def close(self):self.is_open=False


def transport(channel=None,worker=None,data=None):
    import json
    backend=object.__new__(my_linq.LinQBackend)
    backend._connection=Connection(channel or Channel())
    backend._pika=SimpleNamespace(BasicProperties=lambda **k:SimpleNamespace(**k),
                                 exceptions=SimpleNamespace(AMQPError=ConnectionError))
    backend._attrs=('message_id','reply_to','target_uri')
    backend._json=json
    backend._PyTask=dict;backend._serialize=lambda task:b'task'
    backend._deserialize=lambda body:(data if data is not None else dict(zip(my_linq.CUTS,frozen_reply())),
                                     (),worker or {'worker_pid':'test-worker','status_code':0})
    backend.program=my_linq.DEFAULT_PROGRAM;backend.login='test-team'
    backend.program_path='/auto/cache/test-team/'+backend.program
    backend.latencies=[]
    return backend


def test_transport_preserves_uri_headers_and_closes_exclusive_reply_queue():
    channel=Channel();backend=transport(channel);conn=backend._connection
    data,latency=backend._request(np.zeros((1,3,640,640),np.float32))
    assert set(data)==set(my_linq.CUTS) and latency>=0
    assert channel.queue['exclusive'] and channel.queue['auto_delete']
    assert channel.published['properties'].headers['target_uri'].endswith('?v=2&raw=False')
    assert channel.published['routing_key']=='input' and channel.bound['exchange']=='tpu.direct'
    assert not conn.is_open and backend._connection is None
    assert my_linq.get_status()['verified']


def test_reply_for_another_request_is_rejected_and_never_verified():
    backend=transport(Channel(bad_id=True));conn=backend._connection
    with pytest.raises(RuntimeError,match='message ID'):backend._request(np.empty(0))
    assert not conn.is_open and not my_linq.get_status()['verified']


def test_worker_failure_is_not_accepted_even_with_nonempty_tensors():
    backend=transport(worker={'worker_pid':'failed','status_code':2,'error_message':'program failed'})
    with pytest.raises(RuntimeError,match='program failed'):backend._request(np.empty(0))
    assert not my_linq.get_status()['verified']


def test_contract_rejects_missing_wrong_size_nan_and_remote_paths():
    valid=dict(zip(my_linq.CUTS,frozen_reply()))
    for bad in [Path('/remote/result'),{}, {**valid,my_linq.CUTS[0]:np.zeros((1,2,1))},
                {**valid,my_linq.CUTS[0]:np.full((1,2,8400),np.nan)}]:
        with pytest.raises(RuntimeError):my_linq._validated_tensors(bad)


def test_timeout_is_bounded_cleans_connection_and_does_not_fallback(monkeypatch):
    monkeypatch.setattr(my_linq,'TIMEOUT_SECONDS',.001)
    backend=transport(Channel(timeout=True));conn=backend._connection
    with pytest.raises(RuntimeError,match='LinQ TPU'):backend._request(np.empty(0))
    assert not conn.is_open and not my_linq.get_status()['verified']


def raw_backend(pose=True):
    backend=transport()
    backend._connection=None;backend._fall=None;backend._objects=None
    backend._pose=SimpleNamespace(predict=lambda frame:frozen_pose()) if pose else None
    backend._request=lambda tensor:(dict(zip(my_linq.CUTS,frozen_reply())),0.)
    backend.confidence=.4
    return backend


def test_host_pose_restores_track_confirmation_without_inventing_detector_sources():
    from app.inference import PeopleTracker
    frame=np.zeros((240,320,3),np.uint8);backend=raw_backend();tracker=PeopleTracker(5,.4)
    for at in [0,.2,.4]:people=tracker.update(backend.predict(frame),frame.shape,at)
    assert len(people)==1 and people[0].track_id>0 and people[0].alert_eligible
    assert people[0].detection_sources==('person:linq-targeted',)
    assert people[0].keypoints.shape==(17,3)
    unverified=PeopleTracker(5,.4)
    for at in [0,.2,.4]:pending=unverified.update(raw_backend(False).predict(frame),frame.shape,at)
    assert pending[0].track_id==-1 and not pending[0].alert_eligible


def test_linq_pose_allows_measured_hand_phone_rule_to_confirm_once():
    from app.core import RunConfig
    from app.scenario_models import Observation
    from app.scenarios import ScenarioRules
    person=raw_backend().predict(np.zeros((240,320,3),np.uint8))[0];person.track_id=1
    rules=ScenarioRules(RunConfig(phone_detection=True,object_hold=.3))
    phone=Observation((79,121,92,139),.95,'cell_phone','frozen-offline')
    events=[]
    for at in np.arange(0,2,.1):events.extend(rules.step([person],[phone],float(at),320,240,{})[0])
    assert [e['kind'] for e in events]==['possible_phone_use']


def test_host_pose_does_not_add_people_when_tpu_has_no_detections():
    backend=raw_backend();arrays=frozen_reply();arrays[2][:]=0
    backend._request=lambda tensor:(dict(zip(my_linq.CUTS,arrays)),0.)
    assert backend.predict(np.zeros((240,320,3),np.uint8))==[]


def test_adapter_startup_and_status_are_optional_for_other_backends(monkeypatch):
    import sys
    from app.inference import probe_adapter,adapter_status
    monkeypatch.delenv('LINQ_ADAPTER',raising=False)
    assert probe_adapter() is None and not adapter_status()['verified']
    calls=[];module=SimpleNamespace(startup_probe=lambda:calls.append(True),
                                   get_status=lambda:{'verified':True,'mode':'test'})
    monkeypatch.setitem(sys.modules,'test_linq_plugin',module)
    monkeypatch.setenv('LINQ_ADAPTER','test_linq_plugin:create_backend')
    probe_adapter();assert calls==[True] and adapter_status()['mode']=='test'


def test_private_organizer_package_round_trip_when_locally_available():
    wire=pytest.importorskip('tpu_pycloud_format')
    tensor=np.arange(12,dtype=np.float32).reshape(1,3,2,2)
    data,_,worker=wire.deserialize(wire.serialize(wire.PyTask({'images':tensor})))
    np.testing.assert_array_equal(data['images'],tensor)
    assert worker['status_code']==0
