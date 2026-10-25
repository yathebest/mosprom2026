"""Deterministic upload -> LinQ adapter -> tracker/rules -> report/clip check.

Network replies and host pose are frozen test fixtures. This checks application
wiring, NOT accuracy, TPU execution or real network latency. No VPN, stand,
private package, absent work/linq-check files or model downloads are required.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def frozen_reply():
    import numpy as np
    # A known person bbox in a 320x240 input, after the 640 letterbox.
    div=np.zeros((1,2,8400),np.float32)
    sub=np.zeros((1,2,8400),np.float32)
    sig=np.zeros((1,80,8400),np.float32)
    div[0,:,0]=[218/8,330/8]
    sub[0,:,0]=[180/8,380/8]
    sig[0,0,0]=.95
    return div,sub,sig


def frozen_pose():
    import numpy as np
    from app.core import Person
    points=np.zeros((17,3),np.float32)
    for i,(x,y) in {0:(109,50),1:(103,48),2:(115,48),3:(98,52),4:(120,52),
                    5:(86,80),6:(132,80),9:(85,130),10:(132,130),
                    11:(91,151),12:(127,151)}.items():points[i]=[x,y,.95]
    return [Person((64,30,154,220),.95,points)]


def main():
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
    # Supply only an offline import stub; production always uses the real package.
    stub=types.ModuleType('tpu_pycloud_format')
    stub.MESSAGE_ID_ATTR_NAME='message_id';stub.RESPONSE_ROUTING_KEYS_ATTR_NAME='reply_to'
    stub.TARGET_URI_ATTR_NAME='target_uri';stub.PyTask=dict
    stub.serialize=lambda task:b'';stub.deserialize=lambda body:({}, {}, {})
    with patch.dict(os.environ,{'LINQ_ADAPTER':'my_linq:create_backend','LINQ_LOGIN':'repro',
                               'LINQ_FALL_ONNX':'0','SAFETY_ONNX_DEVICE':'cpu'}), \
         patch.dict(sys.modules,{'tpu_pycloud_format':stub}):
        import my_linq
        from app import storage as db,auth
        from app.server import app
        from app.access import Scope,current_scope
        arrays=frozen_reply();reply=dict(zip(my_linq.CUTS,arrays));calls=[]
        def request(self,tensor):
            assert tensor.shape==(1,3,640,640)
            calls.append(tuple(tensor.shape))
            return reply,0.
        class Pose:
            def __init__(self,path):pass
            def predict(self,frame):return frozen_pose()
        with tempfile.TemporaryDirectory(prefix='linq-offline-') as name, \
             patch.object(my_linq.LinQBackend,'_connect',lambda self:None), \
             patch.object(my_linq.LinQBackend,'_request',request), \
             patch.object(my_linq,'_LocalPose',Pose), \
             patch.object(db,'DATA',Path(name)),patch.object(db,'DB',Path(name)/'events.sqlite3'):
            folder=Path(name);source=folder/'input.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),5,(320,240))
            assert writer.isOpened()
            for i in range(20):
                frame=np.full((240,320,3),110,np.uint8)
                cv2.rectangle(frame,(64,30),(154,220),(180,100,80),-1)
                writer.write(frame)
            writer.release()
            with TestClient(app) as client:
                with db.connection() as c:
                    c.execute("INSERT INTO users(id,login,name,password_hash,organization_id,role) "
                              "VALUES('test-admin','test-admin','Offline test',?,'legacy','enterprise_admin')",
                              (auth.password_hash('OfflineTest123!'),))
                r=client.post('/api/auth/login',json={'login':'test-admin','password':'OfflineTest123!'})
                assert r.status_code==200,r.text
                client.headers['X-CSRF-Token']=client.get('/api/auth/me').json()['csrf']
                token=current_scope.set(Scope('legacy'))
                try:
                    r=client.post('/api/jobs',files={'file':('offline.mp4',source.read_bytes(),'video/mp4')})
                    assert r.status_code==200,r.text
                    job=r.json()[0] if isinstance(r.json(),list) else r.json()
                    job_id=job['id']
                    config={'camera':'Offline LinQ check','fall_detection':False,'sample_fps':5,
                        'zone_hold':.2,'zones':[{'name':'Test area','kind':'danger_zone',
                            'points':[[.01,.01],[.99,.01],[.99,.99],[.01,.99]]}]}
                    r=client.post(f'/api/jobs/{job_id}/start',json=config)
                    assert r.status_code==200,r.text
                    deadline=time.monotonic()+60
                    while time.monotonic()<deadline:
                        job=client.get(f'/api/jobs/{job_id}').json()
                        if job['status'] in {'completed','failed','cancelled'}:break
                        time.sleep(.1)
                    assert job['status']=='completed',job
                    rep=client.get(f'/api/jobs/{job_id}/report').json()
                    stats=rep['job']['stats'];events=rep['events']
                    assert 'LinQ TPU' in stats.get('backend',''),stats
                    assert len(events)==1,events
                    event=events[0]
                    assert event['kind']=='danger_zone' and event['track_id']>0,event
                    assert event['clip'] and event['payload']['clip_state']=='ready',event
                    clip=folder/event['clip'];assert clip.is_file()
                    assert client.get('/media/'+event['clip']).status_code==200
                    assert not client.get('/api/health').json()['linq_verified']
                    print(json.dumps({'mode':'mocked transport and frozen host pose; not hardware',
                        'status':job['status'],'backend':stats['backend'],'requests':len(calls),
                        'events':len(events),'clip_ready':True},ensure_ascii=False))
                finally:current_scope.reset(token)
    print('OK: site pipeline works end-to-end with the LinQ adapter (mocked transport)')


if __name__=='__main__':main()
