"""The optimized entry point changes scheduling, never preprocessing or NMS."""
from contextlib import nullcontext
import sys
import threading
from types import SimpleNamespace

import numpy as np

from app.inference import warm_array_predict


class Predictor:
    __module__='ultralytics.models.yolo.detect.predict'
    def __init__(self):
        self.done_warmup=True;self._lock=threading.Lock();self.args=SimpleNamespace(device='cuda:0',iou=.7)
        self.calls=[]
    def setup_source(self,source):
        self.calls.append(('source',source))
        images=source if isinstance(source,list) else [source]
        self.dataset=[([f'array{i}' for i in range(len(images))],images,['']*len(images))]
    def run_callbacks(self,event):self.calls.append(('callback',event))
    def preprocess(self,originals):self.calls.append(('preprocess',originals));return ('tensor',originals)
    def inference(self,tensor):self.calls.append(('forward',tensor));return ('predictions',tensor)
    def postprocess(self,predictions,tensor,originals):
        self.calls.append(('nms',predictions,tensor,originals,self.args.conf,self.args.iou,self.args.classes))
        return [SimpleNamespace(image=image,path=path) for image,path in zip(originals,self.batch[0])]


class Model:
    task='detect';overrides={}
    def __init__(self):self.predictor=Predictor();self.public_calls=[]
    def predict(self,source,**kwargs):self.public_calls.append((source,kwargs));return ['public']


def stub_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules,'ultralytics.cfg',SimpleNamespace(
        get_cfg=lambda old,args:SimpleNamespace(**{**vars(old),**args})))


def test_fast_arrays_use_same_pipeline_order_arguments_and_crop_ownership(monkeypatch):
    stub_dependencies(monkeypatch);model=Model()
    arrays=[np.full((10,12,3),i,np.uint8) for i in (10,20)]
    results=warm_array_predict(model,arrays,{'device':'cuda:0','imgsz':640,'rect':False,'conf':.55,'classes':[0]})
    assert not model.public_calls
    assert all(result.image is original for result,original in zip(results,arrays))
    assert [result.path for result in results]==['array0','array1']
    assert [call[0] for call in model.predictor.calls]==[
        'source','callback','callback','preprocess','forward','nms','callback','callback','callback']
    nms=next(call for call in model.predictor.calls if call[0]=='nms')
    assert nms[-3:]==(.55,.7,[0])
    assert model.predictor.args.imgsz==640 and model.predictor.args.rect is False
    assert model.predictor.seen==2


def test_uninitialized_custom_task_device_and_non_array_inputs_fall_back():
    cases=[]
    m=Model();m.predictor.done_warmup=False;cases.append((m,np.zeros((2,2,3),np.uint8),{}))
    m=Model();m.task='segment';cases.append((m,np.zeros((2,2,3),np.uint8),{}))
    m=Model();cases.append((m,'not-an-array.mp4',{}))
    m=Model();cases.append((m,np.zeros((2,2,3),np.float32),{}))
    m=Model();cases.append((m,np.zeros((2,2,3),np.uint8),{'device':'cpu'}))
    m=Model();m.predictor=None;cases.append((m,np.zeros((2,2,3),np.uint8),{}))
    for model,source,kwargs in cases:
        assert warm_array_predict(model,source,kwargs)==['public']
        assert len(model.public_calls)==1


def test_pipeline_error_releases_predictor_lock_and_does_not_silently_rerun(monkeypatch):
    stub_dependencies(monkeypatch);model=Model()
    def fail(tensor):raise RuntimeError('forward failed')
    model.predictor.inference=fail
    try:warm_array_predict(model,np.zeros((2,2,3),np.uint8),{'device':'cuda:0'})
    except RuntimeError as exc:assert str(exc)=='forward failed'
    else:raise AssertionError('must propagate the model failure')
    assert not model.public_calls and model.predictor._lock.acquire(blocking=False)
    model.predictor._lock.release()
