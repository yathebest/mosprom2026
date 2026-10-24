"""Batching must preserve crop ownership and every native-pixel tile."""
from types import SimpleNamespace

import numpy as np

from app.inference import YoloBackend


class Array:
    def __init__(self,value):self.value=np.asarray(value,np.float32)
    def cpu(self):return self
    def numpy(self):return self.value


class Boxes:
    def __init__(self,rows):self.data=Array(np.asarray(rows,np.float32).reshape(-1,6))
    def __len__(self):return len(self.data.value)


class Result:
    def __init__(self,rows,keypoints=None):
        self.boxes=Boxes(rows)
        self.keypoints=SimpleNamespace(data=Array(keypoints)) if keypoints is not None else None
    def cpu(self):return self


def backend(pose,detector=None):
    instance=YoloBackend.__new__(YoloBackend)
    instance.body=None;instance.model=pose;instance.person_model=detector
    instance.person_models=[detector] if detector else []
    instance.rotated_search=False;instance.imgsz=640;instance.device='cpu'
    instance.confidence=.4;instance.distant_people=True;instance.reflections=[]
    instance.refine_phone_pose=False;instance.refine_head_pose=False;instance.refine_torso_pose=False
    return instance


def coordinate_frame(height,width):
    y,x=np.indices((height,width))
    return np.stack([x,y,np.zeros_like(x)],axis=-1)


def test_batched_pose_refinement_maps_each_output_to_its_own_person():
    boxes=[(50+250*i,30,130+250*i,190) for i in range(4)]
    class Pose:
        def __init__(self):self.crop_batches=[]
        def predict(self,images,**kwargs):
            images=images if isinstance(images,list) else [images]
            if images[0].shape[:2]==(240,1200):
                return [Result([(*box,.9,0) for box in boxes],np.zeros((4,17,3)))]
            self.crop_batches.append(len(images));results=[]
            for image in images:
                x,y=map(int,image[0,0,:2]);box=min(boxes,key=lambda b:abs(b[0]-x-16))
                local=[box[0]-x,box[1]-y,box[2]-x,box[3]-y,.9,0]
                points=np.tile([box[0]+25-x,box[1]+35-y,.9],(17,1))
                # The first pose belongs to a distractor; geometric IoU must
                # select the matching worker independently in each crop.
                distractor=[0,0,5,5,.95,0]
                results.append(Result([distractor,local],np.stack([np.zeros((17,3)),points])))
            return results
    pose=Pose();model=backend(pose);model.refine_phone_pose=True
    people=model.predict_people(coordinate_frame(240,1200))
    assert len(people)==4 and pose.crop_batches==[4]
    for person,box in zip(people,boxes):
        np.testing.assert_allclose(person.box,box)
        np.testing.assert_allclose(person.keypoints[:,:2],np.tile([box[0]+25,box[1]+35],(17,1)))
        np.testing.assert_allclose(person.keypoints[:,2],.9)


def test_tile_chunks_keep_all_origins_edge_filters_and_fp32_resolution():
    class Pose:
        def predict(self,images,**kwargs):return [Result([]) for _ in images]
    class Detector:
        def __init__(self):self.batches=[];self.sizes=[]
        def predict(self,images,**kwargs):
            self.sizes.append(kwargs['imgsz'])
            if not isinstance(images,list):return [Result([])]
            self.batches.append(len(images))
            assert kwargs['rect'] is False and kwargs['imgsz']==640
            return [Result([[20,20,50,120,.9,0], [0,20,5,120,.8,0]]) for _ in images]
    detector=Detector();model=backend(Pose(),detector)
    people=model.predict_people(coordinate_frame(1000,1800))
    expected=[(20+x,20+y,50+x,120+y) for y in [0,296] for x in [0,576,1096]]
    assert sorted(person.box for person in people if person.box[2]-person.box[0]>5)==sorted(expected)
    assert detector.batches==[2,2,2]
    assert detector.sizes[:2]==[960,640]
    # The cut-at-internal-edge proposals are rejected, while the native outer
    # edge proposals remain valid. No crop is accidentally mapped to origin0.
    assert sorted(person.box for person in people if person.box[2]-person.box[0]<=5)==[(0,20,5,120),(0,316,5,416)]


def test_device_lane_is_fifo_and_releases_after_worker_exception():
    import threading
    import time
    from app.scenario_models import FairInferenceLane
    lane=FairInferenceLane();order=[];errors=[]
    def work(name,fail=False):
        try:
            with lane:
                order.append(name)
                if fail:raise ValueError('expected')
        except ValueError as error:
            if not fail:errors.append(error)
        except BaseException as error:errors.append(error)
    def queued(count):
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            with lane.condition:
                if len(lane.queue)==count:return
            time.sleep(.001)
        raise AssertionError('Worker did not enter the admission queue')
    threads=[]
    with lane:
        for i in range(3):
            worker=threading.Thread(target=work,args=(i,i==1),daemon=True)
            worker.start();threads.append(worker);queued(i+2)
        assert order==[]  # Waiting producers cannot overlap the owner.
    for worker in threads:
        worker.join(2);assert not worker.is_alive()
    assert not errors and order==[0,1,2] and not lane.queue
    with lane:order.append(3)
    assert order==[0,1,2,3]
