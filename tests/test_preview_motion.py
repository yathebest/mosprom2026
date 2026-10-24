import cv2
import numpy as np
from app.core import Person
from app.preview_motion import PreviewMotion


def test_preview_projects_moving_pixels_without_turning_them_into_measurements():
    rng=np.random.default_rng(42)
    gray=np.zeros((180,300),np.uint8);gray[40:140,50:110]=rng.integers(0,255,(100,60),dtype=np.uint8)
    person=Person((50,40,110,140),.9,track_id=7)
    visual={'time':0.,'width':300,'gray':gray,'people':[person]}
    shifted=cv2.warpAffine(gray,np.float32([[1,0,25],[0,1,5]]),(300,180))
    motion=PreviewMotion();projected=motion.project(visual,cv2.cvtColor(shifted,cv2.COLOR_GRAY2BGR),.8)
    assert len(projected)==1
    assert np.allclose(projected[0].box,(75,45,135,145),atol=2)
    assert not projected[0].observed and not projected[0].alert_eligible
    assert person.observed and person.box==(50,40,110,140)
    assert motion.project(visual,cv2.cvtColor(shifted,cv2.COLOR_GRAY2BGR),2.6)==[]


def test_preview_hides_boxes_when_texture_is_lost():
    gray=np.random.default_rng(1).integers(0,255,(180,300),dtype=np.uint8)
    visual={'time':0.,'width':300,'gray':gray,'people':[Person((50,40,110,140),.9)]}
    assert PreviewMotion().project(visual,np.zeros((180,300,3),np.uint8),.5)==[]


def test_reduced_flow_preserves_preview_coordinates_keypoints_and_roi_cost(monkeypatch):
    rng=np.random.default_rng(10)
    gray=np.zeros((540,960),np.uint8)
    gray[100:380,200:360]=rng.integers(0,255,(280,160),dtype=np.uint8)
    points=np.tile([500.,300.,.9],(17,1))
    person=Person((400,200,720,760),.9,points,track_id=3)
    visual={'time':0.,'width':1920,'gray':gray,'people':[person]}
    shifted=cv2.warpAffine(gray,np.float32([[1,0,9],[0,1,6]]),(960,540))
    corner_shapes=[];flow_shapes=[]
    corners=cv2.goodFeaturesToTrack;flow=cv2.calcOpticalFlowPyrLK
    def observe_corners(image,*args,**kwargs):
        corner_shapes.append(image.shape);return corners(image,*args,**kwargs)
    def observe_flow(left,right,*args,**kwargs):
        flow_shapes.append((left.shape,right.shape));return flow(left,right,*args,**kwargs)
    monkeypatch.setattr(cv2,'goodFeaturesToTrack',observe_corners)
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',observe_flow)
    projected=PreviewMotion().project(visual,cv2.cvtColor(shifted,cv2.COLOR_GRAY2BGR),.1)
    assert len(projected)==1 and projected[0].track_id==3
    np.testing.assert_allclose(projected[0].box,(209,106,369,386),atol=1.5)
    np.testing.assert_allclose(projected[0].keypoints[:,:2],np.tile([259,156],(17,1)),atol=1.5)
    assert all(max(left)<=640 and left==right for left,right in flow_shapes)
    assert corner_shapes and all(h<360 and w<640 for h,w in corner_shapes)
    assert not projected[0].observed and not projected[0].alert_eligible
    np.testing.assert_allclose(person.keypoints,points)
    assert person.box==(400,200,720,760) and visual['gray'].shape==(540,960)


def test_separate_people_keep_opposite_motion_and_unknown_evidence_flags():
    rng=np.random.default_rng(12)
    gray=np.zeros((540,960),np.uint8);following=np.zeros_like(gray)
    people=[Person((100,100,220,340),.9,track_id=1),Person((600,100,720,340),.9,track_id=2)]
    for person,(dx,dy) in zip(people,[(9,6),(-12,3)]):
        a,b,c,d=map(int,person.box);texture=rng.integers(0,255,(d-b,c-a),dtype=np.uint8)
        gray[b:d,a:c]=texture;following[b+dy:d+dy,a+dx:c+dx]=texture
    visual={'time':0.,'width':960,'gray':gray,'people':people}
    projected=PreviewMotion().project(visual,cv2.cvtColor(following,cv2.COLOR_GRAY2BGR),.1)
    assert len(projected)==2
    for p,expected in zip(projected,[(109,106,229,346),(588,103,708,343)]):
        np.testing.assert_allclose(p.box,expected,atol=1.5)
        assert not p.observed and not p.alert_eligible
