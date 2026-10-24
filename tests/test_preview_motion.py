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
