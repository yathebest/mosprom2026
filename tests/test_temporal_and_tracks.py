import numpy as np
import pytest

from app.core import GroundCalibration, Person, Rules, RunConfig
from app.inference import StableTracks, inverse_rotation


def scene_events(values):
    rules=Rules(RunConfig(fall_detection=True,fast_motion=False))
    events=[]
    for i,value in enumerate(values):
        events.extend(rules.step([],i/10,320,240,{'lying_probability':value})[0])
    return events


def test_scene_fall_is_one_episode_even_if_person_remains_down():
    events=scene_events([.01]*8+[.99]*100)
    assert [e['kind'] for e in events]==['possible_fall']
    assert events[0]['alert_s']==1.3


def test_scene_slow_change_is_down_not_observed_fall():
    events=scene_events([.01]*8+[.5]*20+[.99]*100)
    assert [e['kind'] for e in events]==['person_down']


def test_scene_brief_spike_cannot_confirm():
    assert scene_events([.01]*8+[.99]*2+[.01]*50)==[]


def test_scene_second_fall_after_recovery_rearms():
    events=scene_events([.01]*8+[.99]*15+[.01]*30+[.99]*15)
    assert [e['kind'] for e in events]==['possible_fall']*2
    assert [e['sequence'] for e in events]==[1,2]


def test_metric_motion_uses_floor_calibration():
    calibration=GroundCalibration(image_points=[(0,0),(1,0),(1,1),(0,1)],
                                  world_points=[(0,0),(10,0),(10,10),(0,10)])
    rules=Rules(RunConfig(calibration=calibration,speed_threshold_m_s=1,motion_hold=.5))
    events=[]
    for i in range(30):
        x=i*2
        events.extend(rules.step([Person((x,10,x+10,40),.9,track_id=1)],i/10,100,100)[0])
    assert len(events)==1
    assert events[0]['speed_m_s']==pytest.approx(2,abs=1e-5)


def test_tracker_id_change_after_brief_occlusion_preserves_episode_identity():
    tracker=StableTracks(10)
    first=tracker.update([Person((10,10,40,80),.9,track_id=100)])[0].track_id
    tracker.update([])
    recovered=tracker.update([Person((12,10,42,80),.9,track_id=200)])[0].track_id
    assert recovered==first
    two=tracker.update([Person((12,10,42,80),.9,track_id=200),
                        Person((13,10,43,80),.85,track_id=201)])
    assert len({p.track_id for p in two})==2


def test_tracker_does_not_reidentify_after_long_absence():
    tracker=StableTracks(10)
    first=tracker.update([Person((10,10,40,80),.9,track_id=100)])[0].track_id
    for _ in range(20):tracker.update([])
    assert tracker.update([Person((10,10,40,80),.9,track_id=200)])[0].track_id!=first


def test_rotations_restore_original_image_coordinates():
    points=np.array([[4.,8.],[30.,60.]])
    rotated_90=np.column_stack([points[:,1],99-points[:,0]])
    rotated_270=np.column_stack([79-points[:,1],points[:,0]])
    np.testing.assert_allclose(inverse_rotation(rotated_90,1,100,80),points)
    np.testing.assert_allclose(inverse_rotation(rotated_270,3,100,80),points)
