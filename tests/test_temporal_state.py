from app.temporal_state import TemporalGate


def test_time_weighted_majority_and_three_distinct_samples():
    gate=TemporalGate()
    for t,value in [(0,True),(.5,False),(1,True),(1.5,True)]:assert gate.update(t,value,2,2,.8) is None
    assert gate.update(2,True,2,2,.8)==(0,1)
    assert gate.state=='violation' and gate.confirmation()['agreement']==.75
    assert gate.update(2,True,2,2,.8) is None


def test_unknown_and_long_gap_cannot_confirm_or_close():
    gate=TemporalGate()
    for t in [0,.5,1,1.5,2]:gate.update(t,True,2,2,.8)
    episode=gate.episode_id
    for t in [2.5,3,4,8]:assert gate.update(t,None,2,2,.8) is None
    assert gate.active and gate.episode_id==episode
    closed=[]
    for t in [8.1,8.6,9.1,9.6,10.1]:
        gate.update(t,False,2,2,.8)
        if gate.last_transition:closed.append(gate.last_transition)
    assert gate.state=='clear' and closed[0]['episode_id']==episode
    assert closed[0]['end_s']==8.1


def test_sparse_bursts_do_not_vote_across_missing_time():
    gate=TemporalGate()
    for t in [0,.1,2,2.1]:assert gate.update(t,True,2,2,.8) is None
    assert gate.state=='unknown'


def test_slow_live_expands_window_without_duplicating_votes():
    gate=TemporalGate()
    assert gate.update(0,True,1.5,2,2.5) is None
    assert gate.update(1,True,1.5,2,2.5) is None
    assert gate.update(2,True,1.5,2,2.5)==(0,1)
    assert gate.last_transition['state_confirmation']['window_s']==2
    assert gate.last_transition['state_confirmation']['nominal_window_s']==1.5


def test_clear_snapshot_uses_clear_window_and_unknown_is_not_stale_full_agreement():
    gate=TemporalGate()
    for t in [0,.5,1,1.5,2,2.5,3]:gate.update(t,False,1,3,.8)
    row=gate.snapshot()
    assert row['window_s']==3 and row['state_confirmation']['window_s']==3
    assert row['state_confirmation']['nominal_window_s']==3
    gate.update(3.1,None,1,3,.8)
    assert gate.state=='clear' and gate.snapshot()['observed_state']=='unknown'
    assert gate.snapshot()['agreement']==0 and gate.snapshot()['state_confirmation']['samples']==0


def test_rules_emit_once_close_then_new_episode_and_never_close_on_prediction():
    from app.core import Person,Rules,RunConfig,Zone
    zone=Zone(points=[(0,0),(1,0),(1,1),(0,1)])
    rules=Rules(RunConfig(zones=[zone],fast_motion=False,zone_hold=1,release_hold=1))
    inside=Person((10,0,50,90),.9,track_id=1);outside=Person((110,0,150,90),.9,track_id=1)
    events=[]
    for t in [0,.5,1]:events+=rules.step([inside],t,100,100)[0]
    assert len(events)==1 and events[0]['episode_state']=='open'
    episode=events[0]['episode_id']
    predicted=Person(inside.box,.9,track_id=1,observed=False,alert_eligible=False)
    rules.step([predicted],1.5,100,100)
    assert not rules.event_updates and rules.state_rows[1]['danger_zone:Опасная зона']['state']=='violation'
    patches=[]
    for t in [2,2.5,3]:
        rules.step([outside],t,100,100);patches+=rules.event_updates
    assert len(patches)==1 and patches[0]['episode_id']==episode and patches[0]['end_s']==2
    assert patches[0]['closed_at_s']>=patches[0]['end_s']
    for t in [3.5,4,4.5]:events+=rules.step([inside],t,100,100)[0]
    assert len(events)==2 and events[-1]['episode_id']!=episode
    interrupted=rules.finish(5,'source_lost')
    assert len(interrupted)==1 and interrupted[0]['end_s'] is None
    assert interrupted[0]['episode_state']=='interrupted'


def test_id_handoff_is_mutually_unambiguous_and_preserves_episode():
    from app.core import Person,RunConfig
    from app.temporal_state import PatternStates
    cfg=RunConfig(fast_motion=False);machine=PatternStates(cfg)
    p=Person((10,0,60,180),.9,track_id=1)
    for t in [0,.5,1]:
        machine.begin([p],t);machine.observe(1,'no_helmet','',True,1,2,.8);machine.end(t)
    episode=machine.gates[(1,'no_helmet','')].episode_id
    new=Person(p.box,.9,track_id=2)
    assert machine.begin([new],1.5)=={2:1}
    assert machine.gates[(2,'no_helmet','')].episode_id==episode
    # Two new nearby IDs cannot greedily steal the old episode.
    ambiguous=PatternStates(cfg);ambiguous.boxes[1]=(1.,p.box)
    neighbour=Person((11,0,61,180),.9,track_id=3)
    assert ambiguous.begin([new,neighbour],1.5)=={}


def test_rgb_fall_episode_stays_scene_scoped_as_person_cardinality_changes():
    from app.core import Person,Rules,RunConfig
    high={'lying_probability':.99,'fall_probability':.99,'fall_threshold':.5}
    cfg=RunConfig(fast_motion=False,fall_detection=True,lying_hold=1)
    p=Person((10,0,60,180),.9,track_id=1);q=Person((80,0,130,180),.9,track_id=2)
    for initial,later in [([], [p]),([p],[p,q])]:
        rules=Rules(cfg);events=[]
        for t in [0,.5,1]:events+=rules.step(initial,t,200,200,high)[0]
        for t in [1.5,2,2.5,3]:events+=rules.step(later,t,200,200,high)[0]
        assert len(events)==1 and events[0]['track_id']==0 and events[0]['scope']=='scene'
        assert events[0]['kind']=='person_down'  # No prior confirmed standing.


def test_rgb_transition_unknown_preserves_confirmed_standing_across_detector_dropouts():
    from app.core import Person,Rules,RunConfig
    rules=Rules(RunConfig(fast_motion=False,fall_detection=True))
    person=Person((10,0,60,180),.9,track_id=1)
    predicted=Person(person.box,.9,track_id=1,observed=False,alert_eligible=False)
    events=[]
    for i in range(35):
        events+=rules.step([person],i/10,320,240,{'lying_probability':.01})[0]
    assert rules.state_rows[0]['person_down']['state']=='clear'
    events+=rules.step([person],3.5,320,240,{'lying_probability':.66})[0]
    assert rules.state_rows[0]['person_down']['state']=='clear'
    assert rules.state_rows[0]['person_down']['observed_state']=='unknown'
    for i in range(36,48):
        people=[person] if i%2==0 else [predicted]
        events+=rules.step(people,i/10,320,240,{'lying_probability':.99})[0]
    assert len(events)==1 and events[0]['kind']=='possible_fall'
    assert events[0]['track_id']==0 and events[0]['scope']=='scene'
    assert events[0]['start_s']==3.6 and events[0]['alert_s']==4.1
    assert not any(key[0] for key in rules.gates if key[1] in ('possible_fall','person_down'))


def test_pose_unknown_transition_keeps_previous_confirmed_standing():
    import numpy as np
    from app.core import Person,Rules,RunConfig
    def person(lying=None):
        points=np.zeros((17,3),np.float32)
        if lying is not None:
            points[[5,6]]=[[30,125,.9],[30,135,.9]] if lying else [[25,45,.9],[35,45,.9]]
            points[[11,12]]=[[110,125,.9],[110,135,.9]] if lying else [[25,110,.9],[35,110,.9]]
        return Person((10,100,190,160) if lying else (10,10,50,190),.9,track_id=1,keypoints=points)
    rules=Rules(RunConfig(fast_motion=False,fall_detection=True));events=[]
    for i in range(35):events+=rules.step([person(False)],i/10,320,240)[0]
    events+=rules.step([person()],3.5,320,240)[0]
    for i in range(36,49):events+=rules.step([person(True)],i/10,320,240)[0]
    assert len(events)==1 and events[0]['kind']=='possible_fall'
    assert events[0]['scope']=='person' and events[0]['track_id']==1
    assert events[0]['start_s']==3.6 and events[0]['alert_s']>=4.6


def test_body_recovery_keeps_same_gate_and_rearms_scene_and_pose_falls():
    import numpy as np
    from app.core import Person,Rules,RunConfig
    def person(lying):
        points=np.zeros((17,3),np.float32)
        points[[5,6]]=[[30,125,.9],[30,135,.9]] if lying else [[25,45,.9],[35,45,.9]]
        points[[11,12]]=[[110,125,.9],[110,135,.9]] if lying else [[25,110,.9],[35,110,.9]]
        return Person((10,100,190,160) if lying else (10,10,50,190),.9,track_id=1,keypoints=points)
    for scene in [False,True]:
        rules=Rules(RunConfig(fast_motion=False,fall_detection=True,fall_hold=.5))
        events=[];closed=[]
        values=[False]*25+[True]*15+[False]*30+[True]*15
        for i,lying in enumerate(values):
            people=[] if scene else [person(lying)]
            rgb={'lying_probability':.99 if lying else .01} if scene else None
            events+=rules.step(people,i/10,320,240,rgb)[0]
            closed+=rules.event_updates
            if i==69:
                gate=rules.gates[(0 if scene else 1,'possible_fall','')]
                assert gate.state=='clear' and gate.observed_state=='clear'
                assert gate.agreement>=.75
        assert [event['kind'] for event in events]==['possible_fall','possible_fall']
        assert [event['sequence'] for event in events]==[1,2]
        assert len(closed)==1 and closed[0]['episode_id']==events[0]['episode_id']
        assert events[1]['episode_id']!=events[0]['episode_id']


def test_phone_single_hidden_hand_cannot_close_and_small_smoking_is_unknown():
    import numpy as np
    from app.core import Person,RunConfig
    from app.scenario_models import Observation
    from app.scenarios import ScenarioRules
    person=Person((10,0,60,180),.9,track_id=1,keypoints=np.zeros((17,3),np.float32))
    person.keypoints[9]=[30,60,.9]
    phone=Observation((25,55,35,65),.9,'cell_phone','phone-objects')
    rules=ScenarioRules(RunConfig(phone_detection=True,object_hold=.3))
    for t in [0,.2,.4]:rules.step([person],[phone],t,100,200)
    for t in [1,1.5,2,2.5,3,3.5,4]:
        rules.step([person],[],t,100,200);assert not rules.event_updates
    assert rules.state_rows[1]['possible_phone_use']['state']=='violation'
    assert rules.state_rows[1]['possible_phone_use']['observed_state']=='unknown'
    smoking=ScenarioRules(RunConfig(smoking_detection=True))
    small=Person((10,0,30,60),.9,track_id=1)
    smoking.step([small],[],0,100,100)
    assert smoking.state_rows[1]['possible_smoking']['observed_state']=='unknown'
