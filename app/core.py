from collections import deque
from dataclasses import dataclass, field
import math
from datetime import datetime
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator
from .temporal_state import PatternStates


class GroundCalibration(BaseModel):
    image_points: list[tuple[float,float]] = Field(min_length=4,max_length=4)
    world_points: list[tuple[float,float]] = Field(min_length=4,max_length=4)

    @model_validator(mode="after")
    def valid(self):
        Zone(points=self.image_points)
        p=np.asarray(self.world_points,np.float32)
        if not np.isfinite(p).all() or abs(cv2.contourArea(p))<.01:
            raise ValueError("Калибровка пола должна иметь ненулевую площадь в метрах")
        return self


class Zone(BaseModel):
    name: str = Field(default="Опасная зона", min_length=1, max_length=80)
    points: list[tuple[float, float]] = Field(min_length=3, max_length=32)
    priority: Literal['low','medium','high'] = 'medium'
    kind: Literal['danger_zone','barrier','under_load','under_conveyor','height','equipment','smoking_allowed','fire_allowed','reflection'] = 'danger_zone'

    @model_validator(mode="after")
    def valid_polygon(self):
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.points):
            raise ValueError("Координаты зоны должны быть в диапазоне 0–1")
        p = np.asarray(self.points, np.float32)
        if abs(cv2.contourArea(p)) < 0.0001:
            raise ValueError("Зона должна иметь ненулевую площадь")
        # Reject self-intersecting polygons rather than guessing the interior.
        def cross(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        n = len(p)
        for i in range(n):
            for j in range(i+1, n):
                if j == i+1 or (i == 0 and j == n-1):
                    continue
                a,b,c,d = p[i],p[(i+1)%n],p[j],p[(j+1)%n]
                if cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0:
                    if max(min(a[0],b[0]),min(c[0],d[0])) <= min(max(a[0],b[0]),max(c[0],d[0])) and max(min(a[1],b[1]),min(c[1],d[1])) <= min(max(a[1],b[1]),max(c[1],d[1])):
                        raise ValueError("Границы зоны не должны пересекаться")
        return self


class RunConfig(BaseModel):
    camera: str = Field(default="Камера 01", min_length=1, max_length=80)
    recorded_at: str | None = None
    zones: list[Zone] = Field(default_factory=list, max_length=8)
    zone_hold: float = Field(default=1.0, ge=0.2, le=30)
    fast_motion: bool = True
    speed_threshold: float = Field(default=1.25, ge=0.2, le=10)
    motion_hold: float = Field(default=1.0, ge=0.2, le=10)
    fall_detection: bool = False
    fall_hold: float = Field(default=1.0, ge=0.3, le=10)
    lying_hold: float = Field(default=5.0, ge=1, le=60)
    release_hold: float = Field(default=2.0, ge=0.2, le=30)
    observation_gap: float = Field(default=0.8, ge=0.2, le=3)
    sample_fps: float = Field(default=10, ge=2, le=30)
    confidence: float = Field(default=0.4, ge=0.1, le=0.9)
    calibration: GroundCalibration | None = None
    speed_threshold_m_s: float = Field(default=2.5,ge=.1,le=20)
    body_threshold: float = Field(default=.9,ge=.1,le=.99)
    body_hold: float = Field(default=.5,ge=.1,le=5)
    body_transition_max_s: float = Field(default=1.0,ge=.2,le=3)
    helmet_detection: bool = False
    vest_detection: bool = False
    mask_detection: bool = False
    harness_detection: bool = False
    phone_detection: bool = False
    fire_detection: bool = False
    smoke_detection: bool = False
    smoking_detection: bool = False
    proximity_detection: bool = False
    load_posture_detection: bool = False
    ppe_hold: float = Field(default=1.5,ge=.3,le=15)
    object_hold: float = Field(default=1.0,ge=.3,le=15)
    object_confidence: float = Field(default=.5,ge=.15,le=.95)
    phone_confidence: float = Field(default=.3,ge=.15,le=.95)
    phone_work_context_only: bool = False
    distant_people: bool = True
    smoking_min_person_height: int = Field(default=120,ge=40,le=600)
    ppe_crops: bool = True
    ppe_max_people: int = Field(default=8,ge=1,le=32)
    ppe_memory_s: float = Field(default=.5,ge=0,le=2)
    ppe_conflict_margin: float = Field(default=.15,ge=.05,le=.5)
    proximity_heights: float = Field(default=.6,ge=.1,le=5)
    proximity_m: float = Field(default=3.0,ge=.3,le=15)
    state_confirmation_ratio: float = Field(default=.75,ge=.75,le=1.)
    state_min_samples: int = Field(default=3,ge=3,le=30)
    state_windows: dict[str,float] = Field(default_factory=dict,max_length=32)

    @model_validator(mode="after")
    def names_unique(self):
        if self.recorded_at:
            parsed=datetime.fromisoformat(self.recorded_at)
            if parsed.tzinfo is None:
                raise ValueError('Время начала записи должно содержать часовой пояс')
        if len({z.name for z in self.zones}) != len(self.zones):
            raise ValueError("Имена зон должны различаться")
        if self.observation_gap < 1/self.sample_fps:
            raise ValueError("Допустимый пропуск наблюдений меньше шага обработки")
        allowed={'danger_zone','fast_motion','possible_fall','person_down','no_helmet','no_vest','no_mask',
                 'no_harness_at_height','possible_phone_use','possible_smoking','fire','smoke',
                 'equipment_proximity','barrier_crossing','under_load','under_conveyor','unsafe_load_posture'}
        if set(self.state_windows)-allowed or any(not math.isfinite(v) or not .2<=v<=60 for v in self.state_windows.values()):
            raise ValueError('Окна подтверждения должны задавать известные паттерны и длительность 0.2–60 с')
        return self


@dataclass
class Person:
    box: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray | None = None
    track_id: int = -1
    observed: bool = True
    alert_eligible: bool = True

    @property
    def height(self):
        return max(self.box[3]-self.box[1], 1.0)

    @property
    def foot(self):
        return ((self.box[0]+self.box[2])/2, self.box[3])


@dataclass
class Gate:
    since: float | None = None
    last_true: float | None = None
    last_seen: float | None = None
    active: bool = False
    sequence: int = 0

    def update(self, now, condition, confirm, release, gap):
        """Emit once per episode. Missing observations never confirm a condition."""
        if self.last_seen is not None and now-self.last_seen > gap:
            self.since = None
        self.last_seen = now
        if condition:
            if self.last_true is not None and now-self.last_true >= release:
                self.active = False
                self.since = None
            if self.since is None:
                self.since = now
            self.last_true = now
            if not self.active and now-self.since >= confirm-1e-8:
                self.active = True
                self.sequence += 1
                return self.since, self.sequence
        else:
            self.since = None
            if self.last_true is not None and now-self.last_true >= release:
                self.active = False
        return None


@dataclass
class TrackState:
    history: deque = field(default_factory=lambda: deque(maxlen=300))
    last_seen: float = 0
    last_upright: float | None = None
    lying_since: float | None = None
    fall_candidate: bool = False


def posture(person):
    """Conservative COCO pose geometry; unknown is distinct from standing."""
    k = person.keypoints
    if k is None or k.shape != (17, 3) or not np.isfinite(k).all():
        return "unknown"
    if not all(k[i,2] >= .4 for i in (5,6,11,12)):
        return "unknown"
    shoulders = k[[5,6],:2].mean(0)
    hips = k[[11,12],:2].mean(0)
    dx, dy = hips-shoulders
    length = math.hypot(dx, dy)
    if length < person.height*.12:
        return "unknown"
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    if angle >= 65 and (person.box[2]-person.box[0])/person.height > .85:
        return "lying"
    if angle <= 30 and dy > 0:
        return "upright"
    return "other"


class Rules:
    def __init__(self, config: RunConfig):
        self.config = config
        self.tracks: dict[int, TrackState] = {}
        self.temporal=PatternStates(config);self.gates=self.temporal.gates
        self.event_updates=[];self.state_rows={}
        self.scene = {"upright_since":None,"last_upright":None,"fall_candidate":False,"high_since":None,"owner":None}

    def _body_kind(self, tid, proposed, lying):
        body=[(key[1],gate) for key,gate in self.gates.items()
              if key[0]==tid and key[1] in ('possible_fall','person_down')]
        active=next((kind for kind,gate in body if gate.active),None)
        if active is not None:return active
        # Keep observing recovery on the gate that owned the last posture
        # episode. Switching immediately after clear discards its standing
        # evidence and can mislabel a second observed fall as person_down.
        # Select a new kind only when a new lying observation begins.
        if not lying and body:
            return max(body,key=lambda item:item[1].last_observed_s
                       if item[1].last_observed_s is not None else -math.inf)[0]
        return proposed

    def step(self, people: list[Person], now: float, width: int, height: int, scene_state=None):
        events, details = [], {}
        handoffs=self.temporal.begin(people,now);self.event_updates=self.temporal.event_updates
        for new,old in handoffs.items():
            if old in self.tracks:self.tracks[new]=self.tracks.pop(old)
            if self.scene.get('owner')==old:self.scene['owner']=new
        cfg = self.config
        homography=None
        if cfg.calibration:
            homography=cv2.getPerspectiveTransform(
                (np.asarray(cfg.calibration.image_points)*[width,height]).astype(np.float32),
                np.asarray(cfg.calibration.world_points,np.float32))
        for person in people:
            tid = person.track_id
            if tid < 0:
                continue
            if not person.observed or not person.alert_eligible or person.confidence<cfg.confidence:
                details[tid] = {"speed":0, "speed_m_s":None, "posture":"unknown",
                                "active":[key[1] for key,gate in self.gates.items() if key[0]==tid and gate.active], "tracking":"predicted" if not person.observed else "weak"}
                continue
            state = self.tracks.setdefault(tid, TrackState())
            if state.history and now-state.last_seen > cfg.observation_gap:
                state.history.clear()
                state.last_upright = None
                state.lying_since = None
                state.fall_candidate = False
            state.last_seen = now
            state.history.append((now, *person.foot, person.height))
            while len(state.history)>1 and now-state.history[0][0] > 1.0:
                state.history.popleft()
            speed = 0.0
            speed_ready=False
            speed_m_s = None
            if len(state.history)>=3 and now-state.history[0][0] >= .4:
                speed_ready=True
                arr = np.asarray(state.history)
                ts = arr[:,0]-arr[:,0].mean()
                velocity = (ts[:,None]*(arr[:,1:3]-arr[:,1:3].mean(0))).sum(0)/max(float(ts@ts),1e-9)
                speed = float(np.linalg.norm(velocity)/np.median(arr[:,3]))
                if homography is not None:
                    coords=cv2.perspectiveTransform(arr[:,1:3].astype(np.float32)[None],homography)[0]
                    vel=(ts[:,None]*(coords-coords.mean(0))).sum(0)/max(float(ts@ts),1e-9)
                    speed_m_s=float(np.linalg.norm(vel))
            pose = posture(person)
            if pose == "upright":
                state.last_upright = now
                state.lying_since = None
                state.fall_candidate = False
            elif pose == "lying":
                if state.lying_since is None:
                    state.lying_since = now
                    standing_confirmed=any(key[0]==tid and key[1] in ('possible_fall','person_down') and gate.state=='clear'
                                           for key,gate in self.gates.items())
                    state.fall_candidate = standing_confirmed and state.last_upright is not None and now-state.last_upright <= 1.2
            else:
                state.lying_since = None
                state.fall_candidate = False
            conditions = []
            for zone in cfg.zones:
                poly = np.asarray(zone.points, np.float32)*[width,height]
                inside = cv2.pointPolygonTest(poly.astype(np.float32),person.foot,False)>=0
                conditions.append(("danger_zone",zone.name,inside,cfg.zone_hold))
            if cfg.fast_motion:
                fast = speed_m_s>=cfg.speed_threshold_m_s if speed_m_s is not None else speed>=cfg.speed_threshold
                conditions.append(("fast_motion","",fast if speed_ready else None,cfg.motion_hold))
            if cfg.fall_detection and not scene_state:
                proposed='possible_fall' if state.fall_candidate else 'person_down'
                kind=self._body_kind(tid,proposed,pose=='lying')
                conditions.append((kind,'',True if pose=='lying' else False if pose=='upright' else None,
                                   cfg.fall_hold if kind=='possible_fall' else cfg.lying_hold))
            active = []
            for kind,zone,condition,hold in conditions:
                key = (tid,kind,zone)
                gate,fired=self.temporal.observe(tid,kind,zone,condition,hold,cfg.release_hold,cfg.observation_gap)
                if gate.active:
                    active.append(kind)
                if fired:
                    started,sequence = fired
                    events.append({"track_id":tid,"kind":kind,"zone":zone,
                                   "start_s":round(started,3),"alert_s":round(now,3),
                                   "sequence":sequence,"score":round(person.confidence,4),
                                   "speed_heights_s":round(speed,3),"speed_m_s":speed_m_s,"posture":pose,
                                   "reason":(f"Нахождение в зоне «{zone}» не менее {hold:g} с" if kind=="danger_zone" else
                                              f"Условие «{kind}» устойчиво не менее {hold:g} с"),
                                   "confirmation_s":round(now-started,3),"source":"person_tracking","scope":"person",
                                   **gate.last_transition})
            details[tid] = {"speed":speed,"speed_m_s":speed_m_s,"speed_ready":speed_ready,"posture":pose,"active":active,
                            "tracking":"measured" if person.alert_eligible else "weak"}
        # Absent tracks cannot accumulate confirmation time, and memory is bounded.
        for tid,state in list(self.tracks.items()):
            if now-state.last_seen > max(15,cfg.release_hold+cfg.observation_gap):
                del self.tracks[tid]
                for key in [k for k in self.gates if k[0]==tid]:
                    del self.gates[key]
        if cfg.fall_detection and scene_state is not None:
            # RGB/temporal classifiers measure the whole scene, not a track.
            # Detector dropouts must not split their history between track0
            # and a person ID, nor attribute scene recovery to a lost worker.
            scene_tid=0
            probability=float(scene_state.get("lying_probability",0))
            high=probability>=cfg.body_threshold
            temporal=scene_state.get('fall_probability')
            temporal_high=(temporal is not None and temporal>=scene_state.get('fall_threshold',.8)
                           and (high or any(d['posture']=='lying' and d['tracking']=='measured'
                                            for d in details.values())))
            high=high or temporal_high
            strong_upright=probability<.1 and not high and "lying_probability" in scene_state
            if strong_upright:
                if self.scene["upright_since"] is None:
                    self.scene["upright_since"]=now
                if now-self.scene["upright_since"]>=.2:
                    self.scene["last_upright"]=now
                self.scene["high_since"]=None
                self.scene["fall_candidate"]=False
            elif high:
                if self.scene["high_since"] is None:
                    self.scene["high_since"]=now
                    previous=self.scene["last_upright"]
                    # A transition frame may be unknown. It contributes no
                    # vote but cannot erase previously confirmed standing;
                    # freshness is bounded separately by last_upright below.
                    standing_confirmed=any(key[0]==scene_tid and key[1] in ('possible_fall','person_down') and gate.state=='clear'
                                           for key,gate in self.gates.items())
                    self.scene["fall_candidate"]=standing_confirmed and previous is not None and now-previous<=cfg.body_transition_max_s
                self.scene["upright_since"]=None
            else:
                self.scene["upright_since"]=None
                self.scene["high_since"]=None
            proposed="possible_fall" if self.scene["fall_candidate"] else "person_down"
            kind=self._body_kind(scene_tid,proposed,high)
            for event_kind in (kind,):
                hold=cfg.body_hold if event_kind=="possible_fall" else cfg.lying_hold
                measured=True if high else False if strong_upright else None
                gate,fired=self.temporal.observe(scene_tid,event_kind,'',measured,hold,cfg.release_hold,cfg.observation_gap)
                if fired:
                    self.scene['owner']=scene_tid
                    started,sequence=fired
                    events.append({"track_id":scene_tid,"kind":event_kind,"zone":"","start_s":round(started,3),
                        "alert_s":round(now,3),"sequence":sequence,"score":float(temporal) if temporal_high else probability,
                        "posture":"lying","source":"rgb_temporal_fusion" if temporal_high else "rgb_scene_classifier",
                        "confirmation_s":round(now-started,3),"scope":"person" if scene_tid else "scene",**gate.last_transition,
                        "reason":(f"Временная модель падения и положение человека подтверждают событие в течение {hold:g} с. Требуется проверка оператором."
                                  if temporal_high else f"RGB-модель: вероятность горизонтального положения ≥ {cfg.body_threshold:.2f} в течение {hold:g} с. Требуется проверка оператором.")})
        if not any(gate.active for key,gate in self.gates.items() if key[1] in ('possible_fall','person_down')):self.scene['owner']=None
        self.state_rows=self.temporal.end(now);self.event_updates=self.temporal.event_updates
        for tid,rows in self.state_rows.items():
            if tid in details:details[tid]['patterns']=rows
        return events,details

    def finish(self,now,reason='interrupted'):
        updates=self.temporal.finish(now,reason);self.state_rows=self.temporal.state_rows;return updates
