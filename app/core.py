from collections import deque
from dataclasses import dataclass, field
import math
from datetime import datetime
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator


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
    kind: Literal['danger_zone','barrier','under_load','under_conveyor','height','equipment','smoking_allowed','fire_allowed'] = 'danger_zone'

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
    proximity_heights: float = Field(default=.6,ge=.1,le=5)
    proximity_m: float = Field(default=3.0,ge=.3,le=15)

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
        return self


@dataclass
class Person:
    box: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray | None = None
    track_id: int = -1

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
        self.gates: dict[tuple, Gate] = {}
        self.scene = {"upright_since":None,"last_upright":None,"fall_candidate":False,"high_since":None}

    def step(self, people: list[Person], now: float, width: int, height: int, scene_state=None):
        events, details = [], {}
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
            speed_m_s = None
            if len(state.history)>=3 and now-state.history[0][0] >= .4:
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
                    state.fall_candidate = state.last_upright is not None and now-state.last_upright <= 1.2
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
                conditions.append(("fast_motion","",fast,cfg.motion_hold))
            if cfg.fall_detection and not scene_state:
                conditions.extend([
                    ("possible_fall","",pose=="lying" and state.fall_candidate,cfg.fall_hold),
                    ("person_down","",pose=="lying" and not state.fall_candidate,cfg.lying_hold),
                ])
            active = []
            for kind,zone,condition,hold in conditions:
                key = (tid,kind,zone)
                gate = self.gates.setdefault(key,Gate())
                fired = gate.update(now,condition,hold,cfg.release_hold,cfg.observation_gap)
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
                                   "confirmation_s":round(now-started,3),"source":"person_tracking"})
            details[tid] = {"speed":speed,"speed_m_s":speed_m_s,"posture":pose,"active":active}
        # Absent tracks cannot accumulate confirmation time, and memory is bounded.
        for tid,state in list(self.tracks.items()):
            if now-state.last_seen > max(15,cfg.release_hold+cfg.observation_gap):
                del self.tracks[tid]
                for key in [k for k in self.gates if k[0]==tid]:
                    del self.gates[key]
        if cfg.fall_detection and scene_state is not None:
            probability=float(scene_state.get("lying_probability",0))
            high=probability>=cfg.body_threshold
            strong_upright=probability<.1
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
                    self.scene["fall_candidate"]=previous is not None and now-previous<=cfg.body_transition_max_s
                self.scene["upright_since"]=None
            else:
                self.scene["upright_since"]=None
                self.scene["high_since"]=None
            previous_fall=self.gates.get((0,"possible_fall",""))
            kind="possible_fall" if self.scene["fall_candidate"] or (previous_fall and previous_fall.active) else "person_down"
            for event_kind in ("possible_fall","person_down"):
                hold=cfg.body_hold if event_kind=="possible_fall" else cfg.lying_hold
                gate=self.gates.setdefault((0,event_kind,""),Gate())
                fired=gate.update(now,high and kind==event_kind,hold,cfg.release_hold,cfg.observation_gap)
                if fired:
                    started,sequence=fired
                    events.append({"track_id":0,"kind":event_kind,"zone":"","start_s":round(started,3),
                        "alert_s":round(now,3),"sequence":sequence,"score":probability,
                        "posture":"lying","source":"rgb_scene_classifier","confirmation_s":round(now-started,3),
                        "reason":f"RGB-модель: вероятность горизонтального положения ≥ {cfg.body_threshold:.2f} в течение {hold:g} с. Требуется проверка оператором."})
        return events,details
