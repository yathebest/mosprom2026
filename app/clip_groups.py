"""Bounded incident excerpts shared by person events from one source.

This groups media only. Event rows, episode identity and review decisions remain
independent. No group may grow indefinitely through overlapping neighbours.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import math
import threading
import uuid


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Clip time must be finite')
    return float(value)


def episode_identity(event_id, event):
    if event.get('episode_id'):
        return ('episode', str(event['episode_id']))
    if event.get('sequence') is not None and event.get('track_id') is not None:
        return ('legacy', str(event.get('kind', '')), str(event.get('zone') or ''),
                str(event['track_id']), str(event['sequence']))
    return ('event', str(event_id))


def member_counts(members):
    """Count distinct episodes; join track aliases only through a shared episode."""
    episodes = {}
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    for member in members:
        identity = member['identity']
        previous = episodes.get(identity)
        tids = member.get('track_ids', [member.get('track_id')])
        tids = [tid for tid in tids if isinstance(tid, int) and not isinstance(tid, bool) and tid > 0]
        for tid in tids:
            find(tid)
            if tids:
                parent[find(tid)] = find(tids[0])
            old = previous.get('track_id') if previous else None
            if isinstance(old, int) and not isinstance(old, bool) and old > 0:
                parent[find(tid)] = find(old)
        episodes.setdefault(identity, member)
    counts = {}
    for member in episodes.values():
        counts[member['kind']] = counts.get(member['kind'], 0) + 1
    return {'counts': counts, 'episode_count': len(episodes),
            'people_count': len({find(tid) for tid in parent}),
            'scene_events': sum(member.get('track_id') == 0 for member in episodes.values())}


@dataclass
class ClipGroup:
    id: str
    job_id: str
    camera: str
    organization_id: str | None
    start_s: float
    end_s: float
    first_alert_s: float
    last_alert_s: float
    state: str = 'pending'
    result: dict = field(default_factory=dict)
    _members: dict = field(default_factory=dict, repr=False)
    _lock: object = field(default_factory=threading.RLock, repr=False)

    @property
    def members(self):
        with self._lock:
            return deepcopy(self._members)

    def metadata(self):
        with self._lock:
            counts = member_counts(self._members.values())
            return {'clip_group_id': self.id, 'clip_group_event_ids': list(self._members),
                    'clip_group_counts': counts['counts'],
                    'clip_group_episode_count': counts['episode_count'],
                    'clip_group_people_count': counts['people_count'],
                    'clip_group_scene_events': counts['scene_events'],
                    'clip_scope': 'incident_excerpt'}

    def member_metadata(self, event_id, start_s=None):
        with self._lock:
            member = self._members[event_id]
            start = self.start_s if start_s is None else start_s
            return {**self.metadata(), 'clip_alert_offset_s': round(member['alert_s'] - start, 3)}


class GroupPool:
    def __init__(self, job_id, camera, organization_id=None, *, pre_s=3., post_s=4.,
                 max_span_s=10., anchor_window_s=2.):
        if not job_id or not camera:
            raise ValueError('A clip group needs one source job and camera')
        self.job_id, self.camera, self.organization_id = job_id, camera, organization_id
        self.pre_s, self.post_s = _number(pre_s), _number(post_s)
        self.max_span_s, self.anchor_window_s = _number(max_span_s), _number(anchor_window_s)
        if min(self.pre_s, self.post_s, self.anchor_window_s) < 0 or self.max_span_s < self.pre_s + self.post_s:
            raise ValueError('Invalid bounded clip window')
        self._lock = threading.RLock()
        self._groups, self._events, self._episodes = {}, {}, {}

    @property
    def groups(self):
        with self._lock:
            return list(self._groups.values())

    def get(self, group_id):
        with self._lock:
            return self._groups[group_id]

    def add(self, event_id, event, duration_s=None):
        if not event_id or not event.get('kind'):
            raise ValueError('A clip member needs event ID and kind')
        for key, expected in [('job_id', self.job_id), ('camera', self.camera), ('organization_id', self.organization_id)]:
            if key in event and event[key] is not None and event[key] != expected:
                raise ValueError('Clip groups cannot cross source, camera or organization')
        alert = _number(event.get('alert_s'))
        if alert < 0:
            raise ValueError('Clip alert time must be nonnegative')
        duration = None if duration_s is None else _number(duration_s)
        if duration is not None and (duration <= 0 or alert > duration + .001):
            raise ValueError('Alert lies outside the source duration')
        begin, end = max(0., alert - self.pre_s), alert + self.post_s
        if duration is not None:
            end = min(end, duration)
        identity = episode_identity(event_id, event)
        with self._lock:
            known_event = self._events.get(str(event_id))
            if known_event and known_event._members[str(event_id)]['identity'] != identity:
                raise ValueError('One event ID cannot change episode identity')
            group = self._events.get(str(event_id)) or self._episodes.get(identity)
            if group is None:
                candidates = [candidate for candidate in self._groups.values()
                    if candidate.state == 'pending'
                    and max(candidate.last_alert_s, alert) - min(candidate.first_alert_s, alert) <= self.anchor_window_s + 1e-9
                    and max(candidate.end_s, end) - min(candidate.start_s, begin) <= self.max_span_s + 1e-9
                    and begin <= candidate.end_s and end >= candidate.start_s]
                group = min(candidates, key=lambda candidate: abs(alert - candidate.first_alert_s)) if candidates else None
            if group is None:
                group = ClipGroup(uuid.uuid4().hex, self.job_id, self.camera, self.organization_id,
                                  begin, end, alert, alert, _lock=self._lock)
                self._groups[group.id] = group
            existing = next((member for member in group._members.values() if member['identity'] == identity), None)
            if existing and existing['kind'] != event['kind']:
                raise ValueError('One episode cannot change its pattern kind')
            # Known episode retries keep their first measured anchor even if a
            # caller later supplies another arrival timestamp or track alias.
            measured_alert = existing['alert_s'] if existing else alert
            member = {'event_id': str(event_id), 'episode_id': event.get('episode_id'),
                      'kind': str(event['kind']), 'track_id': event.get('track_id'),
                      'alert_s': measured_alert, 'identity': identity}
            aliases = set(existing.get('track_ids', [existing.get('track_id')])) if existing else set()
            aliases.add(event.get('track_id'))
            member['track_ids'] = sorted(tid for tid in aliases if isinstance(tid, int) and not isinstance(tid, bool) and tid > 0)
            group._members[str(event_id)] = member
            self._events[str(event_id)], self._episodes[identity] = group, group
            if group.state == 'pending' and existing is None:
                group.first_alert_s, group.last_alert_s = min(group.first_alert_s, measured_alert), max(group.last_alert_s, measured_alert)
                group.start_s, group.end_s = min(group.start_s, begin), max(group.end_s, end)
            return group

    def reserve(self, group_id):
        """Atomically freeze new-episode membership and reserve one encoder."""
        with self._lock:
            group = self._groups[group_id]
            if group.state != 'pending':
                return False
            group.state = 'exporting'
            return True

    def finish(self, group_id, state, result=None):
        if state not in ('ready', 'failed', 'cancelled'):
            raise ValueError('Invalid final clip state')
        with self._lock:
            group = self._groups[group_id]
            if group.state in ('ready', 'failed', 'cancelled'):
                return False
            group.state, group.result = state, deepcopy(result or {})
            return True
