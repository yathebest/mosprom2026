"""Scoped operational observations, not a synthetic safety score or model accuracy."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import math
import re
from statistics import median

from . import storage as db
from .access import sql_scope
from .operations import PRIORITIES

CORE_KINDS = frozenset(('no_helmet', 'danger_zone', 'possible_fall', 'person_down'))
PENDING = frozenset(('new', 'forwarded'))
REVIEWED = frozenset(('confirmed', 'false'))
KNOWN_FIXTURE_FILES = frozenset(('no_hardhat-demo.mp4','hardhat-demo.mp4',
    'fire-demo-frame-demo.mp4','no_harness-demo.mp4','phone-demo-demo.mp4','smoking-output2-demo.mp4'))


def instant(value):
    if not isinstance(value, str): return None
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        # A timezone-free date is not a known global recording instant.
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (ValueError, OverflowError): return None


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def period(days, date_from, date_to, now):
    """UI dates are inclusive Moscow calendar days; ISO endpoints are explicit."""
    def boundary(value, is_end=False):
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value or ''):
            try:
                result=datetime.fromisoformat(value).replace(tzinfo=timezone(timedelta(hours=3)))
                return (result+timedelta(days=1) if is_end else result).astimezone(timezone.utc)
            except (ValueError, OverflowError): raise ValueError('Invalid calendar date') from None
        result=instant(value)
        if result is None: raise ValueError('Use YYYY-MM-DD (Moscow) or an ISO timestamp with a timezone')
        return result
    if date_from is not None or date_to is not None:
        if not date_from or not date_to: raise ValueError('Both from and to are required')
        start,end=boundary(date_from),boundary(date_to,True)
        if start>=end: raise ValueError('Period start must be before its exclusive end')
        return start,end
    if days not in (7,30,None): raise ValueError('Days must be 7, 30 or all')
    return now-timedelta(days=days) if days is not None else None,now


def educational(job):
    meta = job.get('meta') or {}
    if any(meta.get(k) is True for k in ('example', 'demo', 'synthetic', 'test_fixture')): return True
    if meta.get('evaluation_split') in ('train', 'val', 'test', 'integration_only'): return True
    source = str(meta.get('source', '')).casefold()
    if 'repeated' in source or source in ('urfd', 'scenario-fixtures'): return True
    # These are the project's explicit public factory dataset clip identifiers.
    # No arbitrary camera name or generic "test" substring classifies a source.
    filename = str(job.get('filename') or '').replace('\\', '/').rsplit('/', 1)[-1]
    return (filename in KNOWN_FIXTURE_FILES
        or bool(re.fullmatch(r'(?:fall|adl)-\d{2}-cam[01]-rgb\.mp4',filename))
        or bool(re.fullmatch(r'\d+_(?:te|tr)\d+\.mp4', filename)))


def source_purpose(job):
    if educational(job): return 'educational'
    if (job.get('meta') or {}).get('source_purpose') in ('production','operational'):
        return 'declared_production'
    return 'unclassified'


def load_scoped():
    """Read all scoped history; db.jobs() intentionally only returns the newest 100."""
    clause, args = sql_scope()
    with db.connection() as connection:
        jobs = [db.decode(row) for row in connection.execute('SELECT * FROM jobs WHERE 1=1'+clause, args)]
        audit = []
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='audit'").fetchone():
            clause, args = sql_scope('jobs', 'events.camera')
            audit = [dict(row) for row in connection.execute(
                "SELECT audit.target,audit.created,audit.details FROM audit JOIN events ON events.id=audit.target "
                "JOIN jobs ON jobs.id=events.job_id WHERE audit.action='event.reviewed' "
                'AND audit.organization_id=jobs.organization_id'+clause, args)]
    return db.events(), jobs, db.cameras(), audit


def review_times(audit):
    import json
    result = defaultdict(list)
    for row in audit:
        try: details = json.loads(row.get('details') or '{}')
        except (ValueError, TypeError): continue
        if not isinstance(details,dict): continue
        when = instant(row.get('created'))
        if details.get('status') in REVIEWED and when:
            result[row['target']].append(when)
    return result


def coverage(jobs, *, cutoff, now, period_end=None, camera='', include_examples=False, time_basis='processing'):
    """Only explicit measured coverage intervals qualify; never wall runtime/duration.

    Current production jobs do not record continuous NN coverage. Their clip
    duration, decode progress and last source timestamp cannot prove that a rule
    was continuously observable. Future producers may persist this schema.
    """
    intervals = defaultdict(list); unknown = 0; eligible = 0
    for job in jobs:
        config = job.get('config') or {}; name = config.get('camera')
        if not name or (camera and name != camera) or (educational(job) and not include_examples): continue
        created = instant(job.get('created'))
        if time_basis == 'processing' and (not created or (cutoff and created < cutoff) or created >= (period_end or now)): continue
        eligible += 1
        samples = (job.get('stats') or {}).get('analytics_coverage')
        valid = []
        if isinstance(samples, dict) and samples.get('basis') == 'continuous_observed_source_time':
            for sample in samples.get('intervals', []) or []:
                if not isinstance(sample, dict): continue
                start, end = instant(sample.get('start_at')), instant(sample.get('end_at'))
                if start and end and end > start and end <= now:
                    start = max(start, cutoff) if cutoff and time_basis == 'recording' else start
                    end = min(end, period_end) if period_end and time_basis == 'recording' else end
                    if end > start: valid.append((start.timestamp(), end.timestamp()))
        if not valid: unknown += 1
        intervals[name].extend(valid)
    durations = {}
    for name, values in intervals.items():
        merged = []
        for start, end in sorted(values):
            if merged and start <= merged[-1][1]: merged[-1][1] = max(end, merged[-1][1])
            else: merged.append([start, end])
        durations[name] = sum(end-start for start, end in merged)
    seconds = sum(durations.values())
    complete = eligible > 0 and unknown == 0 and seconds > 0
    return {'available': complete, 'camera_hours': round(seconds/3600, 6) if complete else None,
        'known_camera_hours': round(seconds/3600, 6) if seconds else None,
        'jobs_without_proven_coverage': unknown, 'eligible_jobs': eligible,
        'per_camera_hours': {key: round(value/3600, 6) for key,value in durations.items()} if complete else {},
        'reason': None if complete else 'continuous_source_coverage_not_recorded',
        'policy': 'Union explicit observed-source intervals per camera. Clip length, processing time and a last frame are not camera-hours. Incomplete coverage is never used as a rate denominator.'}


def aggregate(events, jobs, templates=(), audit=(), *, days=7, camera='', time_basis='processing',
              include_experimental=False, include_examples=False, now=None, allow_events=True,
              date_from=None, date_to=None):
    now = now or datetime.now(timezone.utc)
    if time_basis not in ('processing','recording'): raise ValueError('Invalid time basis')
    cutoff,period_end = period(days,date_from,date_to,now)
    by_job = {job['id']: job for job in jobs}
    first_reviews = review_times(audit)
    records = []; excluded = Counter(); cameras = set(); configured_cameras=set()
    for template in templates:
        if template.get('name'):
            cameras.add(template['name']); configured_cameras.add(template['name'])
    for job in jobs:
        if (job.get('config') or {}).get('camera') and (include_examples or not educational(job)):
            cameras.add(job['config']['camera'])
    for event in events:
        job = by_job.get(event['job_id'])
        if job is None: continue  # never aggregate orphan/cross-scope events
        if include_examples or not educational(job): cameras.add(event['camera'])
        if camera and event['camera'] != camera: continue
        occurred = instant((event.get('payload') or {}).get('occurred_at'))
        created = instant(event.get('created'))
        when = occurred if time_basis == 'recording' else created
        if when is None:
            excluded['unknown_time'] += 1; continue
        if when > now or when >= period_end or (cutoff and when < cutoff): continue
        if educational(job) and not include_examples:
            excluded['examples'] += 1; continue
        if event['kind'] not in CORE_KINDS and not include_experimental:
            excluded['experimental'] += 1; continue
        records.append(event)
    pending = [e for e in records if e['status'] in PENDING]
    confirmed = [e for e in records if e['status'] == 'confirmed']
    rejected = [e for e in records if e['status'] == 'false']
    latency = []; reviewed_without_time = 0; overdue = 0; high_pending = 0
    closed_seconds = []; lifecycles = Counter(); groups = defaultdict(list); camera_events = defaultdict(list)
    details = []
    for event in records:
        payload = event.get('payload') or {}; start = number(event.get('start_s')); end = number(payload.get('end_s'))
        state = payload.get('episode_state')
        if state not in ('open','closed','interrupted'): state='legacy_unknown'
        valid_closed = state == 'closed' and start is not None and end is not None and end >= start
        if state == 'closed' and not valid_closed: state = 'invalid_closed'
        lifecycles[state] += 1
        duration = end-start if valid_closed else None
        if duration is not None: closed_seconds.append(duration)
        created = instant(event.get('created'))
        waiting = max(0, (now-created).total_seconds()) if created else None
        priority = payload.get('priority') or PRIORITIES.get(event['kind'], 'low')
        target = 60 if priority == 'high' else 300
        if event['status'] in PENDING:
            overdue += waiting is not None and waiting > target
            high_pending += priority == 'high'
        if event['status'] in REVIEWED:
            decisions = [t for t in first_reviews.get(event['id'], ()) if created and t >= created and t <= now]
            if decisions: latency.append((min(decisions)-created).total_seconds())
            else: reviewed_without_time += 1
        key = (event['camera'], event['kind'], event.get('zone') or '')
        groups[key].append(event); camera_events[event['camera']].append(event)
        if allow_events:
            details.append({'id':event['id'], 'job_id':event['job_id'], 'track_id':event.get('track_id'), 'kind':event['kind'], 'camera':event['camera'],
                'zone':event.get('zone'), 'status':event['status'], 'created':event.get('created'),
                'occurred_at':payload.get('occurred_at'), 'start_s':start, 'alert_s':event.get('alert_s'),
                'end_s':end if valid_closed else None, 'duration_s':duration, 'episode_state':state,
                'observed_until_s':number(payload.get('observed_until_s')),
                'has_evidence':bool(event.get('image') or event.get('clip')), 'priority':priority,
                'waiting_s':round(waiting) if waiting is not None and event['status'] in PENDING else None,
                'experimental':event['kind'] not in CORE_KINDS, 'educational':educational(by_job[event['job_id']])})
    priorities = []
    for (name,kind,zone), values in groups.items():
        awaiting = [e for e in values if e['status'] in PENDING]
        checked = sum(e['status']=='confirmed' for e in values)
        if not awaiting and not checked: continue
        oldest = min((instant(e.get('created')) for e in awaiting if instant(e.get('created'))), default=None)
        priorities.append({'camera':name, 'kind':kind, 'zone':zone, 'events':len(values),
            'confirmed':checked, 'pending':len(awaiting), 'rejected':sum(e['status']=='false' for e in values),
            'high_priority_pending':sum(((e.get('payload') or {}).get('priority') or PRIORITIES.get(e['kind'],'low'))=='high' for e in awaiting),
            'distinct_jobs':len({e['job_id'] for e in values}),
            'oldest_waiting_s':round(max(0,(now-oldest).total_seconds())) if oldest else None,
            'event_ids':[e['id'] for e in sorted(awaiting,key=lambda e:e.get('created') or '')[:10]] if allow_events else [],
            'experimental':kind not in CORE_KINDS})
    # Transparent ordering, not a probabilistic or invented weighted risk score.
    priorities.sort(key=lambda r:(-bool(r['high_priority_pending']), -bool(r['pending']),
        -r['confirmed'], -(r['oldest_waiting_s'] or 0), -r['events'], r['camera'],r['kind'],r['zone']))
    coverage_info = coverage(jobs,cutoff=cutoff,now=now,period_end=period_end,camera=camera,include_examples=include_examples,time_basis=time_basis)
    camera_rows=[]
    for name in sorted(set(camera_events) | ({camera} if camera and camera in cameras else configured_cameras if not camera else set())):
        values=camera_events[name]
        camera_rows.append({'camera':name, 'events':len(values), 'pending':sum(e['status'] in PENDING for e in values),
            'confirmed':sum(e['status']=='confirmed' for e in values), 'rejected':sum(e['status']=='false' for e in values),
            'top_kind':Counter(e['kind'] for e in values).most_common(1)[0][0] if values else None,
            'camera_hours':coverage_info['per_camera_hours'].get(name)})
    unresolved=sum(lifecycles[state] for state in ('open','interrupted','legacy_unknown','invalid_closed'))
    purposes=Counter(source_purpose(by_job[e['job_id']]) for e in records)
    purpose_jobs=Counter(source_purpose(by_job[key]) for key in {e['job_id'] for e in records})
    return {'filters':{'days':None if date_from else days,'date_from':date_from,'date_to':date_to,
            'time_zone':'Europe/Moscow','camera':camera,'time_basis':time_basis,'include_experimental':include_experimental,'include_examples':include_examples},
        'generated_at':now.isoformat(), 'period_start':cutoff.isoformat() if cutoff else None,'period_end':period_end.isoformat(),
        'available_cameras':sorted(cameras), 'core_kinds':sorted(CORE_KINDS), 'excluded':dict(excluded),
        'kpis':{'automatic_events':len(records),'pending':len(pending),'confirmed':len(confirmed),
            'unresolved_episodes':unresolved,'rejected':len(rejected)},
        'queue':{'pending':len(pending),'overdue':overdue,'high_priority':high_pending,
            'targets_s':{'high':60,'other':300},'basis':'processing_created_time'},
        'review':{'median_first_decision_s':round(median(latency),2) if latency else None,
            'timed_decisions':len(latency),'reviewed_without_audit_time':reviewed_without_time,
            'basis':'First confirmed/rejected decision from scoped audit; not inferred from current status.'},
        'episodes':{'states':dict(lifecycles),'closed_count':len(closed_seconds),
            'closed_interval_total_s':round(sum(closed_seconds),3) if closed_seconds else None,
            'closed_interval_median_s':round(median(closed_seconds),3) if closed_seconds else None,
            'censored_count':unresolved,'note':'Closed interval is a confirmed FSM boundary interval; open/interrupted/legacy endpoints are unknown. Sums are per episode/person and may overlap, not factory downtime or continuous unsafe exposure.'},
        'coverage':coverage_info, 'priorities':priorities[:8], 'priority_groups_total':len(priorities),
        'cameras':camera_rows,'events':sorted(details,key=lambda e:e.get('created') or '',reverse=True),
        'breakdowns':{'by_kind':dict(Counter(e['kind'] for e in records)),
            'by_status':dict(Counter(e['status'] for e in records)),
            'by_processing_day':dict(sorted(Counter(instant(e['created']).astimezone(timezone(timedelta(hours=3))).date().isoformat()
                for e in records if instant(e.get('created'))).items()))},
        'can_drilldown':allow_events,
        'data_quality':{'source_purpose_events':{key:purposes[key] for key in ('educational','declared_production','unclassified')},
            'source_purpose_jobs':{key:purpose_jobs[key] for key in ('educational','declared_production','unclassified')}},
        'counting_policy':'Each stored episode is counted once. Repeat analyses of the same source can create separate episodes; recurrence across jobs is not independent evidence of repeated real-world incidents.',
        'time_policy':'Period uses processing created time by default, never an assumed filming date. occurred_at is the supplied file start plus offset or live server-arrival time, not certified camera exposure time.',
        'origin_policy':'Only explicitly marked examples and known project fixture/dataset filenames are excluded by default. Legacy/live sources without a purpose flag remain unclassified, not verified production; camera names are never used as proof. No history is relabelled or deleted.'}


def build_analytics(*, days=7, camera='', time_basis='processing', include_experimental=False, include_examples=False,
                    allow_events=False,date_from=None,date_to=None,now=None):
    return aggregate(*load_scoped(),days=days,camera=camera,time_basis=time_basis,
        include_experimental=include_experimental,include_examples=include_examples,allow_events=allow_events,
        date_from=date_from,date_to=date_to,now=now)


summary=build_analytics
