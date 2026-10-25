"""Replay public videos through the deployed pipeline, one pattern at a time.

No DB/API mutations and no threshold fitting. Expected event presence is a
case-level integration check, never a precision/recall benchmark.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core import RunConfig, Rules
from app.inference import create_backend, PeopleTracker
from app.scenarios import ScenarioRules, base_config, annotate_objects
from app.scenario_models import required_models
from app.video import MODEL, draw


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def run_case(case, backend, config, output):
    path = ROOT / case['file']
    if not path.is_file():
        return {**case, 'status': 'missing_source', 'passed': False}
    cap = cv2.VideoCapture(str(path))
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count <= 0:
        cap.release()
        return {**case, 'status': 'unreadable_source', 'passed': False}
    start = float(case.get('start_s', 0))
    end = min(count / fps, float(case.get('end_s', count / fps)))
    stride = max(1, round(fps / config.sample_fps))
    tracker = PeopleTracker(fps / stride, config.confidence)
    rules, scenarios = Rules(base_config(config)), ScenarioRules(config)
    temporal = getattr(backend, 'temporal_fall', None)
    if temporal:
        temporal.frames.clear()
        temporal.last_sample = None
    backend.scene_state = None
    backend.objects_state = []
    events, samples = [], []
    object_counts, peak_scores = Counter(), {}
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    seen_people, eligible_people, processed = set(), set(), 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    last_image = None
    try:
        for index in range(int(start * fps), min(count, int(np.ceil(end * fps)))):
            ok, frame = cap.read()
            if not ok:
                break
            if (index - int(start * fps)) % stride:
                continue
            now = index / fps - start
            people = backend.predict(frame)
            if callable(getattr(backend, 'predict_temporal', None)):
                backend.predict_temporal(frame, now)
            people = tracker.update(people, frame.shape, now)
            objects = getattr(backend, 'objects_state', [])
            for obj in objects:
                object_counts[obj.label] += 1
                peak_scores[obj.label] = max(peak_scores.get(obj.label, 0), float(obj.confidence))
            for p in people:
                if p.track_id > 0 and p.observed:
                    seen_people.add(p.track_id)
                    if p.alert_eligible:
                        eligible_people.add(p.track_id)
            observed, details = rules.step(people, now, frame.shape[1], frame.shape[0], backend.scene_state)
            extra, active = scenarios.step(people, objects, now, frame.shape[1], frame.shape[0], details)
            found = observed + extra
            for tid, kinds in active.items():
                if tid in details:
                    details[tid]['active'].extend(kinds)
            rendered = draw(frame, people, details, config, now, getattr(backend, 'scene_state', None),
                            [key[1] for key, gate in rules.gates.items() if key[0] == 0 and gate.active])
            rendered = annotate_objects(rendered, objects, scenarios.ppe_status, people, scenarios.ppe_details, scenarios.state_rows)
            for event in found:
                event = dict(event)
                filename = f'event-{len(events)+1}-{event["kind"]}.jpg'
                cv2.imwrite(str(output / filename), rendered)
                event['audit_evidence'] = str((output / filename).relative_to(ROOT)).replace('\\', '/')
                events.append(event)
            samples.append({'time_s': round(now, 3), 'people': len(people),
                            'eligible': sum(p.track_id > 0 and p.observed and p.alert_eligible for p in people),
                            'object_labels': dict(Counter(o.label for o in objects)),
                            'scene_state': dict(backend.scene_state or {}),
                            'ppe': scenarios.ppe_details, 'new_events': [e['kind'] for e in found]})
            last_image = rendered
            processed += 1
            if processed % 30 == 0:
                print('PROGRESS', case['id'], processed, round(now, 2), flush=True)
    finally:
        cap.release()
    if last_image is not None:
        cv2.imwrite(str(output / 'last.jpg'), last_image)
    dump(output / 'observations.json', samples)
    matching = [e for e in events if e['kind'] in case['event_kinds']]
    expected = bool(case['expected_event'])
    passed = bool(matching) == expected and processed > 0
    return {**case, 'status': 'completed', 'passed': passed,
            'source_sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
            'source_fps': fps, 'source_duration_s': count/fps,
            'processed_frames': processed, 'sample_fps_actual': fps/stride,
            'wall_s': round(time.perf_counter()-began, 3),
            'measured_person_ids': sorted(seen_people), 'eligible_person_ids': sorted(eligible_people),
            'object_observations': dict(object_counts), 'object_peak_scores': peak_scores,
            'events': events, 'matching_events': len(matching),
            'config': config.model_dump(), 'backend': backend.name,
            'observations': str((output/'observations.json').relative_to(ROOT)).replace('\\', '/')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT/'docs/pattern-test-cases-20261025.json')
    parser.add_argument('--pattern', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'data/cache/pattern-audit-20261025')
    parser.add_argument('--private-report', action='store_true', help='Keep reports and all derived data in the ignored local output folder')
    args = parser.parse_args()
    args.output = args.output.resolve()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    cases = [c for c in json.loads(args.cases.read_text(encoding='utf-8'))['cases'] if c['pattern'] == args.pattern]
    if not cases:
        raise ValueError('No frozen cases for '+args.pattern)
    rows, backend, previous = [], None, None
    report_path = (args.output if args.private_report else ROOT/'docs')/f'pattern-audit-{args.pattern}-20261025.json'
    for case in cases:
        config = RunConfig(**{'fast_motion': False, **case['config']})
        # Zone geometry does not configure NN; reuse only when all NN settings match.
        signature = config.model_dump(exclude={'camera','zones','calibration'})
        if signature != previous:
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print('MODEL_SETUP', args.pattern, flush=True)
            backend = create_backend(MODEL, config.confidence, config.fall_detection, config=config)
            previous = signature
        try:
            row = run_case(case, backend, config, args.output/case['id'])
        except Exception as exc:
            row = {**case, 'status': 'error', 'passed': False, 'error': str(exc)}
        rows.append(row)
        weights = [Path(MODEL), ROOT/'data/models/person-yolo12m.pt', ROOT/'data/models/yolo12m.pt']
        weights += [ROOT/'data/models'/name for name in required_models(config)]
        extra = []
        if config.helmet_detection:
            extra += ['ppe-head.onnx', 'ppe-head-classifier.onnx']
        if config.vest_detection:
            extra += ['ppe-vest.onnx', 'ppe-torso-classifier.onnx', 'signal-clothing.onnx']
        if config.fire_detection:
            extra += ['fire-dfire.onnx']
        if config.fall_detection:
            extra += ['fall-resnet18.pt', 'fall-temporal.pt']
        weights += [ROOT/'data/models'/name for name in extra]
        report = {'created_at': datetime.now(timezone.utc).isoformat(), 'pattern': args.pattern,
                  'protocol': 'Frozen public-video cases; isolated enabled pattern; deployed weights and thresholds; no DB mutation or fitting.',
                  'limitations': ['Selected development regressions, not a blind benchmark or quality guarantee.',
                                  'Expected labels are case-level presence/absence, not expert-complete intervals.',
                                  'Native source resolution; original video timestamps; processing speed is not LIVE presentation speed.'],
                  'weights': [{'file': p.name, 'sha256': hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in weights if p.exists()],
                  'rules_sha256': {cls.__name__: hashlib.sha256(inspect.getsource(cls).encode()).hexdigest() for cls in (Rules, ScenarioRules, PeopleTracker)},
                  'rows': rows}
        dump(report_path, report)
        print('CASE_DONE', row['id'], row['status'], 'PASS' if row['passed'] else 'FAIL',
              row.get('matching_events'), row.get('wall_s'), flush=True)


if __name__ == '__main__':
    main()
