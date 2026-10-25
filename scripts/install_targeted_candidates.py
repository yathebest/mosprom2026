"""Install reviewed targeted candidates with checksums and reversible backups.

Input is an already inspected local extraction, never executable archive content.
Quality reports must come from a separate same-data baseline comparison.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_onnx(path, expected_sha, task):
    if digest(path) != expected_sha:
        raise ValueError('Candidate ONNX checksum differs from export report')
    import onnx
    graph = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(graph)
    meta = {p.key: p.value for p in graph.metadata_props}
    if task == 'person':
        names = ast.literal_eval(meta.get('names', '{}'))
        label = names.get(0, names.get('0')) if isinstance(names, dict) else names[0] if names else None
        if label != 'person':
            raise ValueError('Class zero must be person, never head_proposal')
    else:
        thresholds = json.loads(meta.get('class_thresholds', '[]'))
        if (json.loads(meta.get('class_names', '[]')) != ['hi_vis', 'ordinary', 'unobservable']
                or meta.get('policy') != 'vest_or_reflective_uniform'
                or meta.get('crop_policy') != 'person_body_context_1.05'
                or not isinstance(thresholds, list) or len(thresholds) != 3):
            raise ValueError('Clothing labels, policy or crop contract differs')
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or not .8 <= value <= 1.01 for value in thresholds):
            raise ValueError('Clothing thresholds must be finite numbers in [0.8, 1.01]')
        if thresholds[1] <= 1:
            raise ValueError('This release permits positive-only clothing verification')
    return meta


def install(person=None, clothing=None, reports=None, models=None):
    models = Path(models or ROOT / 'data/models')
    reports = reports or {}
    plan = []
    decisions = {}
    if person:
        person = Path(person)
        comparison = json.loads(Path(reports['person']).read_text(encoding='utf-8'))
        if comparison.get('deployment_decision') != 'accept':
            raise ValueError('Person comparison did not approve deployment')
        candidate = json.loads((person / 'candidate.json').read_text(encoding='utf-8'))
        exported = person / 'person-yolo12m-candidate.onnx'
        verified_onnx(exported, candidate['onnx_sha256'], 'person')
        checkpoint = person / 'person/weights/best.pt'
        sys.path.insert(0, str(ROOT))
        from app.inference import validate_person_class
        from ultralytics import YOLO
        detector = YOLO(str(checkpoint), task='detect')
        validate_person_class(detector)
        del detector
        plan += [(checkpoint, 'person-yolo12m.pt'), (exported, 'person-yolo12m.onnx')]
        decisions['person'] = {'mode': 'adapted_person_plus_existing_coco_ensemble',
                               'comparison': str(Path(reports['person']).resolve()),
                               'candidate': candidate}
    if clothing:
        clothing = Path(clothing)
        comparison = json.loads(Path(reports['clothing']).read_text(encoding='utf-8'))
        if comparison.get('deployment_decision') != 'accept_positive_only':
            raise ValueError('Clothing comparison did not approve positive-only deployment')
        candidate = json.loads((clothing / 'candidate.json').read_text(encoding='utf-8'))
        exported = clothing / 'signal-clothing-candidate.onnx'
        verified_onnx(exported, candidate['onnx_sha256'], 'clothing')
        plan += [(exported, 'signal-clothing.onnx')]
        decisions['clothing'] = {'mode': 'positive_only; ordinary class disabled; abstain on uncertain clothing',
                                 'comparison': str(Path(reports['clothing']).resolve()),
                                 'candidate': candidate}
    if not plan:
        raise ValueError('Select at least one independently evaluated candidate')
    models.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = ROOT / 'data/cache' / ('models-before-targeted-' + stamp)
    backup.mkdir(parents=True)
    prior_report = models / 'targeted-installation.json'
    if prior_report.exists():
        shutil.copy2(prior_report, backup / prior_report.name)
    existed = {}
    installed = []
    staged_paths = []
    try:
        for source, name in plan:
            target = models / name
            existed[name] = target.exists()
            if target.exists():
                shutil.copy2(target, backup / name)
            staged = models / (name + '.installing')
            staged_paths.append(staged)
            shutil.copyfile(source, staged)
            if digest(staged) != digest(source):
                raise ValueError('Staged model checksum differs')
            staged.replace(target)
            installed.append(name)
    except BaseException:
        for name in installed:
            if existed[name]:
                shutil.copy2(backup / name, models / name)
            else:
                (models / name).unlink(missing_ok=True)
        for staged in staged_paths:
            staged.unlink(missing_ok=True)
        raise
    report = {'installed_utc': stamp, 'decisions': decisions, 'backup_directory': str(backup),
              'files': {name: {'sha256': digest(models / name), 'bytes': (models / name).stat().st_size}
                        for _, name in plan},
              'linq_compiled': False, 'event_accuracy_measured': False}
    prior_report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--person', type=Path)
    parser.add_argument('--person-comparison', type=Path)
    parser.add_argument('--clothing', type=Path)
    parser.add_argument('--clothing-comparison', type=Path)
    parser.add_argument('--models', type=Path, default=ROOT / 'data/models')
    args = parser.parse_args()
    install(args.person, args.clothing,
            {'person': args.person_comparison, 'clothing': args.clothing_comparison}, args.models)
