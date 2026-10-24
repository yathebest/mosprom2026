"""Install the validated fire-only supplement; preserve baseline fire and smoke."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    report=json.loads((ROOT/'docs/dfire-candidate.json').read_text())
    candidate=ROOT/'data/cache/dfire-models/yolov5s.onnx'
    digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest!=report['onnx_sha256']:raise ValueError('Candidate changed after export')
    for name,minimum in [('fire-candidate-clip-comparison.json',28),('fire-candidate-expanded.json',40)]:
        comparison=json.loads((ROOT/'docs'/name).read_text())
        rows=comparison['models']['dfire_yolov5s']
        if len(rows)<minimum:raise ValueError('Complete the additional source comparison first')
        if any(r['fire_alert'] and not r['declared_fire'] for r in rows):raise ValueError('Unreviewed additional flame alert; candidate not installed')
    from app.scenario_models import OnnxDetector
    detector=OnnxDetector(candidate,.5)
    if detector.names!={0:'smoke',1:'fire'}:raise ValueError('Unexpected class mapping')
    import numpy as np
    if detector.predict(np.zeros((640,640,3),np.uint8)):raise ValueError('Blank-frame control failed')
    target=ROOT/'data/models/fire-dfire.onnx'
    backup=ROOT/'data/model-backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if target.exists():
        backup.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup/target.name)
    temporary=target.with_suffix('.part');shutil.copyfile(candidate,temporary);temporary.replace(target)
    provenance={**report,'status':'installed_fire_only_supplement','file':target.name,
        'accepted_classes':['fire'],'baseline_fire_smoke_preserved':True,
        'source_license':'MIT repository; see https://github.com/pedbrgs/Fire-Detection',
        'validation':['docs/fire-candidate-clip-comparison.json','docs/fire-candidate-expanded.json'],
        'warning':'Small debug source comparisons, not certified factory event accuracy. Allowed hot-work areas require operator zones. LinQ hardware not tested.'}
    (ROOT/'data/models/dfire-installation.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance),flush=True)


if __name__=='__main__':main()
