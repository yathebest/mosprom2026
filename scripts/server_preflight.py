"""Read-only environment/weights check before starting an extracted bundle."""
from importlib import metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


if __name__ == '__main__':
    packages = ['fastapi','uvicorn','python-multipart','pydantic','ultralytics',
                'lap','opencv-python','numpy','imageio-ffmpeg','onnxruntime','torch','torchvision']
    versions, missing = {}, []
    for name in packages:
        try: versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError: missing.append(name)
    report = {'python':sys.version.split()[0], 'packages':versions, 'missing':missing}
    if not missing:
        try:
            import cv2
            import torch
            import onnxruntime as ort
        except (ImportError, OSError) as exc:
            report['native_runtime_error'] = str(exc)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(2)
        from app.core import RunConfig
        from app.pattern_policy import catalog
        from app.scenario_models import required_models
        required = {'yolov8n-pose.pt','yolo12m.pt','person-yolo12m.pt','fall-resnet18.pt','fall-temporal.pt',
                    'ppe-head.onnx','ppe-vest.onnx','ppe-head-classifier.onnx','ppe-torso-classifier.onnx'}
        policy = catalog()
        for flag in policy['enabled_flags']:
            if flag.endswith('_detection'):
                required.update(required_models(RunConfig(**{flag:True})))
        absent = sorted(name for name in required if not (ROOT/'data/models'/name).is_file())
        report.update(cuda=torch.cuda.is_available(), onnx_providers=ort.get_available_providers(),
                      active_patterns=[p['id'] for p in policy['patterns'] if p['enabled']],
                      missing_models=absent, opencv=cv2.__version__, hardware_run_performed=False)
        missing.extend(absent)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(missing))
