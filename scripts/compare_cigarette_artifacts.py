"""Compare saved cigarette candidates on validation, without reading test data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.scenario_models import OnnxDetector, iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT/'data/cache/smoking-yolo12m/dataset')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.data/'images/val').glob('*.jpg'))
    if not files:
        raise ValueError('Validation images not found; test is never used as fallback')
    import numpy as np
    cv2.setNumThreads(1)
    report = {'split': 'validation', 'confidence': .5, 'iou': .5, 'images': len(files),
              'test_evaluated': False, 'deployed': False, 'models': {},
              'limitations': ['CigDet image object boxes, not industrial smoking episodes.',
                              'Upstream baseline training overlap is unknown.',
                              'Downscaling is a synthetic small-object stress test.',
                              'This validation has been used for candidate selection.']}
    for name, path in [('baseline', ROOT/'data/models/smoking.onnx'), ('candidate', args.candidate)]:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        model = OnnxDetector(path, .25)
        scales = {}
        for scale in [1., .5, .25]:
            counts = {'tp': 0, 'fp': 0, 'fn': 0}
            latency = []
            for index, file in enumerate(files):
                image = cv2.imread(str(file))
                h, w = image.shape[:2]
                truth = []
                for line in (args.data/'labels/val'/file.with_suffix('.txt').name).read_text().splitlines():
                    if not line.strip():
                        continue
                    _, x, y, bw, bh = map(float, line.split())
                    truth.append(((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h))
                small = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale)))) if scale != 1 else image
                began = time.perf_counter()
                predictions = [o for o in model.predict(small)
                               if o.label in {'cigarette', 'smoking', 'smooking'} and o.confidence >= .5]
                latency.append((time.perf_counter()-began)*1000)
                used = set()
                for obj in sorted(predictions, key=lambda o: o.confidence, reverse=True):
                    overlap, target = max(((iou(tuple(v/scale for v in obj.box), box), i)
                                          for i, box in enumerate(truth) if i not in used), default=(0, None))
                    if overlap >= .5:
                        used.add(target)
                counts['tp'] += len(used)
                counts['fp'] += len(predictions)-len(used)
                counts['fn'] += len(truth)-len(used)
                if (index+1) % 25 == 0:
                    print(name, scale, index+1, '/', len(files), flush=True)
            precision = counts['tp']/max(1, counts['tp']+counts['fp'])
            recall = counts['tp']/max(1, counts['tp']+counts['fn'])
            scales[str(scale)] = {**counts, 'precision': precision, 'recall': recall,
                                  'f1': 2*precision*recall/max(1e-12, precision+recall),
                                  'p95_ms': float(np.percentile(latency, 95))}
        report['models'][name] = {'file': path.name, 'sha256': digest, 'scales': scales}
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
