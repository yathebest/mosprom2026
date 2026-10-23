"""Static RGB ResNet18 export and numerical parity on real training frames.

This prepares an ONNX reference; it does not claim LinQ compilation/support.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort
import torch

from app.body_classifier import BodyClassifier


def main():
    torch.set_num_threads(4)
    weights=ROOT/'data/models/fall-resnet18.pt'
    body=BodyClassifier(weights)
    target=weights.with_suffix('.onnx')
    example=torch.zeros(1,3,224,224)
    torch.onnx.export(body.model,example,str(target),opset_version=12,dynamo=False,
                      input_names=['rgb_normalized'],output_names=['body_logits'],
                      dynamic_axes=None)
    onnx.checker.check_model(onnx.load(str(target)))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(target),sess_options=options,providers=['CPUExecutionProvider'])
    tests=[]
    calibration=ROOT/'data/linq-calibration';calibration.mkdir(exist_ok=True)
    for sequence in ['fall-01','adl-02']:
        cap=cv2.VideoCapture(str(ROOT/f'data/datasets/urfd/{sequence}-cam0-rgb.mp4'))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in [0,count//2,count-1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'Cannot decode {sequence}:{index}')
            rgb=Image.fromarray(bgr[:,:,::-1]).resize((224,224),Image.Resampling.BILINEAR)
            x=np.ascontiguousarray((np.asarray(rgb,np.float32).transpose(2,0,1)/255-body.mean)/body.std)[None]
            pt=body.model(torch.from_numpy(x)).detach().numpy()
            output=session.run(None,{'rgb_normalized':x})[0]
            error=float(np.max(np.abs(pt-output)))
            if error>1e-4:raise AssertionError(f'ONNX numerical mismatch {error}')
            np.savez_compressed(calibration/f'{sequence}-{index:04d}.npz',input=x,expected_logits=pt)
            tests.append({'sequence':sequence,'frame':index,'max_abs_logit_error':error,
                          'pytorch_lying_probability':body.predict(bgr)})
        cap.release()
    report={'status':'passed','opset':12,'input_shape':[1,3,224,224],
            'output_shape':[1,2],'input':'RGB, PIL bilinear resize, float32 /255, ImageNet mean/std, NCHW',
            'output':'two logits: nonlying, lying; apply softmax then temporal Rules',
            'checkpoint_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
            'onnx_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'parity_tests':tests,'linq_compiled':False,'linq_benchmarked':False,
            'calibration_note':'Six training examples are a parity smoke test, not a sufficient representative quantization dataset.'}
    (ROOT/'docs/body_onnx_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
