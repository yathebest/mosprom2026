"""Compare supplied numpy decode to the matching full person ONNX, without TPU."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
EXPECTED_SHA='b92dd8b59bf461d1463e32333eb0e63135857699054bccc5a0057222d00c33fe'


def main():
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    from app.scenario_models import letterbox_tensor
    from my_linq import CUTS,_decode,_validated_tensors
    source=ROOT/'data/models/person-yolo12m.onnx'
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if sha!=EXPECTED_SHA:raise ValueError('Person ONNX differs from the source specified by the integration kit')
    graph=onnx.load(source)
    for name,shape in zip(CUTS,[(1,2,8400),(1,2,8400),(1,80,8400)]):
        graph.graph.output.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,shape))
    cap=cv2.VideoCapture(str(ROOT/'data/datasets/urfd/fall-01-cam0-rgb.mp4'))
    ok,frame=cap.read();cap.release()
    if not ok:raise RuntimeError('Public verification video is missing')
    tensor,_=letterbox_tensor(frame,(640,640))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(graph.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    values=session.run(None,{session.get_inputs()[0].name:tensor})
    actual=values[0][0].T;data=dict(zip(CUTS,values[1:]))
    decoded=_decode(*_validated_tensors(data))
    np.testing.assert_allclose(decoded,actual,rtol=1e-5,atol=1e-4)
    folder=ROOT/'data/cache/linq-offline';folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'reference-person.npz',ref_div=values[1],ref_sub=values[2],ref_sig=values[3],
                        output0=values[0],tensor=tensor,source_sha=sha)
    report={'mode':'local FP32 ONNX parity, not TPU','source_sha256':sha,
        'max_absolute_error':float(np.max(np.abs(decoded-actual))),'output_shape':list(decoded.shape)}
    (folder/'decode-parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report));print('OK: supplied numpy decode matches the person ONNX tail')


if __name__=='__main__':main()
