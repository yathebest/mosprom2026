"""Check the documented YOLO12m split and optionally compile it with LinQ SDK.

No compatibility claim is made from ONNX checks alone. Run --compile only in
the separate Linux/Python 3.12 SDK environment. Calibration is train-only NPZ.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
import onnxruntime as ort

CUTS = ['/model.21/Div_1_output_0', '/model.21/Sub_1_output_0',
        '/model.21/Sigmoid_output_0']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(model, output, images=()):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    graph = onnx.load(str(model)); onnx.checker.check_model(graph)
    inputs = [v.name for v in graph.graph.input]
    outputs = [v.name for v in graph.graph.output]
    available = {v for n in graph.graph.node for v in n.output}
    if not set(CUTS) <= available:
        raise ValueError('This export differs from the documented YOLO12m graph: missing cut tensors')
    dims = [[d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.graph.input]
    if len(inputs) != 1 or any(d <= 0 for shape in dims for d in shape):
        raise ValueError('One static input is required')
    head, tail = output/'yolo12m-head.onnx', output/'yolo12m-tail.onnx'
    onnx.utils.extract_model(str(model), str(head), inputs, CUTS)
    onnx.utils.extract_model(str(model), str(tail), CUTS, outputs)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    sessions = [ort.InferenceSession(str(p), options, providers=['CPUExecutionProvider'])
                for p in (model, head, tail)]
    # Random data checks graph wiring only; it is NEVER calibration data.
    tensors = [np.random.default_rng(42).random(dims[0], dtype=np.float32)]
    if images:
        import cv2
        from app.scenario_models import letterbox_tensor
        for path in images:
            image = cv2.imread(str(path))
            if image is None: raise ValueError(f'Cannot read {path}')
            tensors.append(letterbox_tensor(image, (dims[0][3], dims[0][2]))[0])
    parity = []
    for tensor in tensors:
        full = sessions[0].run(None, {inputs[0]: tensor})
        intermediate = sessions[1].run(None, {inputs[0]: tensor})
        restored = sessions[2].run(None, dict(zip(CUTS, intermediate)))
        for actual, expected in zip(restored, full):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
        parity.append(max(float(np.max(np.abs(a-b))) for a,b in zip(full, restored)))
    report = {'model_sha256': sha(model), 'onnx_ir': graph.ir_version,
              'opsets': {x.domain: x.version for x in graph.opset_import},
              'input_names': inputs, 'input_shapes': dims, 'cut_tensors': CUTS,
              'head_sha256': sha(head), 'tail_sha256': sha(tail),
              'parity_max_abs_errors': parity, 'parity_samples': len(tensors),
              'real_image_samples': len(tensors)-1, 'status': 'onnx_split_verified',
              'sdk_compiled': False, 'hardware_verified': False,
              'warning': 'ONNX split parity is not SDK compilation, hardware parity or accuracy evaluation.'}
    return report


def compile_head(output, calibration):
    if platform.system() != 'Linux' or sys.version_info[:2] != (3,12):
        raise RuntimeError('LinQ binary SDK needs Linux x86_64 / Python 3.12')
    from tpu_framework import RegularModel, Network, compiler, TPU_H1M, O1, TpuProgram
    from tpu_framework import preliminary_model_conditioning
    from dnn_quant.common import SupportedLayerTypes as Slt
    from dnn_quant.autoquant.cfg_wrappers.cfg_quant_block_wrappers import FLOAT16, PIECEWISE_LUT
    data = np.load(calibration, allow_pickle=False)
    if 'images' not in data or 'split' not in data or str(data['split'].item()) != 'train':
        raise ValueError('Calibration NPZ must contain images and split="train"')
    images = data['images'].astype(np.float32)
    graph = preliminary_model_conditioning(onnx.load(str(output/'yolo12m-head.onnx')))
    shape = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
    if images.ndim != 4 or list(images.shape[1:]) != shape[1:] or len(images) < 8:
        raise ValueError('Need >=8 preprocessed train samples matching the static CHW input')
    if not np.isfinite(images).all() or images.min()<0 or images.max()>1:
        raise ValueError('Calibration requires finite RGB /255 letterboxed tensors')
    regular = RegularModel(model=graph, operation_cfg={Slt.FW_LAYER_REDUCE_MAX: FLOAT16,
                           Slt.FW_LAYER_REDUCE_MEAN: FLOAT16, Slt.FW_LAYER_REDUCE_SUM: FLOAT16},
                           swish_layer_mode=PIECEWISE_LUT)
    name = graph.graph.input[0].name
    name = regular.anchors_mapping.get(name, name)
    thresholds = regular.calibrate_torch(calibration_data={name: images}, device='cpu')
    regular.dump_thresholds(str(output/'thresholds.json'), thresholds)
    quantized = regular.quantize(thresholds)
    quantized.save(str(output), 'yolo12m-head.qm.onnx')
    network, _ = Network.from_quantized_model(quantized)
    executable, descriptions = compiler.compile_(hardware_parameters=TPU_H1M, network=network, parameters=O1)
    target = output/'yolo12m-head-b1-o1-h1m.tpu'
    TpuProgram.from_executable(executable, descriptions).to_file(str(target))
    return {'sdk_compiled': True, 'status': 'compiled_unverified', 'program_sha256': sha(target),
            'calibration_samples': len(images), 'calibration_sha256': sha(calibration),
            'hardware_verified': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='data/models/yolo12m.onnx')
    parser.add_argument('--output', default='data/linq/yolo12m')
    parser.add_argument('--report', default='docs/yolo12m-compatibility.json')
    parser.add_argument('--image', action='append', default=[])
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--calibration')
    args = parser.parse_args()
    report = preflight(args.model, args.output, args.image)
    code = 0
    if args.compile:
        try:
            if not args.calibration: raise ValueError('--calibration is required')
            report.update(compile_head(Path(args.output), args.calibration))
        except Exception as exc:
            report.update(status='compile_failed', compile_error=f'{type(exc).__name__}: {exc}')
            code = 1
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2)); return code


if __name__ == '__main__':
    raise SystemExit(main())
