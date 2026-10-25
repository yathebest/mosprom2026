"""LinQ TPU backend: people detection via the organizers' remote inference service.

The web service stays local; neural inference goes to the LinQ TPU service the
same way the stand example does: RabbitMQ broker + PyTask serialization +
the compiled .tpu program uploaded to /auto/cache/<LOGIN>/.

Requirements:
  1. VPN connected (broker 10.0.114.30 reachable).
  2. The TPU program uploaded on the stand (notebooks/stand-linq-check.ipynb
     cell 2 copies it to /auto/cache/<LOGIN>/).
  3. The `tpu_pycloud_format` package installed from the organizer ZIP:
     python scripts/install_linq_runtime.py /path/to/tpu_pycloud_format.zip
  4. `python -m pip install -r requirements-linq.txt`.

Usage (PowerShell, from the project root):
  $env:LINQ_ADAPTER = "my_linq:create_backend"
  $env:LINQ_LOGIN   = "your_team_login"        # default: hackathon74291
  $env:LINQ_PROGRAM = "person-yolo12m-head-b1-o1-h1m.tpu"   # or yolo12m-head-b1-o1-h1m.tpu
  python -m uvicorn app.server:app --host 127.0.0.1 --port 8765
Or just run start-linq.cmd.

On the TPU only the neural head runs; decode/NMS and all scenario rules stay
on the host. The fall scene classifier (ResNet18 ONNX) runs locally on CPU so
the fall scenarios keep working; disable with $env:LINQ_FALL_ONNX = "0".
Optional local ONNX scenario models (helmets/vests/phone/fire/...) run on the
host alongside TPU person detection (hybrid mode), so the full site works.
"""
import os
import time
import uuid
import threading
from datetime import datetime, timezone

import numpy as np

RABBITMQ_HOST = '10.0.114.30'
EXCHANGE = 'tpu.direct'
INPUT_QUEUE = 'input'
AMQP_PORT = 5672
CREDENTIALS = ('guest', 'guest')
DEFAULT_PROGRAM = 'person-yolo12m-head-b1-o1-h1m.tpu'
FALLBACK_PROGRAM = 'yolo12m-head-b1-o1-h1m.tpu'
INPUT_TENSOR_NAME = 'images'
CUTS = ['/model.21/Div_1_output_0', '/model.21/Sub_1_output_0', '/model.21/Sigmoid_output_0']
STRIDES = np.repeat([8., 16., 32.], [6400, 1600, 400])[None].astype(np.float32)
TIMEOUT_SECONDS = 30.0
MIN_CONF = 0.1          # keep low-confidence boxes for ByteTrack's second pass
_STATUS_LOCK = threading.Lock()
_STATUS = {'verified': False, 'transport': 'RabbitMQ', 'last_reply_at': None}


def get_status():
    with _STATUS_LOCK:
        return dict(_STATUS)


def _status(**values):
    with _STATUS_LOCK:
        _STATUS.update(values)


def _validated_tensors(data):
    if not isinstance(data, dict):
        raise RuntimeError('LinQ reply must contain tensors, not a remote filesystem path')
    arrays = []
    for name, shape in zip(CUTS, [(1, 2, 8400), (1, 2, 8400), (1, 80, 8400)]):
        if name not in data:
            raise RuntimeError('Missing LinQ tensor: ' + name)
        value = np.asarray(data[name], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise RuntimeError(f'Invalid LinQ tensor {name}: expected {shape}, got {value.shape}')
        arrays.append(value)
    return arrays


def _decode(div, sub, sig):
    """Pure-numpy head decode verified locally: exact match with the ONNX tail."""
    return np.concatenate([np.concatenate([div, sub], 1) * STRIDES, sig], 1)[0].T


class LinQBackend:
    def __init__(self, confidence, login=None, program=None):
        import pika
        from tpu_pycloud_format import (MESSAGE_ID_ATTR_NAME, RESPONSE_ROUTING_KEYS_ATTR_NAME,
                                        TARGET_URI_ATTR_NAME, PyTask, serialize, deserialize)
        self._pika = pika
        import json as _json
        self._json = _json
        self._attrs = (MESSAGE_ID_ATTR_NAME, RESPONSE_ROUTING_KEYS_ATTR_NAME, TARGET_URI_ATTR_NAME)
        self._PyTask = PyTask
        self._serialize = serialize
        self._deserialize = deserialize
        self.login = login or os.getenv('LINQ_LOGIN', 'hackathon74291')
        self.program = program or os.getenv('LINQ_PROGRAM', DEFAULT_PROGRAM)
        if any('/' in value or '\\' in value or '?' in value or value in {'.','..'}
               for value in (self.login,self.program)):
            raise ValueError('LINQ_LOGIN and LINQ_PROGRAM must be simple path components')
        self.program_path = f'/auto/cache/{self.login}/{self.program}'
        self.name = f'LinQ TPU / {self.program} ({self.login})'
        self.model_id = f'linq-tpu-{self.program}'
        self.confidence = confidence
        self.latencies = []
        self.scene_state = None
        self.objects_state = []
        self._objects = None
        self._pose = None
        self._reflections = []
        self._connection = None
        self._fall = None

    def configure_scenarios(self, config):
        """Host-side hybrid: TPU persons + local ONNX scenario detectors."""
        from app.scenario_models import ScenarioModels
        from pathlib import Path
        folder = Path(__file__).resolve().parent / 'data/models'
        self._objects = ScenarioModels(folder, config)
        # TPU boxes need host keypoints for current ID confirmation and phone rules.
        # The pose model only enriches existing TPU detections, not local substitutes.
        self._pose = _LocalPose(folder / 'yolov8n-pose.onnx')
        self.name += ' + host ONNX pose'
        self.model_id += '+host-yolov8n-pose'
        if config.fall_detection and os.getenv('LINQ_FALL_ONNX','1')=='1':
            self._fall=_LocalFallClassifier()
            self.name+=' + host RGB ResNet18'
            self.model_id+='+host-fall-resnet18'
        self._reflections = [z for z in config.zones if z.kind == 'reflection'] if config else []
        providers = set()
        for model in self._objects.models:
            providers.add(model.provider.replace('ExecutionProvider', ''))
        if self._objects.models:
            self.name += ' + ONNX ' + '/'.join(sorted(providers)) + ' scenarios'
            self.model_id += '+' + '+'.join(m.name for m in self._objects.models)
        if self._objects.classifiers:
            self.name += ' + trained PPE classifiers'
            self.model_id += '+' + '+'.join(m.name for m in self._objects.classifiers)

    def _connect(self):
        self._close()
        self._connection = self._pika.BlockingConnection(self._pika.ConnectionParameters(
            RABBITMQ_HOST, AMQP_PORT, credentials=self._pika.PlainCredentials(*CREDENTIALS),
            connection_attempts=1,socket_timeout=5,stack_timeout=10,
            blocked_connection_timeout=TIMEOUT_SECONDS,heartbeat=30))

    def _close(self):
        if self._connection is not None and self._connection.is_open:
            try:
                self._connection.close()
            except Exception:
                pass
        self._connection = None

    def _request(self, tensor):
        """One round-trip to the TPU service. Reconnects once on transport errors."""
        started = time.monotonic()
        for attempt in (0, 1):
            try:
                if time.monotonic()-started>=TIMEOUT_SECONDS:
                    raise TimeoutError('LinQ request deadline exceeded')
                connection = self._connection
                if connection is None or not connection.is_open:
                    self._connect()
                    connection = self._connection
                channel = connection.channel()
                task_id = uuid.uuid4().hex
                reply_queue = f'linq_site_{task_id}'
                channel.queue_declare(queue=reply_queue, auto_delete=True, exclusive=True)
                channel.queue_bind(exchange=EXCHANGE, queue=reply_queue, routing_key=reply_queue)
                mid, rkeys, target = self._attrs
                # v=2 busts the stand-side program cache keyed by URI: same
                # program path, new weights (person-yolo12m targeted 25.10).
                channel.basic_publish(
                    exchange=EXCHANGE,
                    routing_key=INPUT_QUEUE,
                    body=self._serialize(self._PyTask({INPUT_TENSOR_NAME: tensor})),
                    properties=self._pika.BasicProperties(headers={
                        target: f'file://{self.program_path}?v=2&raw=False',
                        rkeys: self._json.dumps([reply_queue]),
                        mid: task_id,
                    }))
                for method, properties, body in channel.consume(
                        reply_queue, auto_ack=True, inactivity_timeout=1.0):
                    if method is None:
                        if time.monotonic() - started > TIMEOUT_SECONDS:
                            raise TimeoutError(f'LinQ service: no reply in {TIMEOUT_SECONDS}s')
                        continue
                    if (getattr(properties,'headers',None) or {}).get(mid)!=task_id:
                        raise RuntimeError('LinQ reply message ID does not match this request')
                    channel.cancel()
                    break
                else:
                    raise RuntimeError('LinQ response stream ended without a reply')
                data, _metadata, worker = self._deserialize(body)
                if not data or worker.get('status_code')!=0 or not worker.get('worker_pid'):
                    raise RuntimeError(f'LinQ worker error: {worker.get("error_message")}')
                _validated_tensors(data)
                elapsed = (time.monotonic() - started) * 1000
                _status(verified=True,last_reply_at=datetime.now(timezone.utc).isoformat(),
                        program=self.program,login=self.login,worker_pid=worker['worker_pid'],
                        status_code=0,roundtrip_ms=round(elapsed,2),last_error=None)
                # Evidence line: the reply carries the remote worker's own pid.
                if not getattr(self, '_logged_first', False):
                    self._logged_first = True
                    print(f'[LinQ] подтверждено удалённым воркером: pid={worker.get("worker_pid")} '
                          f'status_code={worker.get("status_code")} roundtrip={elapsed:.0f}ms', flush=True)
                self.latencies.append(elapsed)
                if len(self.latencies) > 3000:
                    self.latencies = self.latencies[-3000:]
                return data, elapsed
            except (self._pika.exceptions.AMQPError, ConnectionError, TimeoutError) as exc:
                _status(verified=False,last_error=type(exc).__name__+': '+str(exc))
                if attempt:
                    raise RuntimeError('LinQ TPU недоступен: проверьте VPN и 10.0.114.30:5672. '
                                       'Локальная подмена детектора людей не выполняется.') from exc
            except Exception as exc:
                _status(verified=False,last_error=type(exc).__name__+': '+str(exc))
                raise
            finally:
                self._close()

    def predict(self, bgr):
        from app.scenario_models import letterbox_tensor
        from app.core import Person
        from app.inference import suppress_duplicates
        if self._fall is not None:
            self.scene_state = {'lying_probability': self._fall.predict(bgr), 'scope': 'scene'}
        tensor, (scale, pad_l, pad_t) = letterbox_tensor(bgr, (640, 640))
        data, _ms = self._request(tensor)
        out = _decode(*_validated_tensors(data))
        h, w = bgr.shape[:2]
        boxes, scores = out[:, :4], out[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(out)), cls]
        cand = sorted((i for i in range(len(out)) if cls[i] == 0 and conf[i] >= MIN_CONF),
                      key=lambda i: -conf[i])
        picked = []
        for i in cand:
            x, y, bw, bh = boxes[i]
            box = np.array([(x - bw / 2 - pad_l) / scale, (y - bh / 2 - pad_t) / scale,
                            (x + bw / 2 - pad_l) / scale, (y + bh / 2 - pad_t) / scale])
            box = tuple(np.clip(box, [0, 0, 0, 0], [w, h, w, h]).tolist())
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            picked.append(Person(box, float(conf[i]),detection_sources=('person:linq-targeted',)))
        people = suppress_duplicates(picked)
        if self._pose is not None:
            from app.scenario_models import iou
            poses=self._pose.predict(bgr)
            for person in people:
                matches=[pose for pose in poses if iou(pose.box,person.box)>=.3]
                if matches:
                    person.keypoints=max(matches,key=lambda pose:iou(pose.box,person.box)).keypoints.copy()
        if self._objects is not None:
            self.objects_state = self._objects.predict(bgr, people)
        else:
            self.objects_state = []
        return people


class _LocalPose:
    """Host keypoints for ID confirmation, PPE regions and measured hands."""
    def __init__(self,path):
        from app.scenario_models import onnx_session
        if not path.is_file():raise FileNotFoundError('Host pose model missing: '+str(path))
        self.session=onnx_session(path)
        self.input_name=self.session.get_inputs()[0].name

    def predict(self,bgr):
        from app.scenario_models import letterbox_tensor,accelerator_lane
        from app.core import Person
        from app.inference import suppress_duplicates
        tensor,(scale,left,top)=letterbox_tensor(bgr,(640,640))
        with accelerator_lane('CUDAExecutionProvider' in self.session.get_providers()):
            raw=np.asarray(self.session.run(None,{self.input_name:tensor})[0])
        if raw.shape!=(1,56,8400) or not np.isfinite(raw).all():
            raise RuntimeError('Unexpected host pose model output')
        h,w=bgr.shape[:2];rows=raw[0].T;people=[]
        for row in rows[rows[:,4]>=.25]:
            x,y,bw,bh,confidence=row[:5]
            box=tuple(np.clip(((x-bw/2-left)/scale,(y-bh/2-top)/scale,
                (x+bw/2-left)/scale,(y+bh/2-top)/scale),[0,0,0,0],[w,h,w,h]).tolist())
            points=row[5:].reshape(17,3).copy()
            points[:,0]=(points[:,0]-left)/scale;points[:,1]=(points[:,1]-top)/scale
            people.append(Person(box,float(confidence),points))
        return suppress_duplicates(people)


class _LocalFallClassifier:
    """Scene-level fall/lying classifier from ONNX, CPU; keeps fall scenarios alive."""

    def __init__(self):
        import onnxruntime as ort
        from pathlib import Path
        path = Path(__file__).resolve().parent / 'data/models/fall-resnet18.onnx'
        if not path.exists():
            raise FileNotFoundError(f'{path.name} missing; run scripts/download_model.py')
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
        self.std = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]

    def predict(self, bgr):
        import cv2
        rgb = cv2.resize(bgr, (224, 224))[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255
        tensor = np.ascontiguousarray((rgb - self.mean) / self.std, dtype=np.float32)
        logits = np.asarray(self.session.run(None, {self.input_name: tensor})[0])[0]
        values = np.exp(logits - logits.max())
        return float(values[1] / values.sum())


def create_backend(confidence=0.4):
    try:
        import pika  # noqa: F401
    except ImportError:
        raise RuntimeError('pip install pika — LinQ adapter needs the AMQP client')
    try:
        import tpu_pycloud_format  # noqa: F401
    except ImportError:
        raise RuntimeError('Install the organizer package: python scripts/install_linq_runtime.py '
                           '/path/to/tpu_pycloud_format.zip (see README)')
    backend = LinQBackend(confidence=confidence)
    # A self-test request proves connectivity before the site starts serving.
    probe = np.zeros((1, 3, 640, 640), np.float32)
    backend._request(probe)
    return backend


def startup_probe():
    backend=create_backend()
    backend._close()
    return get_status()
