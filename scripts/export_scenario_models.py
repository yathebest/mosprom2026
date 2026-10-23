import importlib,json,hashlib,argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import onnx

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'data/cache/scenario-source'
torch.set_num_threads(4)
def safe_type(name):
    permitted={'NoneType':type(None),'bool':bool,'int':int,'float':float,'str':str,'dict':dict,'list':list,'tuple':tuple,'set':set,'type':type}
    if name not in permitted:raise ValueError('Unapproved serialized type')
    return permitted[name]

parser=argparse.ArgumentParser();parser.add_argument('--only');args=parser.parse_args()
report_path=ROOT/'docs/scenario_exports.json'
reports=json.loads(report_path.read_text()) if report_path.exists() else []
for src,name in [('construction-source.pt','construction-machinery'),('fire-source.pt','fire-smoke'),('coco-yolov8n.pt','coco-objects'),('phone-coco-source.pt','phone-objects')]:
    if args.only and args.only!=name:continue
    path=SOURCE/src;allowed=[]
    for qualified in torch.serialization.get_unsafe_globals_in_checkpoint(str(path)):
        if qualified=='dill._dill._load_type':allowed.append((safe_type,qualified));continue
        if not qualified.startswith(('torch.nn.','ultralytics.nn.')):
            raise ValueError('Unapproved checkpoint global: '+qualified)
        module,attribute=qualified.rsplit('.',1);obj=getattr(importlib.import_module(module),attribute)
        if not isinstance(obj,type):raise ValueError('Unapproved non-class global')
        allowed.append(obj)
    with torch.serialization.safe_globals(allowed):checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=YOLO('yolov8n.yaml');model.model=(checkpoint.get('ema') or checkpoint['model']).float();model.task='detect'
    model.model.eval();model.model.task='detect';model.model.pt_path=str(ROOT/f'data/models/{name}.pt')
    target=ROOT/f'data/models/{name}.onnx'
    generated=Path(model.export(format='onnx',imgsz=640,batch=1,dynamic=False,simplify=False,nms=False,opset=12,device='cpu'))
    if generated.resolve()!=target.resolve():generated.replace(target)
    graph=onnx.load(str(target));onnx.checker.check_model(graph)
    report={'file':target.name,'source_checkpoint':src,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'onnx_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'export':'static FP32 1x3x640x640, opset12, host NMS',
            'classes':model.names,'weights_only_restricted_load':True,'linq_compiled':False}
    reports=[r for r in reports if r['file']!=target.name];reports.append(report);print(json.dumps(report),flush=True)
(ROOT/'docs/scenario_exports.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
