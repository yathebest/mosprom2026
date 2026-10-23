from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
SOURCES=ROOT/'data/cache/scenario-source';SOURCES.mkdir(parents=True,exist_ok=True)
items=[
 ('ppe-safetyvision.onnx','https://huggingface.co/ayushgupta7777/safetyvision-yolov8/resolve/56a71758b55f0e9f2b4b2d6b51a779a1f882da10/v2/best_640.onnx','AGPL-3.0','PPE13, includes direct no-hardhat and no-harness classes'),
 ('smoking.onnx','https://raw.githubusercontent.com/alihassanml/Smoking-detection-yolo11/bbef478fad3d6d18e4a18bf02fadce1a09f55e15/best.onnx','MIT','Upstream smoking detector; task labels inspected from model metadata'),
 ('fire-source.pt','https://raw.githubusercontent.com/luminous0219/fire-and-smoke-detection-yolov8/c30923f568d915a53fcf85989be9fe5d4e030518/weights/best.pt','AGPL-3.0','Fire/smoke weights; restricted weights-only loading before export'),
 ('construction-source.pt','https://huggingface.co/baskarmother/yolov8-ppe-construction/resolve/3213ed51de90cbc76e577e6944e84f7c74343526/best.pt','see upstream model card','Machinery/vehicles/PPE; restricted weights-only loading before export'),
 ('coco-yolov8n.pt','https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt','AGPL-3.0','Official COCO detector for cell phone and supported carried objects')
 ,('phone-coco-source.pt','https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt','AGPL-3.0','Official larger COCO model for phone full-frame and person crops')
]
EXPECTED={'ppe-safetyvision.onnx': 'ea18ae903a566e8fa76f3ee1c503075522dca269269315e9c862efa170430b35', 'smoking.onnx': 'ddc8cae492f0c28e1ccbadb11560de49b436dd4544f4382a361d3b047f02f38f', 'fire-source.pt': 'ac0a10257b2bc1f20c9d957f8adeeb61dd6140322fc19d0b4a116cb491776d16', 'construction-source.pt': '8714b4b2bbde95b3a07dcdbe873995e34742b5ce628464a4da232721d4691ffe', 'coco-yolov8n.pt': 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36', 'phone-coco-source.pt': '1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a'}

def one(item):
    name,url,license,note=item;path=(ROOT/'data/models'/name) if name.endswith('.onnx') else SOURCES/name
    if not path.exists():
        with urlopen(Request(url,headers={'User-Agent':'LinQSafety/1.0'}),timeout=120) as r:
            with path.with_suffix(path.suffix+'.part').open('wb') as f:
                while chunk:=r.read(1024*1024):f.write(chunk)
        path.with_suffix(path.suffix+'.part').replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED[name]:raise ValueError('Model checksum mismatch: '+name)
    if path.stat().st_size<100000:raise ValueError('Unexpected model size '+name)
    result={'file':name,'source':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'license':license,'note':note}
    print(json.dumps(result),flush=True);return result
with ThreadPoolExecutor(max_workers=5) as pool:manifest=list(pool.map(one,items))
(ROOT/'data/models/scenario-sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
