from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,zipfile

ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'data/cache/scenario-datasets';CACHE.mkdir(parents=True,exist_ok=True);FOLDER=ROOT/'data/datasets/scenario-fixtures';FOLDER.mkdir(parents=True,exist_ok=True)
items=[('construction-ppe.zip','https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip'),
       ('coco128.zip','https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip'),
       ('fire-demo.gif','https://raw.githubusercontent.com/luminous0219/fire-and-smoke-detection-yolov8/c30923f568d915a53fcf85989be9fe5d4e030518/fire%20demo%201.gif'),
       ('smoking-upstream-output.jpg','https://raw.githubusercontent.com/alihassanml/Smoking-detection-yolo11/bbef478fad3d6d18e4a18bf02fadce1a09f55e15/output.jpg')]
def one(item):
    name,url=item;path=(CACHE/name) if name.endswith('.zip') else FOLDER/name
    if not path.exists():
        with urlopen(Request(url,headers={'User-Agent':'LinQSafety/1.0'}),timeout=120) as r,path.with_suffix(path.suffix+'.part').open('wb') as f:
            while chunk:=r.read(1024*1024):f.write(chunk)
        path.with_suffix(path.suffix+'.part').replace(path)
    result={'file':name,'source':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    if name.endswith('.zip'):
        target=ROOT/'data/datasets'
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if not (target/member).resolve().is_relative_to(target.resolve()):raise ValueError('Bad archive path')
            z.extractall(target)
    print(json.dumps(result),flush=True);return result
with ThreadPoolExecutor(max_workers=4) as pool:manifest=list(pool.map(one,items))
(FOLDER/'manifest.json').write_text(json.dumps({'purpose':'Integration demonstrations only; training overlap may exist; not independent metrics','files':manifest},indent=2))
