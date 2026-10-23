"""Download the official Ultralytics checkpoint and record its hash."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
models = ROOT/"data"/"models"
models.mkdir(parents=True,exist_ok=True)
os.chdir(models)
from ultralytics import YOLO

YOLO("yolov8n-pose.pt")
YOLO("yolo12m.pt")
path = models/"yolov8n-pose.pt"
manifest = {"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "source":"https://github.com/ultralytics/assets/releases", "license":"AGPL-3.0 (Ultralytics)",
            "purpose":"Pretrained COCO person/keypoint model; no custom training"}
strong=models/'yolo12m.pt'
manifest['person_detector']={'file':strong.name,'sha256':hashlib.sha256(strong.read_bytes()).hexdigest(),
                            'source':'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12m.pt',
                            'purpose':'Independent COCO person detector; no PPE fine-tuning',
                            'license':'AGPL-3.0 (Ultralytics)'}
(models/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(json.dumps(manifest,indent=2))
