"""Reference export only. Exact opset/graph cuts must match the LinQ SDK."""
import argparse
from pathlib import Path
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--opset",type=int,default=12)
p.add_argument("--size",type=int,default=640)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
model = YOLO(str(root/"data/models/yolov8n-pose.pt"))
output = model.export(format="onnx",imgsz=args.size,batch=1,dynamic=False,
                      simplify=False,nms=False,opset=args.opset,device="cpu")
print("ONNX reference:",output)
print("This is not a compiled LinQ program. Quantize/compile using the supplied LinQ SDK.")
