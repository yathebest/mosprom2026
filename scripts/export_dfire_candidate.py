"""Export the authors' D-Fire YOLOv5s with restricted checkpoint loading."""
import importlib
import json
import os
from pathlib import Path
import hashlib

os.environ['YOLOv5_AUTOINSTALL']='false'
os.environ['YOLO_AUTOINSTALL']='false'


def main():
    import torch,onnx
    from torch import nn
    root=Path(__file__).resolve().parents[1];source=root/'data/cache/dfire-models/yolov5s.pt'
    expected='7d3082bbe8c3aee97f6a9f31852fa093d181ff296fb737f4e4ab2d08422584e8'
    if not source.exists():
        import requests
        source.parent.mkdir(parents=True,exist_ok=True)
        response=requests.get('https://1drv.ms/u/c/c0bd25b6b048b01d/EeZYmpKPBppNr3lo8oaOqecB9GDj1dDvbogCJyegO0PY1Q?e=u76Iei&download=1',timeout=120)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest()!=expected:raise ValueError('Source download checksum mismatch')
        source.write_bytes(response.content)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Unexpected source weights')
    allowed=[]
    for qualified in torch.serialization.get_unsafe_globals_in_checkpoint(str(source)):
        module,name=qualified.rsplit('.',1)
        if qualified.startswith('torch.nn.'):actual=module
        elif module in {'models.common','models.yolo'}:actual='yolov5.'+module
        else:raise ValueError('Unapproved checkpoint global: '+qualified)
        value=getattr(importlib.import_module(actual),name)
        if not isinstance(value,type):raise ValueError('Only approved NN classes may deserialize')
        allowed.append((value,qualified))
    with torch.serialization.safe_globals(allowed):checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    model=(checkpoint.get('ema') or checkpoint['model']).float().eval()
    for module in model.modules():
        if isinstance(module,nn.Upsample) and not hasattr(module,'recompute_scale_factor'):module.recompute_scale_factor=None
    head=model.model[-1]
    if int(head.nc)!=2:raise ValueError('Expected two fire/smoke classes')
    head.inplace=False;head.export=True
    if not isinstance(head.anchor_grid,list):
        delattr(head,'anchor_grid');head.anchor_grid=[torch.zeros(1) for _ in range(head.nl)]
    class CanonicalOutput(nn.Module):
        def __init__(self,network):super().__init__();self.network=network
        def forward(self,images):
            raw=self.network(images)[0]
            return torch.cat((raw[:,:,:4],raw[:,:,5:]*raw[:,:,4:5]),dim=2).transpose(1,2)
    wrapper=CanonicalOutput(model);torch.set_num_threads(2)
    target=source.with_suffix('.onnx')
    torch.onnx.export(wrapper,torch.zeros(1,3,640,640),str(target),input_names=['images'],output_names=['predictions'],opset_version=12,dynamo=False)
    graph=onnx.load(target);names=model.names
    for key,value in {'names':repr(names),'source':'https://github.com/pedbrgs/Fire-Detection','decoder':'objectness multiplied into class scores; canonical 4+nc output'}.items():
        item=graph.metadata_props.add();item.key=key;item.value=value
    onnx.checker.check_model(graph);onnx.save(graph,target)
    report={'source':'https://github.com/pedbrgs/Fire-Detection','source_sha256':expected,
        'onnx_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'names':names,'status':'candidate_not_deployed',
        'architecture':'YOLOv5s','restricted_weights_only_load':True,'linq_hardware_verified':False}
    (root/'docs/dfire-candidate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
