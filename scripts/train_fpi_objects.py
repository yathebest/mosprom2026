"""Fine-tune YOLO12m for FPI phone/head while preserving COCO head shapes.

Source phone 0 -> COCO phone 67, source head 1 -> output 0. Keep 80
outputs so pretrained phone classification weights are not randomly reset.
This checkpoint must never replace the application's person detector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--minutes',type=float,default=210)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--device',default='0')
    p.add_argument('--resume',action='store_true');p.add_argument('--no-amp',action='store_true');a=p.parse_args()
    a.data=a.data.resolve();a.weights=a.weights.resolve();a.output=a.output.resolve()
    if ',' in a.device:raise ValueError('Use independent single-GPU jobs: callback checkpoints/time limits must survive subprocess launch')
    import torch,cv2,yaml
    from ultralytics import YOLO,settings
    from training_checkpoint import atomic_copy
    torch.set_num_threads(2);cv2.setNumThreads(1)
    dict.update(settings,{k:False for k in ['wandb','mlflow','comet','clearml','neptune','raytune','hub'] if k in settings})
    a.output.mkdir(parents=True,exist_ok=True);model=YOLO(str(a.weights))
    names=dict(model.names);names[0]='head_proposal';names[67]='cell_phone'
    if len(names)!=80:raise ValueError('Expected the official 80-class COCO initialization')
    # Materialize only labels. Images remain read-only source inputs through
    # per-split image path lists; image/label sibling layout stays conventional.
    mapped=a.output/'mapped-data'
    for split in ['train','val','test']:
        images=mapped/split/'images';labels=mapped/split/'labels'
        images.mkdir(parents=True,exist_ok=True);labels.mkdir(exist_ok=True)
        for image in sorted((a.data/split/'images').glob('*.jpg')):
            target=images/image.name
            if not target.exists():
                import os,shutil
                try:target.symlink_to(image.resolve())
                except OSError:shutil.copyfile(image,target)
            source=a.data/split/'labels'/(image.stem+'.txt');rows=[]
            for line in source.read_text().splitlines():
                if not line.strip():continue
                fields=line.split()
                if len(fields)!=5 or fields[0] not in {'0','1'}:raise ValueError('Use normalized reviewed FPI labels')
                fields[0]='67' if fields[0]=='0' else '0';rows.append(' '.join(fields))
            (labels/(image.stem+'.txt')).write_text('\n'.join(rows)+'\n')
    data=a.output/'data.yaml';data.write_text(yaml.safe_dump({'path':str(mapped.resolve()),'train':'train/images','val':'val/images','test':'test/images','names':names}))
    deadline={}
    def begin(trainer):deadline['at']=time.monotonic()+a.minutes*60
    def limit(trainer):
        if time.monotonic()>=deadline['at']:trainer.stop=True
    def save(trainer):
        if not trainer.last.exists():return
        target=trainer.last.with_name('resume-last.pt');atomic_copy(trainer.last,target)
        state=torch.load(target,map_location='cpu',weights_only=False)  # Our own just-written trainer checkpoint.
        if not state.get('optimizer',{}).get('state'):raise ValueError('No optimizer moments were preserved; no verified training step')
        states=len(state['optimizer']['state'])
        del state
        proof={'epoch_index':int(trainer.epoch),'optimizer_present':True,'optimizer_parameter_states':states,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        (target.parent/'checkpoint-state.json').write_text(json.dumps(proof,indent=2))
        print('PHONE_HEAD_CHECKPOINT_VERIFIED',json.dumps(proof),flush=True)
    model.add_callback('on_train_start',begin);model.add_callback('on_train_batch_end',limit);model.add_callback('on_model_save',save)
    kwargs={'data':str(data),'epochs':a.epochs,'imgsz':640,'batch':a.batch,'device':a.device,'workers':2,
        'amp':not a.no_amp,'project':str(a.output),'name':'phone-head','exist_ok':True,'seed':42,'patience':10,
        'optimizer':'AdamW','lr0':.0005,'warmup_bias_lr':0.,'close_mosaic':10,'save_period':-1,'cache':False}
    if a.resume:
        checkpoint=a.output/'phone-head/weights/resume-last.pt'
        if not checkpoint.exists():raise ValueError('No optimizer checkpoint to resume')
        model=YOLO(str(checkpoint));kwargs['resume']=True
        model.add_callback('on_train_start',begin);model.add_callback('on_train_batch_end',limit);model.add_callback('on_model_save',save)
    model.train(**kwargs)
    best=a.output/'phone-head/weights/best.pt';model=YOLO(str(best))
    metrics=model.val(data=str(data),split='val',imgsz=640,batch=a.batch,device=a.device.split(',')[0],verbose=False)
    report={'task':'phone and neutral head proposals','architecture':'YOLO12m / 80 outputs',
        'label_mapping':{'source_phone_0':67,'source_head_1':0},'class_names':model.names,
        'validation':metrics.results_dict,'status':'candidate_not_deployed',
        'warning':'Phone/head box scores are not phone-use/PPE event scores. Other COCO classes in this checkpoint are not valid deployment classes. Original person weights remain separate.'}
    (a.output/'candidate.json').write_text(json.dumps(report,indent=2,default=float))
    exported=Path(model.export(format='onnx',imgsz=640,batch=1,dynamic=False,simplify=False,opset=12,nms=False,device='cpu'))
    atomic_copy(exported,a.output/'phone-head.onnx')
    print('PHONE_HEAD_CANDIDATE_READY',json.dumps(report,default=float),flush=True)


if __name__=='__main__':main()
