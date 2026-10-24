"""Continue our cigarette weights on train-only resolution variants."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='1');p.add_argument('--batch',type=int,default=16)
    p.add_argument('--minutes',type=float,default=40);p.add_argument('--epochs',type=int,default=80);a=p.parse_args()
    a.data=a.data.resolve();a.weights=a.weights.resolve();a.output=a.output.resolve()
    import torch,cv2,yaml,shutil
    from PIL import Image
    from ultralytics import YOLO,settings
    from training_checkpoint import atomic_copy
    torch.set_num_threads(2);cv2.setNumThreads(1)
    dict.update(settings,{k:False for k in ['wandb','mlflow','comet','clearml','neptune','raytune','hub'] if k in settings})
    a.output.mkdir(parents=True,exist_ok=True);data=a.output/'dataset'
    for split in ['train','val','test']:
        for kind in ['images','labels']:(data/kind/split).mkdir(parents=True,exist_ok=True)
        for image in sorted((a.data/'images'/split).glob('*.jpg')):
            label=a.data/'labels'/split/(image.stem+'.txt')
            for scale in ([1.,.5,.25] if split=='train' else [1.]):
                name=image.stem+('-scale'+str(scale) if scale!=1 else '')
                if scale==1:shutil.copyfile(image,data/'images'/split/(name+'.jpg'))
                else:
                    with Image.open(image) as im:
                        im.resize((max(24,round(im.width*scale)),max(24,round(im.height*scale))),Image.Resampling.BILINEAR).save(data/'images'/split/(name+'.jpg'),quality=95)
                shutil.copyfile(label,data/'labels'/split/(name+'.txt'))
    config=a.output/'data.yaml';config.write_text(yaml.safe_dump({'path':str(data.resolve()),'train':'images/train','val':'images/val','test':'images/test','names':{0:'cigarette'}}))
    model=YOLO(str(a.weights));clock={}
    def begin(trainer):clock['end']=time.monotonic()+a.minutes*60
    def limit(trainer):
        if time.monotonic()>=clock['end']:trainer.stop=True
    def save(trainer):
        if not trainer.last.exists():return
        target=trainer.last.with_name('resume-last.pt');atomic_copy(trainer.last,target)
        state=torch.load(target,map_location='cpu',weights_only=False)  # Our own trainer output.
        if not state.get('optimizer',{}).get('state'):raise ValueError('No optimizer moments were preserved; no verified training step')
        states=len(state['optimizer']['state'])
        del state
        proof={'epoch_index':int(trainer.epoch),'optimizer_present':True,'optimizer_parameter_states':states,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        (target.parent/'checkpoint-state.json').write_text(json.dumps(proof));print('CIGARETTE_CHECKPOINT_VERIFIED',json.dumps(proof),flush=True)
    model.add_callback('on_train_start',begin);model.add_callback('on_train_batch_end',limit);model.add_callback('on_model_save',save)
    model.train(data=str(config),epochs=a.epochs,imgsz=640,batch=a.batch,device=a.device,workers=2,amp=True,
        project=str(a.output),name='cigarette',exist_ok=True,seed=42,patience=12,optimizer='AdamW',lr0=.0002,
        warmup_bias_lr=0.,cos_lr=True,close_mosaic=10,save_period=-1,cache=False)
    model=YOLO(str(a.output/'cigarette/weights/best.pt'))
    metrics=model.val(data=str(config),split='val',device=a.device,imgsz=640,batch=a.batch,verbose=False)
    report={'task':'cigarette object','validation':metrics.results_dict,'status':'candidate_not_deployed',
        'initialization':'Warm start from our short YOLO12m cigarette experiment; new optimizer for expanded train-only resolution variants.',
        'warning':'No test leakage through resolution variants; variants only in train. Object scores are not industrial smoking event scores.'}
    (a.output/'candidate.json').write_text(json.dumps(report,indent=2,default=float))
    exported=Path(model.export(format='onnx',imgsz=640,batch=1,dynamic=False,simplify=False,opset=12,nms=False,device='cpu'))
    atomic_copy(exported,a.output/'cigarette.onnx');print('CIGARETTE_CANDIDATE_READY',json.dumps(report,default=float),flush=True)


if __name__=='__main__':main()
