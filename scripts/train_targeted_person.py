"""Bounded YOLO12m person adaptation. A candidate, never an auto-deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from training_checkpoint import atomic_copy
from training_observability import Progress,atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--minutes',type=float,default=250)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--device',default='0')
    p.add_argument('--batch',type=int,default=16);p.add_argument('--resume',type=Path)
    p.add_argument('--no-amp',action='store_true')
    a=p.parse_args();a.data=a.data.resolve();a.weights=a.weights.resolve();a.output=a.output.resolve()
    if a.resume:a.resume=a.resume.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    progress=Progress(a.output/'progress.json')
    try:
        import torch,cv2,yaml
        from ultralytics import YOLO,settings
        torch.set_num_threads(2);cv2.setNumThreads(1)
        torch.multiprocessing.set_start_method('spawn',force=True)
        dict.update(settings,{k:False for k in ['wandb','mlflow','comet','clearml','neptune','raytune','hub'] if k in settings})
        # Retain COCO's initialized person output instead of reinitializing a
        # single-class head. Only class zero is a valid deployment output here.
        model=YOLO(str(a.resume or a.weights));names=dict(model.names)
        if a.resume:
            if model.ckpt.get('epoch',-1)<0 or not model.ckpt.get('optimizer'):
                raise ValueError('Use resume-last.pt with optimizer, not exported best.pt')
        if len(names)!=80 or names.get(0)!='person':raise ValueError('Expected COCO YOLO12m person=0, not phone/head weights')
        data=a.output/'person-data.yaml'
        data.write_text(yaml.safe_dump({'path':str(a.data.resolve()),'train':'images/train','val':'images/val','names':names}),encoding='utf-8')
        if a.resume:
            prior=a.resume.parent/'best.pt';target=a.output/'person/weights/best.pt'
            if not prior.exists():raise ValueError('Attach the prior best.pt beside the optimizer checkpoint')
            if prior.resolve()!=target.resolve():atomic_copy(prior,target)
        clock={};counts={'batch':0,'epoch':0}
        def begin(trainer):clock['end']=time.monotonic()+a.minutes*60
        def epoch(trainer):
            counts.update(batch=0,epoch=int(trainer.epoch)+1)
            progress.update('train',epoch=counts['epoch'],batch=0,total_batches=len(trainer.train_loader),force=True)
        def batch(trainer):
            counts['batch']+=1
            progress.update('train',epoch=counts['epoch'],batch=counts['batch'],total_batches=len(trainer.train_loader))
            if time.monotonic()>=clock['end'] or (a.output/'STOP').exists():trainer.stop=True
        def save(trainer):
            if not trainer.last.exists():return
            target=trainer.last.with_name('resume-last.pt')
            if target.exists():atomic_copy(target,target.with_name('resume-previous.pt'))
            atomic_copy(trainer.last,target)
            state=torch.load(target,map_location='cpu',weights_only=False)  # Our just-created checkpoint.
            if not state.get('optimizer'):raise ValueError('Optimizer missing from checkpoint')
            proof={'epoch_index':int(state['epoch']),'optimizer_present':True,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size}
            del state;atomic_json(target.parent/'checkpoint-state.json',proof)
            print('CHECKPOINT_SAVED',json.dumps(proof),flush=True)
        for event,fn in [('on_train_start',begin),('on_train_epoch_start',epoch),('on_train_batch_end',batch),('on_model_save',save)]:model.add_callback(event,fn)
        # Low-rate adaptation with the earliest features frozen limits forgetting
        # of ordinary/occluded COCO people. Tiny-object variants are train-only.
        progress.update('initializing',force=True)
        model.train(data=str(data),epochs=a.epochs,imgsz=640,batch=a.batch,device=a.device,
            workers=2,cache=False,amp=not a.no_amp,optimizer='AdamW',lr0=0.00008,lrf=.05,
            warmup_epochs=2,warmup_bias_lr=0.,freeze=10,weight_decay=.0005,
            mosaic=.35,close_mosaic=10,mixup=0.,copy_paste=0.,scale=.55,
            translate=.08,fliplr=.5,flipud=0.,degrees=0.,hsv_h=.01,hsv_s=.3,hsv_v=.3,
            patience=15,seed=42,project=str(a.output),name='person',exist_ok=True,
            save_period=-1,resume=bool(a.resume))
        best=a.output/'person/weights/best.pt'
        if not best.exists():raise RuntimeError('No validated best checkpoint; resume weights remain preserved')
        candidate=YOLO(str(best))
        progress.update('final_validation',force=True)
        metrics=candidate.val(data=str(data),split='val',device=a.device,batch=a.batch,workers=2,imgsz=640,verbose=False)
        per_class={str(k):list(map(float,metrics.box.class_result(i))) for i,k in enumerate(metrics.box.ap_class_index)}
        progress.update('export',force=True)
        exported=Path(candidate.export(format='onnx',imgsz=640,batch=1,dynamic=False,nms=False,opset=12,simplify=False,device='cpu'))
        target=a.output/'person-yolo12m-candidate.onnx';atomic_copy(exported,target)
        import onnx
        onnx.checker.check_model(str(target))
        report={'task':'person','architecture':'YOLO12m COCO-80 head; allowed output person=0',
            'validation':{k:float(v) for k,v in metrics.results_dict.items()},'per_class_P_R_AP50_AP5095':per_class,
            'training_budget_minutes':a.minutes,'deployment_allowed_classes':[0],
            'onnx_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'status':'candidate_not_deployed','test_evaluated':False,'linq_compiled':False,'onnx_checker_passed':True,
            'required_gate':'Compare matched people, machine false positives, small/occluded recall, track continuity and latency on fixed reviewed validation. Keep old weights if regression.',
            'warning':'Validation and augmented views are not independent factory event accuracy.'}
        atomic_json(a.output/'candidate.json',report);progress.update('complete',force=True)
        print('CANDIDATE_READY',json.dumps(report,default=float),flush=True)
    except BaseException as e:
        progress.update('failed',error=type(e).__name__,force=True);raise
    finally:progress.close()


if __name__=='__main__':main()
