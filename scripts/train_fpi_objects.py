"""Fine-tune YOLO12m phone/head with observable local-input single-GPU training.

Phone remains class 67; class 0 is a neutral head proposal, NEVER person.
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
    p.add_argument('--resume',action='store_true');p.add_argument('--no-amp',action='store_true')
    p.add_argument('--local-data',type=Path);p.add_argument('--io-workers',type=int,default=8)
    p.add_argument('--loader-workers',type=int,default=2);p.add_argument('--prepare-test',action='store_true');a=p.parse_args()
    a.data=a.data.resolve();a.weights=a.weights.resolve();a.output=a.output.resolve()
    if ',' in a.device:raise ValueError('Use independent single-GPU jobs: callbacks/time limits must not be dropped by DDP')
    if not 1<=a.io_workers<=16:raise ValueError('Use 1..16 bounded I/O workers')
    a.output.mkdir(parents=True,exist_ok=True)
    from training_observability import Progress,materialize_fpi,atomic_json
    progress=Progress(a.output/'training-progress.json')
    try:
        progress.update('imports',force=True)
        import torch,cv2,yaml
        from ultralytics import YOLO,settings
        from training_checkpoint import atomic_copy
        torch.set_num_threads(2);cv2.setNumThreads(1)
        # The loader must not fork a process with initialized CUDA/OpenMP or
        # the progress heartbeat thread. Windows already uses this same path.
        torch.multiprocessing.set_start_method('spawn',force=True)
        dict.update(settings,{k:False for k in ['wandb','mlflow','comet','clearml','neptune','raytune','hub'] if k in settings})
        # One root resolve, bounded parallel reads, then no training image/label
        # access through remote symlinks. Test is not staged unless requested.
        mapped=(a.local_data or a.output/'mapped-data').resolve()
        staged=materialize_fpi(a.data,mapped,progress,workers=a.io_workers,include_test=a.prepare_test)
        atomic_json(a.output/'input-materialization.json',staged)
        progress.update('model_load',weights=a.weights.name,force=True)
        model=YOLO(str(a.weights));names=dict(model.names);names[0]='head_proposal';names[67]='cell_phone'
        if len(names)!=80:raise ValueError('Expected the official 80-class COCO initialization')
        splits={split:split+'/images' for split in staged['counts']}
        data=a.output/'data.yaml';data.write_text(yaml.safe_dump({'path':str(mapped),**splits,'names':names}),encoding='utf-8')
        deadline={};counters={'batch':0,'epoch':0,'val_batch':0,'validation_scope':'epoch'}
        def begin(trainer):
            deadline['at']=time.monotonic()+a.minutes*60
            progress.update('train',epoch=trainer.start_epoch+1,batches_total=len(trainer.train_loader),training_budget_minutes=a.minutes,force=True)
        def epoch(trainer):
            counters.update(batch=0,epoch=int(trainer.epoch)+1)
            progress.update('train',epoch=counters['epoch'],batches_completed=0,batches_total=len(trainer.train_loader),force=True)
        def batch_start(trainer):
            progress.update('batch',epoch=counters['epoch'],batches_completed=counters['batch'],batch_current=counters['batch']+1,batches_total=len(trainer.train_loader),phase='forward_backward')
        def batch_end(trainer):
            counters['batch']+=1
            progress.update('batch',epoch=counters['epoch'],batches_completed=counters['batch'],batches_total=len(trainer.train_loader),phase='completed')
            if time.monotonic()>=deadline['at']:trainer.stop=True
        def val_start(validator):
            counters['val_batch']=0
            progress.update('validation',scope=counters['validation_scope'],epoch=counters['epoch'],batches_completed=0,batches_total=len(validator.dataloader),force=True)
        def val_batch(validator):
            counters['val_batch']+=1
            progress.update('validation',scope=counters['validation_scope'],epoch=counters['epoch'],batches_completed=counters['val_batch'],batches_total=len(validator.dataloader))
        def save(trainer):
            if not trainer.last.exists():return
            progress.update('checkpoint',epoch=int(trainer.epoch)+1,phase='copy_and_verify',force=True)
            target=trainer.last.with_name('resume-last.pt');atomic_copy(trainer.last,target)
            state=torch.load(target,map_location='cpu',weights_only=False)  # Our own just-written trainer checkpoint.
            if not state.get('optimizer',{}).get('state'):raise ValueError('No optimizer moments were preserved; no verified training step')
            states=len(state['optimizer']['state']);del state
            proof={'epoch_index':int(trainer.epoch),'optimizer_present':True,'optimizer_parameter_states':states,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
            atomic_json(target.parent/'checkpoint-state.json',proof)
            progress.update('epoch_complete',epoch=int(trainer.epoch)+1,optimizer_parameter_states=states,force=True)
            print('PHONE_HEAD_CHECKPOINT_VERIFIED',json.dumps(proof),flush=True)
        def attach(network,training=True):
            if training:
                for event,callback in [('on_train_start',begin),('on_train_epoch_start',epoch),('on_train_batch_start',batch_start),('on_train_batch_end',batch_end),('on_model_save',save)]:network.add_callback(event,callback)
            network.add_callback('on_val_start',val_start);network.add_callback('on_val_batch_end',val_batch)
        kwargs={'data':str(data),'epochs':a.epochs,'imgsz':640,'batch':a.batch,'device':a.device,'workers':a.loader_workers,
            'amp':not a.no_amp,'project':str(a.output),'name':'phone-head','exist_ok':True,'seed':42,'patience':10,
            'optimizer':'AdamW','lr0':.0005,'warmup_bias_lr':0.,'close_mosaic':10,'save_period':-1,'cache':False}
        if a.resume:
            checkpoint=a.output/'phone-head/weights/resume-last.pt'
            if not checkpoint.exists():raise ValueError('No optimizer checkpoint to resume')
            model=YOLO(str(checkpoint));kwargs['resume']=True
        attach(model)
        progress.update('scan',detail='Ultralytics dataset verification, model/AMP setup and dataloader initialization',loader_workers=a.loader_workers,force=True)
        model.train(**kwargs)
        best=a.output/'phone-head/weights/best.pt';model=YOLO(str(best));counters['validation_scope']='final';attach(model,training=False)
        progress.update('validation',scope='final',force=True)
        metrics=model.val(data=str(data),split='val',imgsz=640,batch=a.batch,device=a.device,workers=a.loader_workers,verbose=False)
        report={'task':'phone and neutral head proposals','architecture':'YOLO12m / 80 outputs',
            'label_mapping':{'source_phone_0':67,'source_head_1':0},'class_names':model.names,
            'validation':metrics.results_dict,'status':'candidate_not_deployed',
            'warning':'Phone/head box scores are not phone-use/PPE event scores. Other COCO classes are not valid deployment classes. Original person weights remain separate.'}
        (a.output/'candidate.json').write_text(json.dumps(report,indent=2,default=float))
        progress.update('export',format='static_fp32_onnx',force=True)
        exported=Path(model.export(format='onnx',imgsz=640,batch=1,dynamic=False,simplify=False,opset=12,nms=False,device='cpu'))
        atomic_copy(exported,a.output/'phone-head.onnx');progress.update('complete',candidate_deployed=False,force=True)
        print('PHONE_HEAD_CANDIDATE_READY',json.dumps(report,default=float),flush=True)
    except BaseException as error:
        progress.update('failed',error_type=type(error).__name__,failed_stage=progress.payload['stage'],force=True)
        raise
    finally:progress.close()


if __name__=='__main__':main()
