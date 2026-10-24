"""Train one PPE task; test is evaluated only after validation selects weights."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from training_checkpoint import atomic_copy,atomic_torch_save

# Sampled train-only QC found a vertically mirrored border labelled as helmet
# although that crop contains only chin/shoulder. Keep raw data immutable and
# exclude all classifier crops from this image (no complete detector task).
QC_EXCLUDED_IMAGE_SHA256={'8ab6abebd308d47cead1c3b6403e4bc767c288766d4113dde197c3609373a0c3'}


def valid_classifier_file(path):
    p=Path(path)
    return p.suffix.lower() in {'.jpg','.jpeg','.png'} and p.stem[:64] not in QC_EXCLUDED_IMAGE_SHA256


def train_detector(args):
    from ultralytics import YOLO,settings
    # Disable optional account integrations only in this process. Do not write
    # the user's persistent Ultralytics settings on a local training smoke.
    dict.update(settings,{key:False for key in ['wandb','mlflow','comet','clearml','neptune','raytune','hub'] if key in settings})
    import yaml
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((Path(args.data)/args.task/'data.yaml').read_text())
    config['path']=str((Path(args.data)/args.task).resolve())
    data_file=output/(args.task+'-data.yaml');data_file.write_text(yaml.safe_dump(config),encoding='utf-8')
    last=output/args.task/'weights/last.pt'
    continuation=None
    if args.continue_checkpoint:
        import torch,shutil
        source=Path(args.continue_checkpoint)
        # A retry must prefer this experiment's optimizer-preserving checkpoint.
        local_resume=last.with_name('resume-last.pt')
        if args.resume and local_resume.exists():source=local_resume
        # These are our own saved Kaggle checkpoints, not arbitrary uploaded pickle files.
        state=torch.load(source,map_location='cpu',weights_only=False)
        if state.get('epoch',-1)<0 or not state.get('optimizer'):
            raise ValueError('Exact continuation requires an unfinished checkpoint with optimizer state')
        if args.freeze:raise ValueError('Changing frozen parameter groups is incompatible with exact optimizer continuation')
        continuation={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                      'previous_completed_epochs':int(state['epoch'])+1,'optimizer_restored':True}
        groups=state['optimizer']['param_groups']
        optimizer_name='MuSGD' if any(g.get('use_muon') for g in groups) else ('AdamW' if 'betas' in groups[0] else 'SGD')
        # auto chooses a different optimizer when a shorter schedule drops below
        # 10k iterations. Pin the saved optimizer family before restoring groups.
        state['train_args'].update(project=str(output.resolve()),name=args.task,data=str(data_file.resolve()),
            save_dir=str((output/args.task).resolve()),epochs=args.epochs,optimizer=optimizer_name)
        continuation['optimizer_family']=optimizer_name
        last.parent.mkdir(parents=True,exist_ok=True)
        atomic_torch_save(state,last)
        old_best=source.parent/'best.pt'
        if old_best.exists() and old_best.resolve()!=(last.parent/'best.pt').resolve():
            atomic_copy(old_best,last.parent/'best.pt')
        history=source.parent.parent/'results.csv'
        if history.exists():
            rows=history.read_text().splitlines()[:int(state['epoch'])+2]
            (output/args.task/'results.csv').write_text('\n'.join(rows)+'\n')
        del state
        args.resume=True
    model=YOLO(str(last) if args.resume and last.exists() else args.weights)
    resume=bool(args.resume and last.exists())
    completed=resume and (model.ckpt.get('epoch',0)<0 or model.ckpt.get('epoch',0)+1>=args.epochs)
    if completed and model.ckpt.get('epoch',0)<0 and model.ckpt.get('train_args',{}).get('epochs')!=args.epochs:
        raise ValueError('A completed checkpoint cannot resume a different epoch budget; use a new output experiment')
    if not completed:
        if args.train_minutes:
            # Ultralytics' time= scheduler computes an epoch budget from zero.
            # A callback limits added training while retaining the saved epoch/optimizer.
            clock={}
            def begin(trainer):clock['deadline']=time.monotonic()+args.train_minutes*60
            def limit(trainer):
                if args.train_minutes and time.monotonic()>=clock['deadline']:trainer.stop=True
            model.add_callback('on_train_start',begin)
            model.add_callback('on_train_batch_end',limit)
        if args.train_minutes:
            def preserve_resume(trainer):
                if trainer.last.exists():
                    resume_target=trainer.last.with_name('resume-last.pt')
                    if resume_target.exists():atomic_copy(resume_target,resume_target.with_name('resume-previous.pt'))
                    atomic_copy(trainer.last,resume_target)
                    import torch
                    check=torch.load(resume_target,map_location='cpu',weights_only=False)
                    if not check.get('optimizer'):raise ValueError('Checkpoint lost optimizer before export')
                    proof={'saved_epoch_index':int(check['epoch']),'optimizer_present':True,
                           'bytes':resume_target.stat().st_size,
                           'sha256':hashlib.sha256(resume_target.read_bytes()).hexdigest()}
                    del check
                    target=resume_target.parent/'checkpoint-state.json';temporary=target.with_suffix('.tmp')
                    temporary.write_text(json.dumps(proof,indent=2));temporary.replace(target)
                    print('PROJECT_CHECKPOINT_VERIFIED',args.task,json.dumps(proof),flush=True)
            model.add_callback('on_model_save',preserve_resume)
        model.train(data=str(data_file),epochs=args.epochs,
                    imgsz=args.imgsz,batch=args.batch,device=args.device,seed=42,
                    project=args.output,name=args.task,exist_ok=True,patience=8 if args.freeze else 30,
                    close_mosaic=min(3 if args.freeze else 10,args.epochs),workers=2,amp=True,
                    save_period=-1,resume=resume,freeze=args.freeze or None,
                    cache='ram' if args.cache_ram else False,
                    time=None)
    best=Path(args.output)/args.task/'weights/best.pt'
    model=YOLO(str(best))
    metrics=model.val(data=str(data_file),split='test',
                      imgsz=args.imgsz,batch=args.batch,device=args.device,
                      project=str(output),name=args.task+'-test',exist_ok=True)
    exported=Path(model.export(format='onnx',imgsz=args.imgsz,batch=1,dynamic=False,
                               simplify=False,opset=12,nms=False,device='cpu'))
    deployment={'people':'person-yolo12m.onnx','head_ppe':'ppe-head.onnx','vest_ppe':'ppe-vest.onnx'}[args.task]
    import shutil
    target=Path(args.output)/deployment;shutil.copyfile(exported,target)
    if args.task=='people':shutil.copyfile(best,Path(args.output)/'person-yolo12m.pt')
    result={'task':args.task,'test_metrics':{k:float(v) for k,v in metrics.results_dict.items()},
            'onnx_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'linq_status':'not_compiled','event_metrics':'not_evaluated',
            'freeze_layers':args.freeze,'training_time_budget_minutes':args.train_minutes,
            'continuation':continuation,
            'actual_epochs':sum(1 for _ in (output/args.task/'results.csv').open())-1}
    (Path(args.output)/(args.task+'-metrics.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')


def train_classifier(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets,models,transforms
    import onnx
    torch.manual_seed(42)
    root=Path(args.data)/(args.task+'_classifier')
    mean=[.485,.456,.406];std=[.229,.224,.225]
    common=[transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)]
    train=datasets.ImageFolder(root/'train',transforms.Compose([
        transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.15,.15,.15,.03),*common[1:]]),is_valid_file=valid_classifier_file)
    val=datasets.ImageFolder(root/'val',transforms.Compose(common),is_valid_file=valid_classifier_file)
    test=datasets.ImageFolder(root/'test',transforms.Compose(common),is_valid_file=valid_classifier_file)
    if train.class_to_idx!=val.class_to_idx or train.class_to_idx!=test.class_to_idx:
        raise ValueError('Every split must have the same labelled classes')
    if min(len(train),len(val),len(test))<20:raise ValueError('Not enough reviewed crops for independent splits')
    device=torch.device('cuda:'+args.device if str(args.device).isdigit() else args.device)
    if args.classifier=='resnet152':
        model=models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    else:model=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc=torch.nn.Linear(model.fc.in_features,len(train.classes));model.to(device)
    if args.classifier_freeze_early:
        for name,parameter in model.named_parameters():
            parameter.requires_grad=name.startswith(('layer4.','fc.'))
    loaders={k:DataLoader(d,batch_size=args.batch,shuffle=k=='train',num_workers=2)
             for k,d in [('train',train),('val',val),('test',test)]}
    weights=np.bincount(train.targets,minlength=len(train.classes))
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(1/np.maximum(weights,1),dtype=torch.float32,device=device))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    best_loss=float('inf');checkpoint=folder/(args.task+'-classifier.pt')
    start_epoch=0;last=folder/(args.task+'-classifier-last.pt')
    if args.resume and last.exists():
        state=torch.load(last,map_location=device,weights_only=True)
        if state['classes']!=train.classes:raise ValueError('Resume class order differs')
        if state.get('architecture',args.classifier)!=args.classifier:raise ValueError('Resume classifier architecture differs')
        if state.get('freeze_early',args.classifier_freeze_early)!=args.classifier_freeze_early:raise ValueError('Resume optimizer parameter groups differ')
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        if state.get('scaler'):scaler.load_state_dict(state['scaler'])
        if state.get('torch_rng') is not None:torch.set_rng_state(state['torch_rng'].cpu())
        if state.get('cuda_rng') and device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        start_epoch=state['epoch'];best_loss=state['best_loss']
    def evaluate(loader):
        model.eval();losses=[];scores=[];targets=[]
        with torch.no_grad():
            for x,y in loader:
                x,y=x.to(device),y.to(device);logits=model(x)
                losses.append(float(loss_fn(logits,y)));scores.extend(logits.softmax(1).cpu().numpy());targets.extend(y.cpu().numpy())
        return float(np.mean(losses)),np.asarray(scores),np.asarray(targets)
    deadline=time.monotonic()+args.train_minutes*60 if args.train_minutes else float('inf')
    actual_epochs=start_epoch
    for epoch in range(start_epoch,args.epochs):
        model.train()
        if args.classifier_freeze_early:
            for name,module in model.named_children():
                if name not in {'layer4','fc'}:module.eval()
        for x,y in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                loss=loss_fn(model(x.to(device)),y.to(device))
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            if time.monotonic()>=deadline:break
        validation_loss,_,_=evaluate(loaders['val'])
        print(json.dumps({'epoch':epoch+1,'val_loss':validation_loss}),flush=True)
        if validation_loss<best_loss:
            best_loss=validation_loss;atomic_torch_save({'state_dict':model.state_dict(),'classes':train.classes},checkpoint)
        atomic_torch_save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'epoch':epoch+1,'best_loss':best_loss,'classes':train.classes,
                    'architecture':args.classifier,'freeze_early':args.classifier_freeze_early,
                    'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},last,previous=True)
        print('PROJECT_CHECKPOINT_SAVED',args.task,'classifier',epoch+1,last.stat().st_size,flush=True)
        actual_epochs=epoch+1
        if time.monotonic()>=deadline:
            print('Training time budget reached; selecting validated checkpoint and exporting',flush=True)
            break
    saved=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(saved['state_dict'])
    _,probabilities,truth=evaluate(loaders['val'])
    choices=[]
    for threshold in np.linspace(.5,.99,50):
        selected=probabilities.max(1)>=threshold;pred=probabilities.argmax(1)
        per_class=[]
        for label in range(len(train.classes)):
            positives=selected & (pred==label);count=int(positives.sum())
            per_class.append((count,int(((truth==label)&positives).sum())/max(count,1)))
        if all(count>=20 and precision>=.95 for count,precision in per_class):
            choices.append((float(selected.mean()),float(threshold)))
    threshold=max(choices)[1] if choices else .8
    _,test_scores,test_truth=evaluate(loaders['test']);test_pred=test_scores.argmax(1)
    # Rejections count as misses for recall on assessable test crops.
    selected=test_scores.max(1)>=threshold;stats={}
    for label,name in enumerate(train.classes):
        tp=int(((test_pred==label)&(test_truth==label)&selected).sum())
        fp=int(((test_pred==label)&(test_truth!=label)&selected).sum())
        fn=int(((test_truth==label)&((test_pred!=label)|~selected)).sum())
        precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
        stats[name]={'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-12),'tp':tp,'fp':fp,'fn':fn}
    name='ppe-head-classifier.onnx' if args.task=='head_ppe' else 'ppe-torso-classifier.onnx'
    target=folder/name;model.eval().cpu()
    torch.onnx.export(model,torch.zeros(1,3,224,224),str(target),input_names=['images'],
                      output_names=['logits'],opset_version=12,dynamo=False)
    graph=onnx.load(target)
    for key,value in {'class_names':json.dumps(train.classes),'rgb_mean':json.dumps(mean),'rgb_std':json.dumps(std),
                      'abstain_threshold':str(threshold),'crop_policy':'proposal_context_1.3',
                      'resize_algorithm':'pil_bilinear'}.items():
        item=graph.metadata_props.add();item.key=key;item.value=value
    onnx.save(graph,target)
    report={'classes':train.classes,'threshold':threshold,'validation_precision_target_met':bool(choices),
            'actual_epochs':actual_epochs,'training_time_budget_minutes':args.train_minutes,
            'frozen_early_layers':args.classifier_freeze_early,
            'test':stats,'test_coverage':float(selected.mean()),'linq_status':'not_compiled',
            'qc_excluded_image_sha256':sorted(QC_EXCLUDED_IMAGE_SHA256),
            'split_crops':{'train':len(train),'val':len(val),'test':len(test)},
            'warning':'Crop classification metrics are not event metrics; unobservable crops need a separate evaluation set.'}
    (folder/(args.task+'-classifier-metrics.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/datasets/ppe-prepared')
    p.add_argument('--task',choices=['people','head_ppe','vest_ppe'],required=True)
    p.add_argument('--mode',choices=['detect','classify'],default='detect')
    p.add_argument('--weights',default='yolo12m.pt');p.add_argument('--classifier',choices=['resnet50','resnet152'],default='resnet50')
    p.add_argument('--epochs',type=int,default=150);p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--device',default='0')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--freeze',type=int,default=0)
    p.add_argument('--cache-ram',action='store_true')
    p.add_argument('--classifier-freeze-early',action='store_true')
    p.add_argument('--train-minutes',type=float,default=0)
    p.add_argument('--continue-checkpoint',help='Continue our saved detector optimizer/epoch state with an added time budget')
    p.add_argument('--output',default='training-output');a=p.parse_args()
    if a.mode=='classify' and a.task=='people':p.error('Classifier is only for head/torso PPE')
    (train_detector if a.mode=='detect' else train_classifier)(a)
