"""Train one PPE task; test is evaluated only after validation selects weights."""
import argparse
import hashlib
import json
from pathlib import Path

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
    model=YOLO(str(last) if args.resume and last.exists() else args.weights)
    resume=bool(args.resume and last.exists())
    completed=resume and model.ckpt.get('epoch',0)<0
    if completed and model.ckpt.get('train_args',{}).get('epochs')!=args.epochs:
        raise ValueError('A completed checkpoint cannot resume a different epoch budget; use a new output experiment')
    if not completed:
        model.train(data=str(data_file),epochs=args.epochs,
                    imgsz=args.imgsz,batch=args.batch,device=args.device,seed=42,
                    project=args.output,name=args.task,exist_ok=True,patience=30,
                    close_mosaic=min(10,args.epochs),workers=2,amp=True,save_period=-1,resume=resume)
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
            'linq_status':'not_compiled','event_metrics':'not_evaluated'}
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
    loaders={k:DataLoader(d,batch_size=args.batch,shuffle=k=='train',num_workers=2)
             for k,d in [('train',train),('val',val),('test',test)]}
    weights=np.bincount(train.targets,minlength=len(train.classes))
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(1/np.maximum(weights,1),dtype=torch.float32,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    best_loss=float('inf');checkpoint=folder/(args.task+'-classifier.pt')
    start_epoch=0;last=folder/(args.task+'-classifier-last.pt')
    if args.resume and last.exists():
        state=torch.load(last,map_location=device,weights_only=True)
        if state['classes']!=train.classes:raise ValueError('Resume class order differs')
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        if state.get('scaler'):scaler.load_state_dict(state['scaler'])
        start_epoch=state['epoch'];best_loss=state['best_loss']
    def evaluate(loader):
        model.eval();losses=[];scores=[];targets=[]
        with torch.no_grad():
            for x,y in loader:
                x,y=x.to(device),y.to(device);logits=model(x)
                losses.append(float(loss_fn(logits,y)));scores.extend(logits.softmax(1).cpu().numpy());targets.extend(y.cpu().numpy())
        return float(np.mean(losses)),np.asarray(scores),np.asarray(targets)
    for epoch in range(start_epoch,args.epochs):
        model.train()
        for x,y in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                loss=loss_fn(model(x.to(device)),y.to(device))
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        validation_loss,_,_=evaluate(loaders['val'])
        print(json.dumps({'epoch':epoch+1,'val_loss':validation_loss}),flush=True)
        if validation_loss<best_loss:
            best_loss=validation_loss;torch.save({'state_dict':model.state_dict(),'classes':train.classes},checkpoint)
        torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'epoch':epoch+1,'best_loss':best_loss,'classes':train.classes},last)
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
    p.add_argument('--output',default='training-output');a=p.parse_args()
    if a.mode=='classify' and a.task=='people':p.error('Classifier is only for head/torso PPE')
    (train_detector if a.mode=='detect' else train_classifier)(a)
