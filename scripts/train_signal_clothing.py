"""Three-way clothing verifier: visible signal garment / ordinary / unobservable.

Reflective work uniforms are compliant by the owner's stated policy. Upstream
no_vest labels are NOT automatically ordinary-clothing ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from training_checkpoint import atomic_torch_save,atomic_copy
from training_observability import Progress,atomic_json

CLASSES=['hi_vis','ordinary','unobservable']


def balanced_weights(rows,factory_share=.35):
    """Balance classes and source groups; reserve a bounded factory share.

    Otherwise a few reflective uniforms are overwhelmed by upstream bright
    vests. More crops from one source image/video are not independent groups.
    """
    from collections import Counter,defaultdict
    counts=Counter();groups=defaultdict(set);domains=defaultdict(set)
    for row in rows:
        domain='factory' if row['source_group'].startswith('factory:') else 'upstream'
        key=(row['label'],domain,row['source_group']);counts[key]+=1
        groups[row['label'],domain].add(row['source_group']);domains[row['label']].add(domain)
    result=[]
    for row in rows:
        label=row['label'];domain='factory' if row['source_group'].startswith('factory:') else 'upstream'
        mass=(factory_share if domain=='factory' else 1-factory_share) if len(domains[label])==2 else 1.
        result.append(mass/len(CLASSES)/len(groups[label,domain])/counts[label,domain,row['source_group']])
    return result


def classification_report(scores,truth,thresholds):
    import numpy as np
    pred=scores.argmax(1);accepted=scores.max(1)>=np.asarray(thresholds)[pred]
    # An unobservable prediction is an explicit abstention, never a violation.
    result={};matrix=np.zeros((3,4),dtype=int)
    for y,p,ok in zip(truth,pred,accepted):matrix[int(y),int(p) if ok else 3]+=1
    for k,name in enumerate(CLASSES):
        tp=int(((truth==k)&(pred==k)&accepted).sum());fp=int(((truth!=k)&(pred==k)&accepted).sum())
        fn=int(((truth==k)&((pred!=k)|~accepted)).sum());pr=tp/max(1,tp+fp);re=tp/max(1,tp+fn)
        result[name]={'tp':tp,'fp':fp,'fn':fn,'precision':pr,'recall':re,'f1':2*pr*re/max(1e-12,pr+re)}
    return {'classes':result,'confusion_truth_by_prediction_or_abstain':matrix.tolist(),
        'coverage':float(accepted.mean()),'actionable_coverage':float((accepted&(pred!=2)).mean()),
        'macro_f1':float(np.mean([r['f1'] for r in result.values()]))}


def calibrate(scores,truth):
    import numpy as np
    pred=scores.argmax(1);thresholds=[];passed=[]
    for k in range(3):
        choices=[]
        for t in np.linspace(.8,.995,40):
            mask=(pred==k)&(scores[:,k]>=t);n=int(mask.sum());precision=float((truth[mask]==k).mean()) if n else 0.
            if n>=10 and precision>=(.98 if k==1 else .95):choices.append((n,float(t)))
        thresholds.append(max(choices)[1] if choices else 1.01);passed.append(bool(choices))
    return thresholds,passed


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minutes',type=float,default=250);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--device',default='0');p.add_argument('--batch',type=int,default=64);p.add_argument('--resume',type=Path)
    a=p.parse_args();a.data=a.data.resolve();a.weights=a.weights.resolve();a.output=a.output.resolve()
    if a.resume:a.resume=a.resume.resolve()
    a.output.mkdir(parents=True,exist_ok=True);progress=Progress(a.output/'progress.json')
    try:
        import numpy as np
        import torch
        from torchvision import models,transforms
        from torch.utils.data import DataLoader,WeightedRandomSampler,Dataset
        from PIL import Image
        torch.set_num_threads(2);torch.manual_seed(42);random.seed(42);np.random.seed(42)
        torch.multiprocessing.set_start_method('spawn',force=True)
        device=torch.device('cuda:'+a.device if a.device.isdigit() else a.device)
        # Loaders are kept in-process: no fork of initialized CUDA, heartbeat
        # threads or mutable random state. The 224px crops are staged locally.
        normalize=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        base=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])
        augment=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(5,translate=(.04,.04),scale=(.9,1.05)),
            transforms.ColorJitter(.12,.12,.1,.015),transforms.ToTensor(),normalize])
        class Rows(Dataset):
            def __init__(self,split,train=False):
                self.rows=[r for r in map(json.loads,(a.data/'manifest.jsonl').read_text().splitlines()) if r['split']==split]
                self.transform=augment if train else base
                self.targets=[CLASSES.index(r['label']) for r in self.rows]
            def __len__(self):return len(self.rows)
            def __getitem__(self,index):
                row=self.rows[index]
                with Image.open(a.data/row['file']) as im:image=im.convert('RGB')
                return self.transform(image),self.targets[index]
        train=Rows('train',True);val=Rows('val')
        for data in [train,val]:
            counts=np.bincount(data.targets,minlength=3)
            if counts.min()<5:raise ValueError('Each class needs at least five real reviewed samples per training/validation split')
        model=models.resnet50(weights=None)
        checkpoint=torch.load(a.weights,map_location='cpu',weights_only=True)
        state=checkpoint.get('state_dict',checkpoint)
        state={k:v for k,v in state.items() if not k.startswith('fc.')}
        model.fc=torch.nn.Linear(2048,3);model.load_state_dict(state,strict=False)
        # The old binary fc is deliberately discarded: no_vest is not no_hi_vis.
        for name,param in model.named_parameters():param.requires_grad=name.startswith(('layer3.','layer4.','fc.'))
        model.to(device)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5,weight_decay=.001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=2e-6)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        sampling=torch.tensor(balanced_weights(train.rows),dtype=torch.double)
        start_epoch=0;best=-1.;bad=0
        if a.resume:
            saved=torch.load(a.resume,map_location=device,weights_only=True)
            if saved['classes']!=CLASSES:raise ValueError('Resume classes differ')
            model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler']);scheduler.load_state_dict(saved['scheduler'])
            start_epoch=int(saved['next_epoch']);best=saved['best_score'];bad=saved.get('bad_epochs',0)
            torch.set_rng_state(saved['torch_rng'].cpu());random.setstate(saved['python_rng'])
            if device.type=='cuda' and saved.get('cuda_rng'):torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
            prior=a.resume.parent/'best.pt'
            if not prior.exists():raise ValueError('Attach the prior best.pt beside the optimizer checkpoint')
            if prior.resolve()!=(a.output/'best.pt').resolve():atomic_copy(prior,a.output/'best.pt')
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=.03)
        val_loader=DataLoader(val,batch_size=a.batch,num_workers=0)
        def evaluate():
            model.eval();scores=[];truth=[]
            with torch.inference_mode():
                for x,y in val_loader:scores.append(model(x.to(device)).softmax(1).cpu().numpy());truth.extend(y.tolist())
            return np.concatenate(scores),np.asarray(truth)
        def save(path,next_epoch,partial=False):
            payload={'classes':CLASSES,'architecture':'resnet50','state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'scheduler':scheduler.state_dict(),
                'next_epoch':next_epoch,'partial_epoch_replayed_on_resume':partial,'best_score':best,'bad_epochs':bad,
                'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
            atomic_torch_save(payload,path,previous=True)
            print('CHECKPOINT_SAVED',path.name,next_epoch,partial,flush=True)
        deadline=time.monotonic()+a.minutes*60;began=time.monotonic();last_recovery=began
        history=[]
        for epoch in range(start_epoch,a.epochs):
            generator=torch.Generator().manual_seed(42+epoch)
            sampler=WeightedRandomSampler(sampling,len(train),replacement=True,generator=generator)
            loader=DataLoader(train,batch_size=a.batch,sampler=sampler,num_workers=0)
            model.train();model.conv1.eval();model.bn1.eval();model.layer1.eval();model.layer2.eval()
            progress.update('train',epoch=epoch+1,batch=0,total_batches=len(loader),force=True)
            for step,(x,y) in enumerate(loader,1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):loss=loss_fn(model(x.to(device)),y.to(device))
                scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                progress.update('train',epoch=epoch+1,batch=step,total_batches=len(loader),loss=float(loss.detach()))
                if time.monotonic()-last_recovery>=300:
                    save(a.output/'recovery-last.pt',epoch,partial=True);last_recovery=time.monotonic()
                if time.monotonic()>=deadline or (a.output/'STOP').exists():break
            progress.update('validation',epoch=epoch+1,force=True)
            scores,truth=evaluate();metrics=classification_report(scores,truth,[0.,0.,0.])
            score=metrics['macro_f1'];scheduler.step()
            if score>best+1e-5:
                best=score;bad=0
                atomic_torch_save({'state_dict':model.state_dict(),'classes':CLASSES,'epoch':epoch+1},a.output/'best.pt')
            else:bad+=1
            history.append({'epoch':epoch+1,'validation':metrics,'loss_last_batch':float(loss.detach())})
            atomic_json(a.output/'history.json',history);save(a.output/'resume-last.pt',epoch+1)
            print('EPOCH_RESULT',json.dumps(history[-1]),flush=True)
            if time.monotonic()>=deadline or (a.output/'STOP').exists() or (time.monotonic()-began>90*60 and bad>=15):break
        saved=torch.load(a.output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(saved['state_dict'])
        scores,truth=evaluate();thresholds,passed=calibrate(scores,truth)
        report={'task':'signal_clothing','architecture':'ResNet50 / three output logits','classes':CLASSES,
            'validation':classification_report(scores,truth,thresholds),'thresholds':thresholds,
            'validation_precision_gate_passed':passed,'best_epoch':saved['epoch'],'training_budget_minutes':a.minutes,
            'test_evaluated':False,'status':'candidate_not_deployed','linq_compiled':False,
            'policy':'Bright signal vest OR visually identifiable reflective work uniform accepted. Unobservable is abstention.',
            'sampler':'Class and source-group balance; factory share capped at 35% within each class when both domains are present.',
            'warning':'Assistant-reviewed development labels, not independent expert safety certification. A threshold above 1 disables that class when validation support/precision is insufficient.'}
        progress.update('export',force=True);model.eval().cpu()
        target=a.output/'signal-clothing-candidate.onnx'
        torch.onnx.export(model,torch.zeros(1,3,224,224),str(target),input_names=['images'],output_names=['logits'],opset_version=12,dynamo=False)
        import onnx
        graph=onnx.load(target)
        metadata={'class_names':json.dumps(CLASSES),'class_thresholds':json.dumps(thresholds),
            'rgb_mean':json.dumps([.485,.456,.406]),'rgb_std':json.dumps([.229,.224,.225]),
            'crop_policy':'person_body_context_1.05','resize_algorithm':'pil_bilinear',
            'policy':'vest_or_reflective_uniform','candidate_status':'not_deployed'}
        for key,value in metadata.items():entry=graph.metadata_props.add();entry.key=key;entry.value=value
        onnx.checker.check_model(graph);onnx.save(graph,target)
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        session=ort.InferenceSession(str(target),sess_options=options,providers=['CPUExecutionProvider'])
        sample=np.zeros((1,3,224,224),np.float32)
        with torch.inference_mode():reference=model(torch.from_numpy(sample)).numpy()
        actual=session.run(None,{'images':sample})[0]
        error=float(np.abs(reference-actual).max())
        if error>1e-3:raise ValueError('Exported ONNX logits fail FP32 parity')
        report['onnx_fp32_parity_max_error']=error
        report['onnx_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
        atomic_json(a.output/'candidate.json',report);progress.update('complete',force=True)
        print('CANDIDATE_READY',json.dumps(report),flush=True)
    except BaseException as error:progress.update('failed',error=type(error).__name__,force=True);raise
    finally:progress.close()


if __name__=='__main__':main()
