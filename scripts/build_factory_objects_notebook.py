"""Self-contained Kaggle training with independent GPUs and remote backups."""
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def cell(code):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(keepends=True)}


def main():
    files={name:(ROOT/'scripts'/name).read_text(encoding='utf-8') for name in [
        'train_fpi_objects.py','train_cigarette_candidate.py','training_checkpoint.py','training_observability.py','backup_training.py']}
    setup="""import os, sys, subprocess, json, time, zipfile
from pathlib import Path
OVERALL_STARTED=time.monotonic()
os.environ['PYTHONUNBUFFERED']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.88','onnx==1.19.1','kagglehub==1.0.3'],check=True)
import torch
assert torch.cuda.is_available(), 'Enable a GPU before Save & Run All'
print('CUDA_DEVICES',torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],flush=True)
PROGRAM=Path('/kaggle/working/factory-program');PROGRAM.mkdir(exist_ok=True)
sys.path.insert(0,str(PROGRAM))
OUTPUT=Path('/kaggle/working/factory-object-training');OUTPUT.mkdir(exist_ok=True)
"""
    install='PROGRAM_FILES='+repr(files)+'\nfor name,body in PROGRAM_FILES.items():\n    (PROGRAM/name).write_text(body,encoding="utf-8")\n'
    prepare="""markers=list(Path('/kaggle/input').rglob('FPI_PREPARED_SOURCE.json'))
if not markers:
    archive=next(Path('/kaggle/input').rglob('linq-factory-objects-input.zip'))
    unpack=Path('/kaggle/working/factory-input');unpack.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            assert (unpack/entry.filename).resolve().is_relative_to(unpack.resolve())
        z.extractall(unpack)
    markers=list(unpack.rglob('FPI_PREPARED_SOURCE.json'))
assert len(markers)==1, 'Attach exactly the prepared factory objects dataset'
INPUT=markers[0].parent.parent
source=json.loads(markers[0].read_text())
for split,count in source['counts'].items():
    assert len(list((INPUT/'fpi'/split/'images').glob('*.jpg')))==count
print('INPUT_VERIFIED',source['counts'],flush=True)
BACKUP='yathebest/linq-factory-object-checkpoints-20261024'
# Explicitly disabled by the owner: no write API token is configured.
ENABLE_REMOTE_BACKUP=False
probe=subprocess.run([sys.executable,str(PROGRAM/'backup_training.py'),'--root',str(OUTPUT),'--handle',BACKUP,'--once','--probe']) if ENABLE_REMOTE_BACKUP else None
remote_verified=probe is not None and probe.returncode==0
backup_log=(OUTPUT/'backup-worker.log').open('w')
stop_backup=OUTPUT/'STOP_BACKUP'
stop_backup.unlink(missing_ok=True)
backup=None
if remote_verified:
    backup=subprocess.Popen([sys.executable,str(PROGRAM/'backup_training.py'),'--root',str(OUTPUT),'--handle',BACKUP,'--interval','1200','--stop-file',str(stop_backup)],stdout=backup_log,stderr=subprocess.STDOUT,start_new_session=True)
    print('REMOTE_CHECKPOINT_BACKUP_VERIFIED',BACKUP,flush=True)
else:
    print('REMOTE_BACKUP_NOT_AVAILABLE: epoch optimizer checkpoints remain in /kaggle/working and completed notebook outputs; a destroyed VM is NOT covered.',flush=True)
"""
    run=r"""import re
from training_observability import bounded_tail
PHONE_HEAD_ONLY=True  # Cigarette already completed; this run starts fresh phone/head only.
commands={
 'phone-head':[sys.executable,'-u',str(PROGRAM/'train_fpi_objects.py'),'--data',str(INPUT/'fpi'),'--weights',str(INPUT/'weights/yolo12m.pt'),'--output',str(OUTPUT/'phone-head-experiment'),'--epochs','40','--minutes','210','--batch','16','--device','0','--local-data','/kaggle/temp/linq-factory-fpi','--io-workers','8','--loader-workers','2'],
 'cigarette':[sys.executable,str(PROGRAM/'train_cigarette_candidate.py'),'--data',str(INPUT/'cigdet'),'--weights',str(INPUT/'weights/cigarette-start.pt'),'--output',str(OUTPUT/'cigarette-experiment'),'--epochs','80','--minutes','40','--batch','16','--device','1' if torch.cuda.device_count()>1 else '0']}
if PHONE_HEAD_ONLY:commands.pop('cigarette',None)
processes={};handles={};reported={};statuses={};started_at={};last_log={}
def start(name):
    handles[name]=(OUTPUT/(name+'.log')).open('w');started_at[name]=time.monotonic()
    processes[name]=subprocess.Popen(commands[name],stdout=handles[name],stderr=subprocess.STDOUT,cwd=PROGRAM)
    print('TRAINING_PROCESS_STARTED',name,processes[name].pid,flush=True)
start('phone-head')
if 'cigarette' in commands and torch.cuda.device_count()>1:start('cigarette')
deadline=OVERALL_STARTED+5*3600
try:
    while len(statuses)<len(commands):
        if time.monotonic()>deadline:raise TimeoutError('Overall five-hour wall limit reached; existing checkpoints retained')
        for name,process in list(processes.items()):
            if name in statuses:continue
            progress_path=OUTPUT/('phone-head-experiment' if name=='phone-head' else 'cigarette-experiment')/'training-progress.json'
            if progress_path.exists():
                try:progress=json.loads(progress_path.read_text());print('TRAINING_HEARTBEAT',name,json.dumps(progress),flush=True)
                except (OSError,ValueError):print('PROGRESS_SNAPSHOT_PENDING',name,flush=True)
            else:print('PROCESS_HEARTBEAT',name,'elapsed_s',round(time.monotonic()-started_at[name]),'waiting for first stage record',flush=True)
            # Seek only the final 2 KiB. Never read the entire growing tqdm log.
            tail=bounded_tail(OUTPUT/(name+'.log'),2048)
            tail=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',tail)
            parts=[line.strip() for line in tail.replace('\r','\n').splitlines() if line.strip()]
            latest=parts[-1][-600:] if parts else ''
            if latest and latest!=last_log.get(name):
                last_log[name]=latest;print('PROCESS_LOG_TAIL',name,latest,flush=True)
            if process.poll() is not None:
                statuses[name]={'exit_code':process.returncode};handles[name].flush()
                print('TRAINING_PROCESS_FINISHED',name,process.returncode,flush=True)
                if process.returncode:
                    lines=bounded_tail(OUTPUT/(name+'.log'),8192).splitlines()
                    print('ERROR_TAIL',name,'\n'.join(lines[-12:]),flush=True)
        for csv in OUTPUT.rglob('results.csv'):
            lines=csv.read_text(errors='replace').splitlines()
            if len(lines)>1 and reported.get(str(csv))!=lines[-1]:
                reported[str(csv)]=lines[-1];print('EPOCH_RESULT',str(csv.relative_to(OUTPUT)),lines[-1],flush=True)
        for proof in OUTPUT.rglob('checkpoint-state.json'):
            body=proof.read_text()
            if reported.get(str(proof))!=body:
                reported[str(proof)]=body;print('CHECKPOINT_SAVED',str(proof.relative_to(OUTPUT)),body,flush=True)
        if 'cigarette' in commands and torch.cuda.device_count()==1 and 'phone-head' in statuses and 'cigarette' not in processes:start('cigarette')
        time.sleep(20)
finally:
    for name,process in processes.items():
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    for handle in handles.values():handle.close()
    stop_backup.touch()
    if backup is not None:
        try:backup.wait(timeout=360)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(backup.pid,signal.SIGTERM)
            try:backup.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(backup.pid,signal.SIGKILL);backup.wait()
    backup_log.close()
    final_backup=subprocess.run([sys.executable,str(PROGRAM/'backup_training.py'),'--root',str(OUTPUT),'--handle',BACKUP,'--once']) if remote_verified else None
    status={'jobs':statuses,'remote_final_backup_verified':final_backup is not None and final_backup.returncode==0,'runtime_models_replaced':False}
    (OUTPUT/'RUN_STATUS.json').write_text(json.dumps(status,indent=2))
    print('FINAL_STATUS',json.dumps(status),flush=True)
    with zipfile.ZipFile('/kaggle/working/linq-factory-object-candidates.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for file in OUTPUT.rglob('*'):
            if file.is_file() and (file.suffix in {'.onnx','.json','.csv','.log'} or file.name in {'best.pt','resume-last.pt'}):archive.write(file,str(file.relative_to(OUTPUT)))
assert all(v['exit_code']==0 for v in statuses.values()), 'A task failed; inspect logs and preserved optimizer checkpoints'
print('CANDIDATES_READY_FOR_INDEPENDENT_VALIDATION_NOT_DEPLOYED',flush=True)
"""
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# LinQ Factory Objects 2026\n','Phone/head and cigarette candidates. Existing app weights remain unchanged.\n',
        'Prepared source splits exclude exact/near-image overlap. This does not prove camera-independent event accuracy.\n',
        'Epoch checkpoints include optimizer state and are included in notebook outputs. Remote backup is disabled by the owner until a write API token is configured; a destroyed VM is not covered.\n']}]+[cell(s) for s in [setup,install,prepare,run]],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'data/cache/kaggle-factory-objects/linq-factory-objects-2026.ipynb';target.parent.mkdir(exist_ok=True,parents=True)
    target.write_text(json.dumps(book,ensure_ascii=False,indent=1),encoding='utf-8');print(target)


if __name__=='__main__':main()
