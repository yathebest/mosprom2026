"""Generate the approved five-task, 4–5 hour continuation notebook."""
import ast
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
cells=[]


def cell(source, markdown=False, hidden=False):
    source=textwrap.dedent(source).strip()
    if not markdown:ast.parse(source)
    result={'cell_type':'markdown' if markdown else 'code','metadata':{},'source':source,'id':f'night-{len(cells):03}'}
    if not markdown:result.update(outputs=[],execution_count=None,metadata={'jupyter':{'source_hidden':hidden}})
    cells.append(result)


cell('''
# LinQ Safety: ночное продолжение, ориентир 4–5 часов

Пять задач на двух T4. **People не исключён:** продолжение 17 завершённых эпох,
ещё до 20 минут / максимум 25 эпох суммарно. Каски продолжаются с 3 завершённых
эпох; жилеты впервые стартуют с COCO pretrained. Детекторы YOLO12m — максимум
40 эпох, классификаторы ResNet50 — до 12. Сохраняется optimizer исходных детекторов,
но расписание LR пересчитывается под меньший общий предел эпох.

Данные полные, исходные train/val/test неизменны; исключены известные ошибочные crop.
Лучшие веса выбираются на validation, test оценивается отдельно. СИЗ не обучает
падения. Зоны — геометрические правила поверх треков. Качество >0.95 не гарантируется.

Запускать **Save & Run All** с GPU T4 x2 и Internet. Inputs: приватный расширенный
Dataset и output версии **361611151**. Без last.pt с optimizer запуск остановится.
Каждую эпоху: best, last и отдельный resume-last с optimizer; предыдущая копия
сохраняется атомарно. Попытка внешнего backup в приватный Kaggle Dataset проверяется
до обучения. При отсутствии прав загрузки это явно записывается: локальный checkpoint
защищает от ошибки процесса, но не гарантирует сохранность при потере всей VM.
''',True)
cell('''
import time
NIGHT_START=time.monotonic()
print('LINQ_NIGHT_WORKER_BOOTED',time.time(),flush=True)
from datetime import datetime
GLOBAL_STOP_UTC='2026-10-24T06:00:00+00:00'  # 09:00 Moscow; shared with the parallel fall job.
remaining_seconds=datetime.fromisoformat(GLOBAL_STOP_UTC).timestamp()-time.time()
assert remaining_seconds>30*60,'The approved overnight deadline has passed; set a new explicit budget.'
DEADLINE=NIGHT_START+min(270*60,remaining_seconds)
import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','--quiet',
 'ultralytics==8.4.88','onnx==1.19.1','protobuf==5.29.5','onnxruntime==1.29.0',
 'pyyaml==6.0.3','kagglehub'],check=True)
from pathlib import Path
import json,shutil,hashlib,zipfile,torch
WORK=Path('/kaggle/working');WORK.mkdir(exist_ok=True)
assert torch.cuda.is_available() and torch.cuda.device_count()>=2,'Select GPU T4 x2 before committing.'
print('NIGHT_GPU',[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],flush=True)
OUTPUT=WORK/'night-training-output';OUTPUT.mkdir(exist_ok=True)
''')
names=['train_ppe.py','run_ppe_training.py','training_checkpoint.py','backup_training.py']
cell("SCRIPTS=WORK/'scripts';SCRIPTS.mkdir(exist_ok=True)\n"+'\n'.join(
 f"(SCRIPTS/{name!r}).write_text({(ROOT/'scripts'/name).read_text(encoding='utf-8')!r},encoding='utf-8')" for name in names),hidden=True)
cell('''
input_root=Path('/kaggle/input')
manifests=sorted(input_root.rglob('manifest.jsonl'))
assert len(manifests)==1,'Attach exactly one expanded PPE Dataset.'
FULL_DATA=manifests[0].parent
roots=[p.parent for p in input_root.rglob('training-progress.json')
       if (p.parent/'people/weights/last.pt').exists() and (p.parent/'head_ppe/weights/last.pt').exists()]
assert len(roots)==1,'Attach saved version 361611151 output; no fresh restart is allowed.'
CHECKPOINT_ROOT=roots[0]
from ultralytics import YOLO
lineage={}
for task,expected_epoch in [('people',17),('head_ppe',3)]:
    path=CHECKPOINT_ROOT/task/'weights/last.pt'
    loaded=YOLO(path);state=loaded.ckpt
    assert state['epoch']+1==expected_epoch and state.get('optimizer'),(task,'Unexpected checkpoint or missing optimizer')
    lineage[task]={'completed_epochs':state['epoch']+1,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print('RESTORED_INPUT',task,lineage[task],flush=True)
    del loaded,state
(OUTPUT/'initial-checkpoint-lineage.json').write_text(json.dumps(lineage,indent=2))
# Copy once onto the VM's fast local disk. Keep the 2.7GB corpus out of saved output.
DATA=Path('/tmp/linq-night-data')
if not (DATA/'.complete').exists():
    from concurrent.futures import ThreadPoolExecutor
    files=[p for p in FULL_DATA.rglob('*') if p.is_file()]
    def copy_one(source):
        target=DATA/source.relative_to(FULL_DATA);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index,_ in enumerate(pool.map(copy_one,files)):
            if index%10000==0:print('LOCAL_DATA_COPY',index,'/',len(files),flush=True)
    (DATA/'.complete').write_text(hashlib.sha256((FULL_DATA/'manifest.jsonl').read_bytes()).hexdigest())
assert (DATA/'.complete').read_text()==hashlib.sha256((FULL_DATA/'manifest.jsonl').read_bytes()).hexdigest()
shutil.copyfile(DATA/'manifest.jsonl',OUTPUT/'data-manifest.jsonl')
print('DATA_READY full corpus, original splits; setup minutes',(time.monotonic()-NIGHT_START)/60,flush=True)
''')
cell('''
BACKUP_HANDLE='yathebest/linq-night-checkpoints-20261024'
try:
    probe=subprocess.run([sys.executable,str(SCRIPTS/'backup_training.py'),'--root',str(OUTPUT),
     '--handle',BACKUP_HANDLE,'--once','--probe'],timeout=180)
except subprocess.TimeoutExpired:
    (OUTPUT/'remote-backup-status.json').write_text(json.dumps({'verified_upload':False,'error_type':'probe_timeout'}))
remote=json.loads((OUTPUT/'remote-backup-status.json').read_text()) if (OUTPUT/'remote-backup-status.json').exists() else {'verified_upload':False}
BACKUP_PROCESS=None
if remote.get('verified_upload'):
    BACKUP_PROCESS=subprocess.Popen([sys.executable,str(SCRIPTS/'backup_training.py'),'--root',str(OUTPUT),'--handle',BACKUP_HANDLE])
else:
    print('BACKUP_LIMITATION: remote upload not authenticated. Epoch backups and saved-version output only; VM-loss protection not verified.',flush=True)
''')
cell('''
remaining=max(1,(DEADLINE-time.monotonic())/60-15)
print('NIGHT_PLAN people=20min; head=175min; vest=105min; ResNet50 head=55min vest=40min; parallel GPU queues',flush=True)
try:
    training=subprocess.run([sys.executable,str(SCRIPTS/'run_ppe_training.py'),'--data',str(DATA),
      '--output',str(OUTPUT),'--night','--epochs','40','--classifier-epochs','12',
      '--batch','8','--classifier','resnet50','--classifier-batch','64',
      '--max-minutes',str(remaining),'--checkpoint-root',str(CHECKPOINT_ROOT)])
    print('NIGHT_TRAINING_EXIT',training.returncode,flush=True)
finally:
    if BACKUP_PROCESS:
        BACKUP_PROCESS.terminate()
        try:BACKUP_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:BACKUP_PROCESS.kill()
''')
cell('''
# Representative calibration uses TRAIN images only, never held-out test data.
import numpy as np
from PIL import Image
for task,name in [('people','person-yolo12m'),('head_ppe','ppe-head'),('vest_ppe','ppe-vest')]:
    files=sorted((DATA/task/'images/train').glob('*.jpg'))
    rng=np.random.default_rng(42)
    samples=[]
    for index in rng.choice(len(files),min(64,len(files)),replace=False):
        image=Image.open(files[int(index)]).convert('RGB');w,h=image.size;r=min(640/w,640/h)
        image=image.resize((round(w*r),round(h*r)))
        canvas=Image.new('RGB',(640,640),(114,114,114));canvas.paste(image,((640-image.width)//2,(640-image.height)//2))
        samples.append(np.asarray(canvas,dtype=np.float32).transpose(2,0,1)/255)
    np.savez_compressed(OUTPUT/(name+'-calibration.npz'),images=np.stack(samples))
progress=json.loads((OUTPUT/'training-progress.json').read_text())
complete=len(progress['jobs'])==5 and all(j['status']=='done' for j in progress['jobs'].values())
manifest={'recipe':'night-5-tasks','complete':complete,'elapsed_minutes':(time.monotonic()-NIGHT_START)/60,
 'data_manifest_sha256':hashlib.sha256((DATA/'manifest.jsonl').read_bytes()).hexdigest(),
 'test_scope':'complete original held-out test splits','classifier':'resnet50','detector':'yolo12m',
 'remote_backup':json.loads((OUTPUT/'remote-backup-status.json').read_text()),
 'linq_hardware_verified':False,'event_metrics':'not_evaluated'}
(OUTPUT/'run-manifest.json').write_text(json.dumps(manifest,indent=2))
if remote.get('verified_upload'):
    try:subprocess.run([sys.executable,str(SCRIPTS/'backup_training.py'),'--root',str(OUTPUT),'--handle',BACKUP_HANDLE,'--once'],timeout=300)
    except subprocess.TimeoutExpired:print('Final remote backup timed out; local artifacts are retained.',flush=True)
archive=WORK/'linq-night-trained-models.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in OUTPUT.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(OUTPUT))
print('DOWNLOAD',archive,'ALL_FIVE_DONE',complete,flush=True)
assert complete,'Partial run: checkpoints/logs are retained; inspect failed jobs before deployment.'
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
 'language_info':{'name':'python','version':'3.13'},'kaggle':{'accelerator':'gpu','isInternetEnabled':True}},'nbformat':4,'nbformat_minor':5}
target=ROOT/'notebooks/LinQ-PPE-Kaggle-Night.ipynb'
target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(target)
