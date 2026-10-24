"""Independent CPU training alongside the PPE GPU job, within one shared deadline."""
import ast
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1];cells=[]


def cell(source,markdown=False):
    source=textwrap.dedent(source).strip()
    if not markdown:ast.parse(source)
    result={'cell_type':'markdown' if markdown else 'code','metadata':{},'source':source,'id':f'fall-{len(cells):03}'}
    if not markdown:result.update(outputs=[],execution_count=None)
    cells.append(result)


cell('''
# LinQ Safety: падения, новый временной RGB-классификатор

**Обучать одновременно с PPE**, на CPU — не занимать два T4 основного задания.
Новая задача от ImageNet ResNet18 + GRU, без прежних task checkpoints.
2 083 исходных RGB-видео: 2 013 уникальных Figshare и 70 прежних URFD.
Figshare разделён по участникам (20 train / 4 val / 5 test); URFD сохраняет прежний
video split, независимость людей в URFD не доказана. Начало падения размечено лишь
в URFD: Figshare даёт метрики **классификации клипов**, не event F1 и не задержку.

До 30 эпох temporal head, затем до 4 эпох / 30 минут дообучения layer4.
Ограничение CPU-задачи 100 минут, общий предел ночи **11:15 МСК 24 октября**.
Checkpoints каждой эпохи, optimizer, RNG, предыдущая копия; приватный внешний backup.
Frame encoder экспортируется отдельно для LinQ ResNet18; GRU выполняется на CPU.
Новые веса не устанавливаются в сайт без независимой проверки.
''',True)
cell('''
import time,sys,subprocess
from datetime import datetime
print('LINQ_FALL_WORKER_BOOTED',time.time(),flush=True)
STOP_UTC='2026-10-24T08:15:00+00:00'
remaining=(datetime.fromisoformat(STOP_UTC).timestamp()-time.time())/60-10
assert remaining>20,'The approved overnight deadline has passed.'
CPU_BUDGET=min(100,remaining)
subprocess.run([sys.executable,'-m','pip','install','--quiet',
 'onnx==1.19.1','protobuf==5.29.5','onnxruntime==1.29.0','kagglehub'],check=True)
from pathlib import Path
import json,shutil,zipfile,hashlib
WORK=Path('/kaggle/working');OUTPUT=WORK/'fall-training-output';OUTPUT.mkdir(exist_ok=True)
SCRIPTS=WORK/'scripts';SCRIPTS.mkdir(exist_ok=True)
APP=WORK/'app';APP.mkdir(exist_ok=True);(APP/'__init__.py').write_text('')
''')
cell('\n'.join(f"(SCRIPTS/{name!r}).write_text({(ROOT/'scripts'/name).read_text(encoding='utf-8')!r},encoding='utf-8')" for name in
 ['train_fall_temporal.py','training_checkpoint.py','backup_training.py'])+'\n'+
 f"(APP/'fall_temporal.py').write_text({(ROOT/'app/fall_temporal.py').read_text(encoding='utf-8')!r},encoding='utf-8')")
cell('''
manifests=list(Path('/kaggle/input').rglob('fall-manifest.jsonl'))
assert len(manifests)==1,'Attach LinQ Fall Expanded 2026 Dataset only.'
SOURCE=manifests[0].parent;DATA=Path('/tmp/linq-fall-data')
from concurrent.futures import ThreadPoolExecutor
files=[p for p in SOURCE.rglob('*') if p.is_file()]
def copy_one(source):
    target=DATA/source.relative_to(SOURCE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(copy_one,files))
for name in ['fall-summary.json','fall-manifest.jsonl','SOURCE-NOTICES.txt']:
    shutil.copyfile(DATA/name,OUTPUT/name)
print('FALL_DATA_READY',json.loads((DATA/'fall-summary.json').read_text())['splits'],flush=True)
''')
cell('''
HANDLE='yathebest/linq-fall-checkpoints-20261024'
CPU_BUDGET=min(100,(datetime.fromisoformat(STOP_UTC).timestamp()-time.time())/60-10)
assert CPU_BUDGET>10,'Insufficient time remaining before the overnight deadline.'
backup=subprocess.Popen([sys.executable,str(SCRIPTS/'backup_training.py'),'--root',str(OUTPUT),'--handle',HANDLE])
code=124
try:
    training=subprocess.run([sys.executable,str(SCRIPTS/'train_fall_temporal.py'),'--fresh',
      '--data',str(DATA),'--output',str(OUTPUT),'--device','cpu','--batch','4','--cpu-threads','4',
      '--workers','2','--head-epochs','30','--finetune-epochs','4','--finetune-minutes','30',
      '--max-minutes',str(CPU_BUDGET)],timeout=(CPU_BUDGET+10)*60)
    code=training.returncode
except subprocess.TimeoutExpired:
    print('FALL_TRAINING_TIMEOUT; preserving available checkpoints.',flush=True)
finally:
    backup.terminate()
    try:backup.wait(timeout=10)
    except subprocess.TimeoutExpired:backup.kill()
print('FALL_TRAINING_EXIT',code,flush=True)
''')
cell('''
try:
    subprocess.run([sys.executable,str(SCRIPTS/'backup_training.py'),'--root',str(OUTPUT),'--handle',HANDLE,'--once'],timeout=300)
except subprocess.TimeoutExpired:print('Final external backup timed out; local artifacts retained.',flush=True)
archive=WORK/'linq-fall-trained-models.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for path in OUTPUT.rglob('*'):
        if path.is_file():z.write(path,path.relative_to(OUTPUT))
print('DOWNLOAD',archive,flush=True)
assert code==0,'Training incomplete; inspect preserved checkpoints and logs.'
''')
n={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
 'kaggle':{'accelerator':'none','isInternetEnabled':True}},'nbformat':4,'nbformat_minor':5}
target=ROOT/'notebooks/LinQ-Fall-Kaggle.ipynb';target.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(target)
