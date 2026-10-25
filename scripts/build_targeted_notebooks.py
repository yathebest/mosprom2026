"""Generate portable notebooks: 4-5h EACH, separate or parallel on two GPUs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=['train_targeted_person.py','train_signal_clothing.py','training_checkpoint.py','training_observability.py']


def code(source):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(keepends=True)}


def build(task,output):
    title={'person':'Люди: дальний план, перекрытия и ложные станки','clothing':'Сигнальная одежда: жилет или отражающая форма','both':'Две модели параллельно: две независимые GPU'}[task]
    markdown=f'''# LinQ · {title}

**Бюджет — до 4–5 часов на КАЖДОЕ направление.** Два задания не делят один тренировочный бюджет.
Нужен соответствующий `linq-person-training-input.zip` / `linq-clothing-training-input.zip`, добавленный как Kaggle Dataset → Add Input.
Для `both` добавьте оба архива и включите **GPU T4 ×2**. При одной GPU используйте один из отдельных ноутбуков.

250 минут — потолок оптимизации каждой модели. Валидация и экспорт используют резерв; внешний предел 285 минут.
Ранняя остановка допустима: расходовать все часы после устойчивого плато не требуется.
Чекпойнты и логи сохраняются в `/kaggle/working/targeted-results`; результаты нужно сохранить при завершении.
Копии внутри одной VM не защищают от её уничтожения. Запись в отдельный Dataset/API отключена по выбору владельца.

Обучение использует только train/val. Test не копируется в рабочий тренировочный набор и не участвует в подборе весов/порогов.
Разметка новых фабричных примеров выполнена ассистентом и не заменяет независимую экспертную оценку.
Ни один выходной файл не устанавливается в приложение автоматически. Совместимость ONNX-графа проверяется; компиляция LinQ требует отдельной проверки.
'''
    setup=f'''import os,sys,time,json,zipfile,subprocess,shutil,hashlib
from pathlib import Path
NOTEBOOK_STARTED=time.monotonic()
TASK={task!r}
TRAIN_MINUTES=250
# Optional: optimizer checkpoints from your own previous Outputs. Set remaining
# TRAIN_MINUTES explicitly on resume, and attach sibling best.pt as well.
RESUME_CHECKPOINTS={{'person':None,'clothing':None}}
os.environ['PYTHONUNBUFFERED']='1'
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
print('SETUP_STARTED',TASK,flush=True)
subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.88','onnx==1.19.1','onnxruntime>=1.20,<2','PyYAML>=6.0'],check=True)
import torch
assert torch.cuda.is_available(),'Kaggle Settings: enable GPU before Save & Run All'
print('GPU_READY',torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],flush=True)
if TASK=='both':assert torch.cuda.device_count()>=2,'For both tasks use T4 x2; otherwise run separate notebooks, each with its own 4-5h budget.'
PROGRAM=Path('/kaggle/working/targeted-program');PROGRAM.mkdir(exist_ok=True)
RESULTS=Path('/kaggle/working/targeted-results');RESULTS.mkdir(exist_ok=True)
'''
    program='PROGRAM_FILES='+repr({n:(ROOT/'scripts'/n).read_text(encoding='utf-8') for n in FILES})+'\nfor name,body in PROGRAM_FILES.items():\n    (PROGRAM/name).write_text(body,encoding="utf-8")\n'
    prepare='''# Staging from FUSE is explicit, bounded and logged; no repeated remote per-frame reads.
SELECTED=['person','clothing'] if TASK=='both' else [TASK]
markers=list(Path('/kaggle/input').rglob('LINQ_TARGETED_INPUT.json'))
found={}
for marker in markers:
    meta=json.loads(marker.read_text())
    if meta['task'] in SELECTED:
        assert meta['task'] not in found,'Attach exactly one input version per task'
        found[meta['task']]=(marker.parent,meta)
for name in SELECTED:
    if name in found:continue
    archives=list(Path('/kaggle/input').rglob(f'linq-{name}-training-input.zip'))
    assert len(archives)==1,f'Attach exactly linq-{name}-training-input.zip as input'
    dest=Path('/kaggle/working/targeted-unpacked')/name;dest.mkdir(parents=True,exist_ok=True)
    print('UNPACK_START',name,archives[0].stat().st_size,flush=True)
    with zipfile.ZipFile(archives[0]) as z:
        for member in z.infolist():
            assert (dest/member.filename).resolve().is_relative_to(dest.resolve()),'Unsafe ZIP path'
        z.extractall(dest)
    meta=json.loads((dest/'LINQ_TARGETED_INPUT.json').read_text());assert meta['task']==name
    found[name]=(dest,meta)
    print('UNPACK_DONE',name,flush=True)
from concurrent.futures import ThreadPoolExecutor
STAGED={}
for name in SELECTED:
    source,meta=found[name];data=source/meta['data_directory']
    local=Path('/kaggle/temp/linq-targeted')/name;local.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in (data/'manifest.jsonl').read_text().splitlines() if line]
    selected=[r for r in rows if r['split'] in ('train','val')]
    jobs=[]
    for row in selected:
        src=data/row['file'];dst=local/row['file']
        assert dst.resolve().is_relative_to(local.resolve())
        jobs.append((src,dst,row.get('image_sha256')))
        if name=='person':
            rel=Path(row['file']);label=Path('labels')/rel.parent.name/rel.with_suffix('.txt').name
            jobs.append((data/label,local/label,None))
    def copy_one(job):
        src,dst,sha=job;dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as incoming,dst.open('wb') as outgoing:
            hasher=hashlib.sha256()
            while chunk:=incoming.read(1024*1024):outgoing.write(chunk);hasher.update(chunk)
        if sha:assert hasher.hexdigest()==sha,'Input image checksum mismatch'
    print('STAGING_START',name,len(jobs),flush=True)
    # map's result stream is consumed; errors cannot silently disappear.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(copy_one,jobs),1):
            if i%200==0:print('STAGING_PROGRESS',name,i,len(jobs),flush=True)
    (local/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in selected))
    weights=local/'initial.pt';shutil.copyfile(source/meta['weights_file'],weights)
    with weights.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==meta['initial_weights_sha256'],'Initial weights checksum mismatch'
    STAGED[name]={'data':local,'weights':weights}
    print('STAGING_VERIFIED',name,len(selected),'test_not_staged',sum(r['split']=='test' for r in rows),flush=True)
'''
    run='''from collections import deque
processes={};streams={};started={};statuses={}
for index,name in enumerate(SELECTED):
    output=RESULTS/name;output.mkdir(exist_ok=True)
    (output/'STOP').unlink(missing_ok=True)
    script='train_targeted_person.py' if name=='person' else 'train_signal_clothing.py'
    # The child's physical GPU is isolated; --device 0 is then its visible GPU.
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(index if TASK=='both' else 0)}
    command=[sys.executable,'-u',str(PROGRAM/script),'--data',str(STAGED[name]['data']),'--weights',str(STAGED[name]['weights']),
        '--output',str(output),'--minutes',str(TRAIN_MINUTES),'--device','0']
    if RESUME_CHECKPOINTS.get(name):command+=['--resume',str(RESUME_CHECKPOINTS[name])]
    streams[name]=(RESULTS/(name+'.log')).open('w')
    processes[name]=subprocess.Popen(command,stdout=streams[name],stderr=subprocess.STDOUT,env=env,cwd=PROGRAM)
    started[name]=time.monotonic();print('JOB_STARTED',name,processes[name].pid,'own_gpu',env['CUDA_VISIBLE_DEVICES'],flush=True)
def tail(path,size=1600):
    with path.open('rb') as f:f.seek(max(0,path.stat().st_size-size));return f.read().decode(errors='replace')
try:
    while len(statuses)<len(SELECTED):
        for name,process in processes.items():
            if name in statuses:continue
            age=(time.monotonic()-NOTEBOOK_STARTED)/60
            output=RESULTS/name;progress=output/'progress.json'
            if progress.exists():
                try:print('HEARTBEAT',name,progress.read_text(),flush=True)
                except (OSError,ValueError):print('PROGRESS_UPDATE_IN_FLIGHT',name,flush=True)
            else:print('PROCESS_ALIVE_WAITING_FOR_STAGE',name,round(time.monotonic()-started[name]),flush=True)
            print('BOUNDED_LOG_TAIL',name,tail(RESULTS/(name+'.log')),flush=True)
            if age>=270 and not (output/'STOP').exists():
                (output/'STOP').touch();print('GRACEFUL_STOP_REQUEST',name,'reserve validation/export',flush=True)
            if age>=285 and process.poll() is None:
                process.terminate();print('WALL_LIMIT_TERMINATE',name,'preserving existing checkpoints',flush=True)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            if process.poll() is not None:
                statuses[name]={'exit_code':process.returncode};streams[name].flush()
                print('JOB_FINISHED',name,process.returncode,flush=True)
        if len(statuses)<len(SELECTED):time.sleep(20)
finally:
    for name,process in processes.items():
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        streams[name].close()
    (RESULTS/'RUN_STATUS.json').write_text(json.dumps({'jobs':statuses,'models_installed':False,'remote_backup':False},indent=2))
    archive=Path('/kaggle/working/linq-targeted-trained-candidates.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for file in RESULTS.rglob('*'):
            if file.is_file() and not file.name.endswith('.writing'):z.write(file,file.relative_to(RESULTS))
    print('RESULT_ARCHIVE',str(archive),archive.stat().st_size,flush=True)
assert len(statuses)==len(SELECTED) and all(s['exit_code']==0 for s in statuses.values()),'At least one job failed; inspect its bounded log tail and preserve resume checkpoints from Outputs.'
'''
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'kaggle':{'accelerator':'gpu','isInternetEnabled':True}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':markdown.splitlines(keepends=True)},code(setup),code(program),code(prepare),code(run)]}
    for i,c in enumerate(notebook['cells']):
        c['id']=f'{task}-{i}'
        if c['cell_type']=='code':compile(''.join(c['source']),f'<{task}-cell-{i}>','exec')
    output.mkdir(parents=True,exist_ok=True)
    target=output/({'person':'LinQ-People-4h30.ipynb','clothing':'LinQ-Signal-Clothing-4h30.ipynb','both':'LinQ-Two-GPU-Parallel-4h30.ipynb'}[task])
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    try:
        import nbformat;nbformat.validate(nbformat.read(target,as_version=4))
    except ImportError:pass
    print(target,flush=True)


if __name__=='__main__':
    for task in ['person','clothing','both']:build(task,ROOT/'output/targeted-training')
