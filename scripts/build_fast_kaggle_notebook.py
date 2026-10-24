"""Build a separate bounded-budget Kaggle experiment; retain the full recipe."""
import ast
import json
from pathlib import Path
import runpy
import textwrap

ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/build_kaggle_notebook.py'))
notebook=json.loads((ROOT/'notebooks/LinQ-PPE-Kaggle.ipynb').read_text(encoding='utf-8'))
for cell in notebook['cells']:
    source=cell['source']
    if source.startswith('# LinQ Safety:'):
        source='''# LinQ Safety: быстрый эксперимент на двух T4

Все пять задач: три YOLO12m и два ResNet50. Это короткое
дообучение с ограниченным бюджетом, а не эквивалент 150 эпох.
Люди и каски продолжаются из last.pt сохранённой версии 361611151; без них запуск
останавливается. Сохраняются optimizer и номер эпохи, ограничивается дополнительное
время. Жилеты и оба классификатора в прежнем запуске ещё не обучались: они начинают
с pretrained с замороженными ранними слоями.
Ожидаемое время 60-90 минут, общий бюджет 110 минут плюс ожидание ресурсов Kaggle.
Приватный расширенный корпус остаётся целым. Используется фиксированная выборка,
сохранены исходные train/val/test и группы. Метрики относятся к выбранной части test.
Расширенную проверку на всём held-out и целевых видео нужно выполнить отдельно.
Оценка >0.95 не гарантирована. Результаты не устанавливаются в сайт автоматически.
Добавьте приватный Dataset LinQ PPE Expanded 2026, GPU T4 x2 и Internet.
'''
    elif source.startswith('SCRIPTS='):
        source+='\n'+f"(SCRIPTS/'prepare_fast_ppe.py').write_text({(ROOT/'scripts/prepare_fast_ppe.py').read_text(encoding='utf-8')!r},encoding='utf-8')"
    elif source.startswith("MODE='smoke'"):
        source='''import time
MODE='fast'
DEADLINE=time.monotonic()+110*60
IMGSZ=640
BATCH=16
EPOCHS=40  # Maximum total, including restored epochs; time callback may stop sooner.
CLASSIFIER_EPOCHS=4
LABELS_REVIEWED=True  # 96 sampled train crops reviewed, not the complete corpus.
OUTPUT=WORK/'fast-training-output';OUTPUT.mkdir(exist_ok=True)
'''
    elif source.startswith('input_root='):
        source='''input_root=Path('/kaggle/input')
manifests=sorted(input_root.rglob('manifest.jsonl'))
assert manifests, 'Attach the prepared expanded Dataset; no slow download fallback in fast mode.'
FULL_DATA=manifests[0].parent
checkpoint_roots=[p.parent for p in input_root.rglob('training-progress.json')
                  if (p.parent/'people/weights/last.pt').is_file() and (p.parent/'head_ppe/weights/last.pt').is_file()]
assert len(checkpoint_roots)==1, 'Attach output of saved version 361611151 with both last.pt checkpoints. No fresh restart.'
CHECKPOINT_ROOT=checkpoint_roots[0]
print('Saved checkpoint folder:',CHECKPOINT_ROOT,flush=True)
DATA=WORK/'fast-data'
subprocess.run([sys.executable,str(SCRIPTS/'prepare_fast_ppe.py'),'--source',str(FULL_DATA),
                '--output',str(DATA),'--workers','8'],check=True,timeout=25*60)
shutil.copyfile(DATA/'fast-data-summary.json',OUTPUT/'fast-data-summary.json')
shutil.copyfile(DATA/'selection.jsonl',OUTPUT/'selection.jsonl')
print('Fast local dataset:',DATA,flush=True)
'''
    elif source.startswith("assert MODE=='smoke'"):
        source="print('Fast continuation: saved optimizer/epochs for people and head, 15 added minutes per task',flush=True)"
    elif source.startswith('subprocess.run([sys.executable,str(SCRIPTS/\'run_ppe_training.py\')'):
        source='''remaining=max(1,min(75,(DEADLINE-time.monotonic())/60-10))
training=subprocess.run([sys.executable,str(SCRIPTS/'run_ppe_training.py'),'--data',str(DATA),
                '--epochs',str(EPOCHS),'--classifier-epochs',str(CLASSIFIER_EPOCHS),
                '--imgsz',str(IMGSZ),'--batch',str(BATCH),'--classifier','resnet50',
                '--classifier-batch','32','--fast','--job-minutes','15','--max-minutes',str(remaining),
                '--checkpoint-root',str(CHECKPOINT_ROOT),
                '--output',str(OUTPUT)])
print('Training exit code:',training.returncode,'Check training-progress.json for all five tasks.',flush=True)
'''
    elif source.startswith('# Train-only representative'):
        source=source.replace('min(300,len(files))','min(64,len(files))')
        source=source.replace("subprocess.run([sys.executable,str(SCRIPTS/'check_linq_compatibility.py'),", "if (OUTPUT/'person-yolo12m.onnx').exists():\n    subprocess.run([sys.executable,str(SCRIPTS/'check_linq_compatibility.py'),")
    elif source.startswith('После полного обучения'):
        source='''После быстрого продолжения скачайте `linq-fast-trained-models.zip`. В `checkpoint-lineage.json`
сохраняются SHA256 исходных last.pt и число уже завершённых эпох. Люди и каски
сохраняют optimizer и номер эпохи; данные и лимит времени изменены. `resume-last.pt`
сохраняет optimizer для следующего продолжения даже после финального ONNX-экспорта.
Проверьте метрики, полный held-out и реальные видео перед установкой в сайт.
Для LinQ нужны отдельная SDK-компиляция и аппаратная проверка.'''
    elif source.startswith('import hashlib'):
        source='''import hashlib
progress=json.loads((OUTPUT/'training-progress.json').read_text())
complete=len(progress['jobs'])==5 and all(j['status']=='done' for j in progress['jobs'].values())
manifest={'mode':'fast','complete':complete,'torch':torch.__version__,
          'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
          'source_manifest_sha256':hashlib.sha256((FULL_DATA/'manifest.jsonl').read_bytes()).hexdigest(),
          'selection_sha256':hashlib.sha256((DATA/'selection.jsonl').read_bytes()).hexdigest(),
          'trainer_sha256':hashlib.sha256((SCRIPTS/'train_ppe.py').read_bytes()).hexdigest(),
          'test_scope':'fixed bounded subset; full held-out not evaluated',
          'linq_hardware_verified':False,'event_metrics':'not_evaluated',
          'warning':'Short fine-tuning; independent quality must be measured before deployment.'}
(OUTPUT/'run-manifest.json').write_text(json.dumps(manifest,indent=2))
import zipfile
archive=WORK/'linq-fast-trained-models.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for file in OUTPUT.rglob('*'):
        if file.is_file():z.write(file,file.relative_to(OUTPUT))
print('Download:',archive,'All five tasks complete:',complete,flush=True)
assert complete, 'Partial run: inspect saved progress/checkpoints; do not treat missing models as completed.'
'''
    cell['source']=textwrap.dedent(source).strip()
    if cell['cell_type']=='code':ast.parse(cell['source'])
for index,cell in enumerate(notebook['cells']):cell['id']=f'fast-{index:03}'
notebook['metadata']['kaggle']['accelerator']='gpu'
path=ROOT/'notebooks/LinQ-PPE-Kaggle-Fast.ipynb'
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(path)
