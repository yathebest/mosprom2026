"""Build a self-contained notebook: no mutable GitHub checkout is required."""
import ast
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
cells=[]


def markdown(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip()})


def code(text,hidden=False):
    text=textwrap.dedent(text).strip();ast.parse(text)
    cells.append({'cell_type':'code','metadata':{'jupyter':{'source_hidden':hidden}},
                  'source':text,'outputs':[],'execution_count':None})


markdown('''
# LinQ Safety: обучение людей, касок и жилетов

Самодостаточный notebook. В Kaggle включите GPU и Internet. Прикрепите
`linq-ppe-training-data.zip` как приватный Dataset: Kaggle может распаковать его
самостоятельно. Notebook найдёт `manifest.jsonl`. Без готового корпуса он скачает
открытые источники и подготовит данные сам. Ключи и пароли не нужны.

Сначала `MODE="smoke"`: один цикл проверяет тракт. Затем новая полная сессия с
`MODE="full"`. Сохранённые last-checkpoints позволяют продолжать прерванный запуск.
При переносе outputs предыдущей сессии укажите `RESUME_FROM`.

Архитектура: YOLO12m для людей и двух задач СИЗ, ResNet152 для головы и торса.
Разные задачи имеют разные датасеты: отсутствие разметки жилета не равно его отсутствию.
Данные уже очищаются по SHA-256/pHash, исходным split и Roboflow-группам.
Отдельно нужны проверка разметки человеком и независимые реальные видео с зеркалами.
Не выдавайте качество на crop за event F1. Метрики >0.95 здесь не предзаписаны.
''')
code('''
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','--quiet',
                'ultralytics==8.4.88','onnx==1.16.2','onnxruntime==1.29.0',
                'gdown==5.2.0','pyyaml==6.0.3'],check=True)
from pathlib import Path
import json, shutil, zipfile, numpy as np, torch
WORK=Path('/kaggle/working'); WORK.mkdir(exist_ok=True)
assert torch.cuda.is_available(), 'Выберите GPU в Settings -> Accelerator и перезапустите сессию.'
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/2**30, 'GiB')
''')
embedded=['download_ppe_data.py','prepare_ppe_data.py','finalize_ppe_data.py','train_ppe.py','check_linq_compatibility.py']
code("SCRIPTS=WORK/'scripts'; SCRIPTS.mkdir(exist_ok=True)\n"+'\n'.join(
    f"(SCRIPTS/{name!r}).write_text({(ROOT/'scripts'/name).read_text(encoding='utf-8')!r},encoding='utf-8')" for name in embedded),True)
code('''
MODE='smoke'  # После проверки смените на 'full'; smoke не является обученной финальной моделью.
RESUME_FROM=None  # Например Path('/kaggle/input/my-previous-checkpoints/training-output')
SOURCES=['construction','site_safety','shwd','hardhat_vest']
# 'sh17' (~14 GB) добавляйте только с учётом CC BY-NC-SA 4.0 и свободного диска.
# HardHat-Vest: только явные положительные crop; head/person не означают отрицательные СИЗ.
EPOCHS=1 if MODE=='smoke' else 150
CLASSIFIER_EPOCHS=1 if MODE=='smoke' else 30
LABELS_REVIEWED=False  # В full ставить True после проверки разметки и crop, не по результатам smoke.
IMGSZ=640  # Сначала статический 640 для воспроизведения LinQ; 960/1280 - отдельная проверка.
BATCH=4
OUTPUT=WORK/('smoke-output' if MODE=='smoke' else 'training-output')
OUTPUT.mkdir(exist_ok=True)
if RESUME_FROM: shutil.copytree(RESUME_FROM,OUTPUT,dirs_exist_ok=True)
''')
code('''
input_root=Path('/kaggle/input')
manifests=list(input_root.rglob('manifest.jsonl'))
archives=list(input_root.rglob('linq-ppe-training-data.zip'))
if manifests:
    DATA=manifests[0].parent
elif archives:
    DATA=WORK/'ppe-prepared'
    with zipfile.ZipFile(archives[0]) as z:
        for member in z.infolist():
            if not (DATA/member.filename).resolve().is_relative_to(DATA.resolve()):
                raise ValueError('Unsafe dataset archive path')
        z.extractall(DATA)
else:
    DATA=WORK/'ppe-prepared'; RAW=WORK/'ppe-sources'
    subprocess.run([sys.executable,str(SCRIPTS/'download_ppe_data.py'),
                    '--root',str(RAW),'--sources',*SOURCES],check=True)
    subprocess.run([sys.executable,str(SCRIPTS/'prepare_ppe_data.py'),
                    '--root',str(RAW),'--output',str(DATA),
                    '--summary',str(OUTPUT/'dataset-summary.json')],check=True)
for task in ['people','head_ppe','vest_ppe']:
    assert (DATA/task/'data.yaml').exists(), f'Не найден dataset {task}'
print('Dataset:', DATA)
''')
markdown('''
Перед полным запуском просмотрите случайные crop и проверьте классы. Особенно важны
`no_hardhat` и `no_vest`. Слишком маленькая или закрытая область должна попадать
в отдельный вручную размеченный набор видимости. Notebook не создаёт фиктивный
класс unknown. До появления такого набора используется отказ по порогу уверенности.
Автоматически исключаются crop меньше 24 px по стороне / 1024 px по площади,
крайние пропорции и сильное размытие. Это не проверяет правильность исходной разметки.
''')
code('''
import matplotlib.pyplot as plt
from PIL import Image
rng=np.random.default_rng(42)
paths=list((DATA/'head_ppe_classifier/train').rglob('*.jpg'))+list((DATA/'vest_ppe_classifier/train').rglob('*.jpg'))
sample=rng.choice(len(paths),min(16,len(paths)),replace=False)
fig,axes=plt.subplots(4,4,figsize=(12,12))
for ax,index in zip(axes.flat,sample):
    p=paths[int(index)];ax.imshow(Image.open(p));ax.set_title(p.parent.name);ax.axis('off')
plt.tight_layout()
''')
code('''
assert MODE=='smoke' or LABELS_REVIEWED, 'Просмотрите разметку и crop, затем задайте LABELS_REVIEWED=True.'
if MODE=='smoke':
    # A bounded pipeline test, never a quality benchmark or release model.
    FULL_DATA=DATA; DATA=WORK/'smoke-data'
    for task in ['people','head_ppe','vest_ppe']:
        target=DATA/task;target.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(FULL_DATA/task/'data.yaml',target/'data.yaml')
        for split,limit in [('train',48),('val',16),('test',16)]:
            for image in sorted((FULL_DATA/task/'images'/split).glob('*'))[:limit]:
                for kind,source in [('images',image),('labels',FULL_DATA/task/'labels'/split/(image.stem+'.txt'))]:
                    dest=target/kind/split/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(source,dest)
    for task in ['head_ppe_classifier','vest_ppe_classifier']:
        for split,limit in [('train',48),('val',16),('test',16)]:
            for classdir in sorted((FULL_DATA/task/split).iterdir()):
                for source in sorted(classdir.glob('*.jpg'))[:limit]:
                    dest=DATA/task/split/classdir.name/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(source,dest)
    print('Smoke uses small fixed subsets:',DATA)
''')
code('''
for task in ['people','head_ppe','vest_ppe']:
    subprocess.run([sys.executable,str(SCRIPTS/'train_ppe.py'),'--data',str(DATA),
                    '--task',task,'--epochs',str(EPOCHS),'--imgsz',str(IMGSZ),
                    '--batch',str(BATCH),'--device','0','--output',str(OUTPUT),'--resume'],check=True)
''')
code('''
for task in ['head_ppe','vest_ppe']:
    subprocess.run([sys.executable,str(SCRIPTS/'train_ppe.py'),'--data',str(DATA),
                    '--task',task,'--mode','classify','--classifier','resnet152',
                    '--epochs',str(CLASSIFIER_EPOCHS),'--batch','16','--device','0',
                    '--output',str(OUTPUT),'--resume'],check=True)
''')
code('''
# Train-only representative calibration tensors. No test images enter this file.
import cv2
for task in ['people','head_ppe','vest_ppe']:
    files=sorted((DATA/task/'images/train').glob('*'))
    rng=np.random.default_rng(42);chosen=rng.choice(len(files),min(300,len(files)),replace=False)
    tensor=[]
    for index in chosen:
        image=cv2.imread(str(files[int(index)]));h,w=image.shape[:2]
        scale=min(IMGSZ/w,IMGSZ/h);nw,nh=round(w*scale),round(h*scale)
        left,top=round((IMGSZ-nw)/2-.1),round((IMGSZ-nh)/2-.1)
        padded=np.full((IMGSZ,IMGSZ,3),114,np.uint8)
        padded[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh))
        tensor.append(padded[:,:,::-1].transpose(2,0,1).astype(np.float32)/255)
    np.savez_compressed(OUTPUT/f'calibration-{task}-train.npz',images=np.stack(tensor),split=np.array('train'))
    del tensor
# Split parity is local ONNX verification; SDK compilation is performed separately.
subprocess.run([sys.executable,str(SCRIPTS/'check_linq_compatibility.py'),
                '--model',str(OUTPUT/'person-yolo12m.onnx'),'--output',str(OUTPUT/'linq-person'),
                '--report',str(OUTPUT/'linq-preflight.json')],check=True)
''')
markdown('''
После полного обучения скачайте `linq-trained-models.zip`. Сначала изучите metrics JSON
и результаты на независимых видео. Детекторные mAP и crop F1 не заменяют event F1.
Для LinQ перенесите ONNX и train-калибровку в Linux 3.12 с SDK 2026.9.45, выполните
`check_linq_compatibility.py --compile --calibration ...`, затем проверьте `.tpu` на устройстве.
После обучения меняется число классов: разрез графа снова проверяется, совместимость
предобученного COCO-графа автоматически на новый checkpoint не переносится.

Для локального сайта: проверенные `person-yolo12m.pt`, `ppe-head.onnx`, `ppe-vest.onnx`,
`ppe-head-classifier.onnx`, `ppe-torso-classifier.onnx` положить в `data/models` и перезапустить.
Старые веса и отчёты сохраните для отката. Smoke-артефакты в рабочий сайт не устанавливать.
''')
code('''
manifest={'mode':MODE,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
          'linq_hardware_verified':False,'event_metrics':'not_evaluated',
          'warning':'A smoke run is a pipeline check, not a release model.'}
(OUTPUT/'run-manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.make_archive(str(WORK/'linq-trained-models'),'zip',OUTPUT)
print('Скачать:',WORK/'linq-trained-models.zip')
''')
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
          'language_info':{'name':'python'},'kaggle':{'accelerator':'gpu','isInternetEnabled':True}},'cells':cells}
for i,cell in enumerate(cells):cell['id']=f'linq-{i:03}'
path=ROOT/'notebooks/LinQ-PPE-Kaggle.ipynb';path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(path)
