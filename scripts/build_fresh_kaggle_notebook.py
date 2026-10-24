"""Fresh PPE run with a shared overnight deadline; fall runs concurrently on CPU."""
import ast
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/build_night_kaggle_notebook.py'))
n=json.loads((ROOT/'notebooks/LinQ-PPE-Kaggle-Night.ipynb').read_text(encoding='utf-8'))
for cell in n['cells']:
    s=cell['source']
    if cell['cell_type']=='markdown':
        s='''# LinQ Safety: новое обучение, бюджет 6–7 часов

Новый эксперимент без прежних task checkpoints. YOLO12m — официальные COCO-веса;
два ResNet50 — ImageNet. Люди: до 25 эпох / 45 минут. Каски и жилеты: до 60 эпох,
временные бюджеты 320 и 170 минут. Классификаторы: до 16 эпох / 70 и 50 минут.
Полные исходные train/val/test. Падения обучаются одновременно отдельной CPU-задачей
на расширенных 2 083 видео, поэтому не отнимают GPU и не удлиняют общий срок.

Общий предел этой ночи: **24 октября, 11:15 МСК**. Save & Run All, GPU T4 x2,
Internet. Каждый checkpoint проверяется загрузкой и наличием optimizer; первая
копия каждой новой задачи и последующие периодические снимки загружаются в
приватный Dataset. Публикуются фактические метрики; качество >0.95 не обещается.
'''
    elif s.startswith('import time'):
        s=s.replace("2026-10-24T06:00:00+00:00","2026-10-24T08:15:00+00:00").replace('09:00 Moscow','11:15 Moscow').replace('270*60','400*60')
    elif s.startswith('input_root='):
        start=s.index('# Copy once')
        s="""input_root=Path('/kaggle/input')
manifests=sorted(input_root.rglob('manifest.jsonl'))
assert len(manifests)==1,'Attach exactly one expanded PPE Dataset.'
FULL_DATA=manifests[0].parent
print('FRESH_TRAINING official pretrained initialization; no previous task checkpoint',flush=True)
"""+s[start:]
    elif s.startswith('BACKUP_HANDLE='):
        s=s.replace('linq-night-checkpoints-20261024','linq-fresh-checkpoints-20261024')
    elif s.startswith('remaining='):
        s=s.replace('people=20min; head=175min; vest=105min; ResNet50 head=55min vest=40min',
                    'people=45min; head=320min; vest=170min; ResNet50 head=70min vest=50min')
        s=s.replace("'--night','--epochs','40','--classifier-epochs','12'","'--night','--fresh','--epochs','60','--classifier-epochs','16'")
        s=s.replace(",'--checkpoint-root',str(CHECKPOINT_ROOT)","")
    else:s=s.replace("'recipe':'night-5-tasks'","'recipe':'fresh-5-tasks'")
    cell['source']=s
    if cell['cell_type']=='code':ast.parse(s)
    cell['id']=cell['id'].replace('night-','fresh-')
target=ROOT/'notebooks/LinQ-PPE-Kaggle-Fresh.ipynb'
target.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(target)
