"""Create one portable application+weights ZIP. No DB, secrets or camera media."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'linq-server'

README = '''LINQ SAFETY — ПАКЕТ ДЛЯ СЕРВЕРА

Внутри приложение, все текущие PT/ONNX-веса из data/models, их конфиги,
готовый имеющийся комплект LinQ в data/linq, открытые примеры и SHA256-манифест.
Рабочие папки data/models и data/linq уже расположены на правильных местах.

1. Распакуйте архив в НОВУЮ папку на Linux x86_64 / Python 3.12.
   python3 -m zipfile -e linq-server-package-20261025.zip ./release
   cd release/linq-server
   python3 scripts/verify_server_bundle.py

2. Создайте ОТДЕЛЬНОЕ окружение приложения, сохранив окружение Framework:
   python3.12 -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements-server.txt
   python scripts/server_preflight.py

   Установка зависимостей требует доступа к пакетным репозиториям. Если пакеты
   уже установлены в подходящем отдельном окружении, начните с preflight.
   Для минимального Linux нужны системные libgl1 и libglib2.0-0 для OpenCV;
   при их отсутствии установку выполняет администратор сервера.

3. Создайте администратора интерактивно, задав свои логин и пароль:
   python scripts/create_admin.py
   Скрипт не заменяет существующего администратора.

4. Функциональный запуск всех моделей на CPU:
   export CUDA_VISIBLE_DEVICES=-1
   export SAFETY_ONNX_DEVICE=cpu
   python -m uvicorn app.server:app --host 127.0.0.1 --port 8765
   Откройте http://127.0.0.1:8765/ В БРАУЗЕРЕ НА СЕРВЕРЕ либо через SSH-туннель.
   Для туннеля на своём компьютере: ssh -L 8765:127.0.0.1:8765 USER@SERVER
   Затем на своём компьютере откройте http://127.0.0.1:8765/.
   Доступ к SSH и внешнему порту зависит от настроек организаторов.
   Jupyter сам по себе не делает localhost сервера localhost вашего компьютера.

5. Проверьте «Анализ записей» → «Заводская камера · без каски» и падение.
   Затем подключайте свои источники. Для конкретной камеры отметьте её зоны.
   Пользователи и новые клипы будут храниться в data, создаваемой при запуске.

NVIDIA CUDA: установите согласованные Torch/torchvision и ONNX Runtime GPU
для драйвера сервера в отдельном окружении; удалите два CPU-ограничения из п.4.
Обычный pip onnxruntime из requirements — CPU-runtime; preflight показывает
фактически доступные провайдеры. Сам пакет не меняет драйверы сервера.

LINQ TPU:
Полный комплект всех моделей находится в форматах PT/ONNX. Эти файлы не
становятся TPU-программами от простого копирования. В data/linq/yolo12m лежит
уже скомпилированный COCO YOLO12m head .tpu и его согласованные tail/scales/
thresholds. Это только один имеющийся аппаратный комплект, не TPU-версии всех
новых дообученных моделей. SDK/драйвер/pytpu/удалённый клиент предоставляются
организаторами. Порядок отдельного переноса — docs/linq-port.md и совместимый
исходный ONNX — data/models/yolo12m.onnx. Не смешивайте этот .tpu с другим tail.
Пакет обеспечивает перенос приложения и весов; он не подтверждает запуск
всех сценариев на LinQ TPU. В режиме п.4 приложение использует CPU сервера.

SERVER_MANIFEST.json фиксирует все включённые файлы, размеры и SHA256.
MODEL_INVENTORY.json разделяет рабочие модели, альтернативные представления
и сохранённые модели отключённых дополнительных сценариев.

Текущая доступность сценариев задаётся app/pattern-policy.json. Веса отключённых
сценариев сохранены для воспроизводимости, но их наличие не включает сценарий.
Локальные пользователи, действующая БД, приглашения, пароли SMTP/VPN и записи
камер в пакет не входят. Новая установка создаёт пустую локальную БД.

Обслуживание УЖЕ СУЩЕСТВУЮЩЕЙ серверной БД (не требуется для новой установки):
   python scripts/backfill_clip_metadata.py --data-dir data
   python scripts/coalesce_legacy_clips.py --data-dir data
Это предварительный просмотр. Добавляйте --apply только после проверки отчёта;
оба скрипта сохраняют резервную копию БД. Скрипт группировки не удаляет медиа
и не объединяет события разных источников. Решения по событиям сохраняются.
Группировка новых клипов уже выполняется приложением автоматически.
'''


def sources(include_linq_runtime=False):
    # Include new application modules/assets without staging the shared tree.
    # --exclude-standard omits ignored caches/local files from untracked input.
    raw = subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard','--','app'], cwd=ROOT)
    paths = {ROOT/name for name in raw.decode('utf-8').split('\0') if name}
    for name in ['requirements.txt','requirements-server.txt','requirements-linq.txt','my_linq.py','start-linq.cmd',
                 'scripts/repro_linq_adapter.py','scripts/verify_linq_decode.py','scripts/install_linq_runtime.py',
                 'notebooks/stand-linq-check.ipynb',
                 'docs/linq-site-integration.ru.md','scripts/create_admin.py','scripts/verify_server_bundle.py',
                 'docs/linq-integration-verification-20261025.json',
                 'scripts/server_preflight.py','scripts/check_linq_compatibility.py',
                 'scripts/backfill_clip_metadata.py','scripts/coalesce_legacy_clips.py',
                 'scripts/build_server_bundle.py']:
        paths.add(ROOT/name)
    for name in ['quality_report.json','quality_code_audit.json','coverage.json','linq-port.md',
                 'repository-criteria-audit-20261025.json',
                 'framework-audit.json','yolo12m-compatibility.json','linq-sdk-2026.9.45-freeze.txt',
                 'scenarios.ru.md','trained-deployment.json','trained-model-metrics.json',
                 'pattern-audit-summary-20261025.json','pattern-unsupported-audit-20261025.json',
                 'pattern-availability-ui.ru.md']:
        if (ROOT/'docs'/name).exists(): paths.add(ROOT/'docs'/name)
    for folder, extensions in [('data/models',{'.pt','.onnx','.json'}),
                               ('data/linq/yolo12m',{'.tpu','.onnx','.json','.npz'})]:
        paths.update(p for p in (ROOT/folder).glob('*') if p.is_file() and p.suffix in extensions)
    for name in ['fall-01-cam0-rgb.mp4','fall-02-cam0-rgb.mp4','adl-01-cam0-rgb.mp4','adl-02-cam0-rgb.mp4','manifest.json']:
        p=ROOT/'data/datasets/urfd'/name
        if p.exists(): paths.add(p)
    for name in ['no_hardhat-demo.mp4','hardhat-demo.mp4','fire-demo-frame-demo.mp4','manifest.json','selected-manifest.json']:
        p=ROOT/'data/datasets/scenario-fixtures'/name
        if p.exists(): paths.add(p)
    for name in ['data/datasets/factory-expansion/test/6_te9.mp4','data/datasets/factory-cctv/ПРОЧИТАЙ МЕНЯ.txt']:
        if (ROOT/name).exists(): paths.add(ROOT/name)
    if include_linq_runtime:
        names=['tpu_pycloud_format/'+name for name in
               ['__init__.py','_io.py','headers.py','message.proto','message_pb2.py']]
        names.append('data/linq/person-yolo12m/person-yolo12m-head-b1-o1-h1m.tpu')
        for name in names:
            if not (ROOT/name).is_file():raise FileNotFoundError('LinQ kit component missing: '+name)
            paths.add(ROOT/name)
    result=[]
    for path in sorted(paths):
        relative=path.relative_to(ROOT).as_posix()
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc': continue
        if relative.startswith(('data/runs/','data/cache/')) or '.env' in path.name or path.suffix in {'.sqlite3','.ovpn','.key','.pem','.p12','.pfx'} or path.name.lower() in {'access.json','kaggle.json','credentials.json','mail.local.json'}:
            raise ValueError('Private/local file in release plan: '+relative)
        if path.suffix in {'.onnx','.pt','.tpu'}:
            with path.open('rb') as stream:
                if stream.read(100).startswith(b'version https://git-lfs.github.com/spec'):
                    raise ValueError('Unresolved LFS pointer: '+relative)
        result.append((relative,path))
    return result


def inventory(files):
    roles={
        'yolov8n-pose':'Pose/keypoints; PT runtime and ONNX alternative',
        'person-yolo12m':'Adapted person detector', 'yolo12m':'COCO person ensemble / LinQ source',
        'ppe-safetyvision':'General PPE proposals', 'ppe-head':'Head PPE detector / classifier',
        'ppe-vest':'Torso PPE detector', 'ppe-torso':'Torso PPE classifier',
        'phone-objects':'Phone detector', 'fire-dfire':'Additional flame detector',
        'fire-smoke':'Fire/smoke detector', 'smoking':'Smoking/cigarette detector',
        'fall-resnet18':'RGB body-state model', 'fall-temporal':'Temporal fall model',
        'construction-machinery':'Retained equipment model; scenario currently disabled',
        'coco-objects':'Retained object model; load-posture scenario currently disabled'}
    return [{'file':name,'role':next((role for prefix,role in roles.items() if Path(name).stem.startswith(prefix)),'Model metadata or alternative'),
             'bytes':path.stat().st_size} for name,path in files if name.startswith('data/models/')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--with-linq',action='store_true',help='Include private organizer runtime and trained TPU backup for internal transfer')
    args=parser.parse_args();files=sources(args.with_linq);models=inventory(files)
    summary={'files':len(files),'model_files':sum(Path(name).suffix in {'.pt','.onnx'} for name,_ in files if name.startswith('data/models/')),
             'uncompressed_bytes':sum(p.stat().st_size for _,p in files)}
    if args.plan: print(json.dumps(summary));return
    target=args.output.resolve()
    if target.exists():raise FileExistsError('Refusing to overwrite '+str(target))
    target.parent.mkdir(parents=True,exist_ok=True)
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),
              'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
              'application_source':'tracked and nonignored untracked app files from the current worktree; exact bytes fixed by file SHA256',
              'model_files':summary['model_files'],'linq_runtime_included':args.with_linq,'files':[]}
    linq_note='''
ДОПОЛНЕНИЕ: УДАЛЁННЫЙ ГИБРИДНЫЙ LINQ
Добавлен my_linq.py и requirements-linq.txt. Для удалённого режима нужны
доступ к брокеру организаторов и приватный tpu_pycloud_format рядом с адаптером.
Установка: python -m pip install -r requirements-linq.txt
Windows: start-linq.cmd. Linux:
  export LINQ_ADAPTER=my_linq:create_backend
  export LINQ_LOGIN=hackathon74291
  export LINQ_PROGRAM=person-yolo12m-head-b1-o1-h1m.tpu
  python -m uvicorn app.server:app --host 127.0.0.1 --port 8765
Сервис проверяется в startup. Без доступа к TPU режим завершается ошибкой;
скрытой подмены person-детекции на CPU нет. Поза и сценарные модели остаются
на хосте. Версия URI v2, адаптированная person-модель не является COCO-комплектом
из п. LINQ TPU выше. Подробности и фактические SHA: docs/linq-site-integration.ru.md.
'''
    linq_note+=('Приватный сериализатор и резервный trained .tpu ВКЛЮЧЕНЫ в этот локальный архив.\n'
                if args.with_linq else 'Приватный сериализатор НЕ включён; перенесите его из integration kit отдельно.\n')
    generated={'README_SERVER.txt':(README+linq_note).encode('utf-8'),
               'MODEL_INVENTORY.json':json.dumps(models,ensure_ascii=False,indent=2).encode('utf-8')}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as archive:
        for name,path in files:
            digest=hashlib.sha256();size=0
            with path.open('rb') as source,archive.open(PREFIX+'/'+name,'w',force_zip64=True) as destination:
                while block:=source.read(1024*1024):
                    destination.write(block);digest.update(block);size+=len(block)
            manifest['files'].append({'path':name,'bytes':size,'sha256':digest.hexdigest()})
            if path.suffix in {'.onnx','.pt','.tpu'}:print('PACKED',name,flush=True)
        for name,data in generated.items():
            archive.writestr(PREFIX+'/'+name,data)
            manifest['files'].append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        archive.writestr(PREFIX+'/SERVER_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8'))
    # A complete independent read validates decompression/CRC and every SHA256.
    with zipfile.ZipFile(target) as archive:
        bad=archive.testzip()
        if bad:raise ValueError('ZIP CRC verification failed: '+bad)
        for row in manifest['files']:
            with archive.open(PREFIX+'/'+row['path']) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=row['sha256']:
                    raise ValueError('ZIP verification failed: '+row['path'])
    print(json.dumps({**summary,'archive':str(target),'archive_bytes':target.stat().st_size,
                      'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),
                      'integrity':'every archived file CRC/SHA256 verified'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
