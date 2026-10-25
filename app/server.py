from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import csv
import io
import os
import json
import shutil
from pathlib import Path
import threading
from typing import Literal
import uuid
import asyncio
import hashlib
import inspect
from functools import lru_cache

from fastapi import FastAPI, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,model_validator

from . import storage as db
from .core import RunConfig
from . import video
from . import operations
from . import streams
from . import live
from . import auth

MAX_PARALLEL=max(1,min(4,int(os.getenv("SAFETY_MAX_PARALLEL","2"))))
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_PARALLEL,thread_name_prefix="video")
LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    db.init()
    auth.init()
    yield
    for event in list(video.CANCEL.values()):
        event.set()
    for session in list(live.SESSIONS.values()):session.stop_event.set()


app = FastAPI(title="LinQ Safety",version="0.3.0",lifespan=lifespan)
auth.install(app)
from . import analytics_export
app.include_router(analytics_export.router)
app.mount("/static",StaticFiles(directory=Path(__file__).parent/"static"),name="static")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent/"static"/"index-v2.html")


@app.get("/api/health")
def health():
    import torch
    cuda = torch.cuda.is_available()
    return {"ok":True,"model_ready":video.MODEL.exists(),"backend":"LinQ adapter" if os.getenv("LINQ_ADAPTER") else "PyTorch",
            "device":torch.cuda.get_device_name(0) if cuda else "CPU","cuda":cuda,
            "linq_verified":False,"version":"0.3.0","max_parallel":MAX_PARALLEL,
            "fall_model_ready":(db.DATA/'models/fall-resnet18.pt').exists(),
            "trained_models":{name:(db.DATA/'models'/name).exists() for name in (
                'person-yolo12m.pt','ppe-head.onnx','ppe-vest.onnx',
                'ppe-head-classifier.onnx','ppe-torso-classifier.onnx','fall-temporal.pt')}}


@app.get("/api/samples")
def samples():
    examples=[('fall-01','Падение в комнате','Проверка падений; обучающий пример, не финальный тест.'),
              ('adl-02','Обычное действие','Нормальная активность без тревоги падения.'),
              ('helmet-missing','Без каски · учебный','Повтор изображения Construction-PPE; проверка конвейера, не качества.'),
              ('helmet-worn','В каске · учебный','Отрицательный пример для тревоги «без каски»; повтор изображения.'),
              ('fire-demo','Огонь · учебный','Повтор кадра демонстрации автора модели; не независимый тест.')]
    return [{"id":name,"name":title,"description":description,
             "available":sample_source(name)[0].exists()}
            for name,title,description in examples]


def sample_source(name):
    extras={
        'helmet-missing':('no_hardhat-demo.mp4',{'helmet_detection':True}),
        'helmet-worn':('hardhat-demo.mp4',{'helmet_detection':True}),
        'fire-demo':('fire-demo-frame-demo.mp4',{'fire_detection':True}),
    }
    if name in extras:
        file,config=extras[name]
        return db.DATA/'datasets/scenario-fixtures'/file,{'fast_motion':False,'fall_detection':False,'zones':[],**config}
    return db.DATA/f'datasets/urfd/{name}-cam0-rgb.mp4',None


class LiveRequest(BaseModel):
    source_type: Literal['network','browser']
    url: str | None = Field(default=None,max_length=2048)
    config: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode='after')
    def valid_source(self):
        if self.source_type=='network':
            if not self.url:raise ValueError('Укажите адрес видеопотока')
            streams.StreamRequest(url=self.url)
        return self


@app.post('/api/live')
def start_live(body: LiveRequest):
    validate_scenarios(body.config)
    try:return live.create(body.config,body.source_type,body.url).report()
    except PermissionError as exc:raise HTTPException(403,str(exc)) from None
    except ValueError as exc:raise HTTPException(409,str(exc)) from None


@app.get('/api/live')
def list_live():
    with live.LOCK:sessions=list(live.SESSIONS.values())
    return [s.report() for s in sessions if db.job(s.id)]


@app.get('/api/live-events')
def live_events():
    from .access import sql_scope
    clause,args=sql_scope('jobs','events.camera')
    with db.connection() as connection:
        rows=connection.execute("SELECT events.* FROM events JOIN jobs ON jobs.id=events.job_id "
            "WHERE json_extract(jobs.meta,'$.live')=1"+clause+
            " ORDER BY events.created DESC,events.rowid DESC LIMIT 100",args).fetchall()
    return [db.decode(row) for row in rows]


def get_live(session_id):
    with live.LOCK:session=live.SESSIONS.get(session_id)
    if not session or not db.job(session_id):raise HTTPException(404,'Сеанс не найден')
    return session


@app.get('/api/live/{session_id}')
def live_status(session_id: str):
    return get_live(session_id).report()


@app.post('/api/live/{session_id}/stop')
def stop_live(session_id: str):
    session=get_live(session_id);session.stop()
    return {'ok':True}


@app.post('/api/live/{session_id}/frame')
async def live_frame(session_id: str,request: Request):
    import cv2
    import numpy as np
    session=get_live(session_id)
    if session.source_type!='browser' or session.state not in ('starting','running'):
        raise HTTPException(409,'Сеанс не принимает кадры устройства')
    content=await request.body()
    if len(content)>3*1024*1024:raise HTTPException(413,'Кадр слишком большой')
    if not content.startswith(b'\xff\xd8\xff'):raise HTTPException(400,'Нужен JPEG-кадр')
    # JPEG decoding must not block event polling or the MJPEG stream.
    frame=await asyncio.to_thread(cv2.imdecode,np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
    if frame is None or frame.shape[0]*frame.shape[1]>4096*2160:
        raise HTTPException(400,'Нужен JPEG-кадр не более 4K')
    if session.ingest(frame,encoded=content) is False:raise HTTPException(409,'Эфир уже остановлен')
    return {'accepted':True}


@app.get('/api/live/{session_id}/snapshot')
def live_snapshot(session_id: str):
    session=get_live(session_id)
    jpeg=session.jpeg
    if not jpeg:raise HTTPException(404,'Кадр ещё не получен')
    return Response(jpeg,media_type='image/jpeg',headers={'Cache-Control':'no-store'})


@app.get('/api/live/{session_id}/view')
async def live_view(session_id: str,request: Request):
    session=get_live(session_id)
    async def generate():
        version=-1
        while not await request.is_disconnected():
            # The same revocation/scope check remains before every stream poll.
            # SQLite I/O must not stall all MJPEG consumers and event polling.
            user=await asyncio.to_thread(auth.resolve,request.cookies.get(auth.COOKIE))
            if not user or 'live.read' not in user['permissions']:break
            if user['organization_id']!=request.state.user['organization_id'] or user['cameras']!=request.state.user['cameras']:break
            if session.jpeg and session.version!=version:
                version=session.version
                jpeg=session.jpeg
                yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n'
            if session.state in ('stopped','failed'):break
            await asyncio.sleep(.025)
    return StreamingResponse(generate(),media_type='multipart/x-mixed-replace; boundary=frame',
                             headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})


@app.post("/api/samples/{sample_id}/import")
def import_sample(sample_id: str):
    if sample_id not in {'fall-01','adl-02','helmet-missing','helmet-worn','fire-demo'}:
        raise HTTPException(404,"Пример не найден")
    source,recommended=sample_source(sample_id)
    if not source.exists():
        raise HTTPException(404,"Файл примера пока не скачан")
    job_id=uuid.uuid4().hex;folder=db.DATA/'runs'/job_id;folder.mkdir(parents=True)
    shutil.copyfile(source,folder/'source.video')
    meta=video.probe(folder/'source.video',folder)
    meta['example']=True;meta['source']='URFD';meta['evaluation_split']='train'
    if recommended:
        meta.update(source='Selected upstream image repeated as video',evaluation_split='integration_only',recommended_config=recommended)
    db.add_job(job_id,source.name,meta)
    return db.job(job_id)


@app.get("/api/quality")
def quality():
    path=db.ROOT/'docs/quality_report.json'
    if not path.exists():
        return {"status":"not_measured","message":"Финальная проверка пока не выполнена","linq_measured":False}
    result=json.loads(path.read_text(encoding='utf-8'))
    weights=db.DATA/'models/fall-resnet18.pt'
    from .core import Rules
    from .body_classifier import BodyClassifier
    audit=result.get('audit',{})
    unchanged=(weights.exists() and _file_hash(str(weights),weights.stat().st_mtime_ns)==result.get('checkpoint_sha256')
        and hashlib.sha256(inspect.getsource(Rules.step).encode()).hexdigest()==audit.get('rules_step_sha256')
        and hashlib.sha256(inspect.getsource(BodyClassifier).encode()).hexdigest()==audit.get('body_classifier_sha256'))
    if not unchanged:
        result['status']='stale'
        result['message']='Модель или правила изменились после оценки. Сохранённые метрики относятся к прежней версии.'
    result['current_version_matches_evaluation']=bool(unchanged)
    coverage=db.ROOT/'docs/coverage.json'
    if coverage.exists():result['requirements']=json.loads(coverage.read_text(encoding='utf-8'))
    result['additional_scenarios']={'status':'not_measured','message':'Интеграционные проверки не заменяют независимую оценку новых сценариев.'}
    return result


@app.get('/api/scenarios')
def scenarios():
    from .scenario_models import required_models
    rows=[]
    for flag in ['helmet','vest','mask','harness','phone','fire','smoke','smoking','proximity','load_posture']:
        names=required_models(RunConfig(**{flag+'_detection':True}))
        rows.append({'flag':flag+'_detection','models':names,'ready':all((db.DATA/'models'/n).exists() for n in names),'quality':'not_measured'})
    return {'scenarios':rows,'runtime':'ONNX Runtime CPU; pose/fall use PyTorch','linq_verified':False}


def validate_scenarios(config):
    from .scenario_models import required_models
    if config.harness_detection and not any(z.kind=='height' for z in config.zones):
        raise HTTPException(422,'Для контроля страховочной привязи отметьте зону «Работа на высоте»')
    missing=[n for n in required_models(config) if not (db.DATA/'models'/n).exists()]
    if missing:raise HTTPException(409,'Не установлены модели: '+', '.join(missing))


@lru_cache(maxsize=8)
def _file_hash(path,mtime):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@app.get("/api/defaults")
def defaults():
    result=RunConfig().model_dump()
    path=db.DATA/'models/fall-operating-point.json'
    if path.exists():
        operating=json.loads(path.read_text())
        result.update(body_threshold=operating['threshold'],body_hold=operating['hold_s'])
    return result


from fastapi import Query


@app.get("/api/summary")
def summary(request: Request, days: Literal['7','30','all']='7', camera: str='',
            time_basis: Literal['processing','recording']='processing',
            include_experimental: bool=False, include_examples: bool=False,
            date_from: str | None=Query(None,alias='from'), date_to: str | None=Query(None,alias='to')):
    from . import analytics
    if len(camera)>200:raise HTTPException(422,'Название камеры слишком длинное')
    records=db.events();runs=db.jobs()
    try:
        operational=analytics.build_analytics(days=None if days=='all' else int(days),camera=camera,time_basis=time_basis,
            include_experimental=include_experimental,include_examples=include_examples,date_from=date_from,date_to=date_to,
            allow_events='events.read' in request.state.user['permissions'])
    except ValueError as exc:raise HTTPException(422,str(exc)) from None
    types={};cameras={};days={};hours={}
    for event in records:
        types[event['kind']]=types.get(event['kind'],0)+1
        cameras[event['camera']]=cameras.get(event['camera'],0)+1
        day=event['created'][:10];days[day]=days.get(day,0)+1
        occurred=(event['payload'] or {}).get('occurred_at')
        if occurred:
            hour=occurred[11:13]+':00';hours[hour]=hours.get(hour,0)+1
    return {"events":len(records),"new":sum(e['status']=='new' for e in records),
            "confirmed":sum(e['status']=='confirmed' for e in records),
            "false":sum(e['status']=='false' for e in records),
            "by_type":types,"by_camera":cameras,"by_processing_day":days,"by_recorded_hour":dict(sorted(hours.items())),
            "without_recorded_time":sum(not (e['payload'] or {}).get('occurred_at') for e in records),
            "running":sum(j['status'] in ('running','exporting') for j in runs),"queued":sum(j['status']=='queued' for j in runs),
            "max_parallel":MAX_PARALLEL,
            "analytics":operational}


@app.get('/api/triage')
def triage():
    return operations.triage(db.events())


@app.get('/api/review-dataset')
def review_dataset():
    return Response(operations.review_archive(),media_type='application/zip',
                    headers={'Content-Disposition':'attachment; filename=operator-review.zip'})


@app.get("/api/cameras")
def list_cameras():
    return db.cameras()


@app.post("/api/cameras")
def save_camera(config: RunConfig):
    db.save_camera(config.model_dump())
    return {"ok":True}


@app.get("/api/jobs")
def list_jobs():
    return db.jobs()


@app.post('/api/streams')
def connect_stream(body: streams.StreamRequest):
    return streams.prepare(body)


@app.post('/api/live-preview')
def live_preview(body: streams.StreamRequest):
    import cv2
    cap=None
    try:
        cap=streams.open_capture(body.url);ok,frame=cap.read()
        if not ok:raise ValueError('No frame')
        h,w=frame.shape[:2]
        if w*h>4096*2160:raise ValueError('Too large')
        encoded=cv2.imencode('.jpg',frame)[1].tobytes()
        return Response(encoded,media_type='image/jpeg')
    except Exception:
        raise HTTPException(400,'Не удалось получить кадр. Проверьте адрес и доступность камеры.') from None
    finally:
        if cap is not None:cap.release()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    result = db.job(job_id)
    if not result:
        raise HTTPException(404,"Запись не найдена")
    return result


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str):
    return {"job":get_job(job_id),"events":db.events(job_id),"quality":quality(),
            "note":"Метрики распознавания относятся к отдельной тестовой выборке URFD; события этой записи требуют проверки."}


@app.post("/api/jobs")
async def upload(file: UploadFile):
    if Path(file.filename or "").suffix.lower() not in {".mp4",".avi",".mov",".mkv",".webm"}:
        raise HTTPException(400,"Выберите видео MP4, AVI, MOV, MKV или WebM")
    job_id = uuid.uuid4().hex
    folder = db.DATA/"runs"/job_id
    folder.mkdir(parents=True)
    target = folder/"source.video"
    size = 0
    try:
        with target.open("wb") as output:
            while chunk:=await file.read(1024*1024):
                size += len(chunk)
                if size>500*1024*1024:
                    raise HTTPException(413,"Максимальный размер видео — 500 МБ")
                output.write(chunk)
        meta = video.probe(target,folder)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(400,str(exc)) from exc
    finally:
        await file.close()
    db.add_job(job_id,Path(file.filename).name,meta)
    return db.job(job_id)


@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str,config: RunConfig):
    validate_scenarios(config)
    with LOCK:
        item = get_job(job_id)
        if len(video.CANCEL)>=8:
            raise HTTPException(409,"Очередь заполнена: дождитесь завершения одной из записей")
        if item["status"]!="ready":
            raise HTTPException(409,"Этот файл уже обработан. Загрузите его снова для нового эксперимента")
        if item['meta'].get('live') and job_id not in streams.SOURCES:
            raise HTTPException(409,'Адрес камеры отсутствует после перезапуска. Добавьте подключение заново.')
        if not video.MODEL.exists() and not os.getenv("LINQ_ADAPTER"):
            raise HTTPException(409,"Модель пока не установлена: python scripts/download_model.py")
        cancel = threading.Event()
        video.CANCEL[job_id] = cancel
        db.update_job(job_id,status="queued",config=config.model_dump())
        EXECUTOR.submit(video.run,job_id,config,cancel)
    return {"status":"queued"}


@app.post("/api/jobs/{job_id}/clone")
def clone_job(job_id: str):
    item=get_job(job_id)
    if not (db.DATA/'runs'/job_id/'source.video').exists():
        raise HTTPException(409,'Дождитесь окончания записи источника')
    new_id=uuid.uuid4().hex;folder=db.DATA/'runs'/new_id;folder.mkdir(parents=True)
    shutil.copyfile(db.DATA/'runs'/job_id/'source.video',folder/'source.video')
    shutil.copyfile(db.DATA/'runs'/job_id/'preview.jpg',folder/'preview.jpg')
    meta={**item['meta'],'live':False,'replayed_live_recording':bool(item['meta'].get('live'))}
    db.add_job(new_id,item['filename'],meta)
    return db.job(new_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    get_job(job_id)
    with LOCK:
        if job_id in video.CANCEL:
            video.CANCEL[job_id].set()
    return {"ok":True}


@app.get("/api/events")
def list_events(job_id: str|None=None,kind: str|None=None,camera: str|None=None,
                zone: str|None=None,status: str|None=None,start: float|None=None,end: float|None=None):
    return db.events(job_id,kind,camera,zone,status,start,end)


class Review(BaseModel):
    status: Literal["new","confirmed","false","forwarded"]


@app.patch("/api/events/{event_id}")
def review(event_id: str,body: Review,request: Request):
    if not db.review(event_id,body.status):
        raise HTTPException(404,"Событие не найдено")
    user=request.state.user
    db.patch_payload(event_id,{'reviewed_by':user['name'],'reviewed_by_id':user['id']})
    auth.audit(user,'event.reviewed',event_id,{'status':body.status})
    return {"ok":True}


@app.get("/api/export")
def export(job_id: str|None=None):
    events = db.events(job_id)
    output = io.StringIO()
    writer = csv.writer(output)
    columns = ["id","camera","kind","zone","track_id","start_s","alert_s","status","created"]
    writer.writerow(columns)
    for event in events:
        values = [event[k] for k in columns]
        values = ["'"+v if isinstance(v,str) and v.startswith(("=","+","-","@")) else v for v in values]
        writer.writerow(values)
    return Response("\ufeff"+output.getvalue(),media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition":"attachment; filename=events.csv"})


@app.get("/media/{path:path}")
def media(path: str):
    root = (db.DATA/"runs").resolve()
    target = (db.DATA/path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404,"Файл не найден")
    if target.suffix not in {".jpg",".mp4",".json"}:
        raise HTTPException(404,"Файл недоступен")
    parts=target.relative_to(root).parts
    if len(parts)<2 or not db.job(parts[0]):raise HTTPException(404,'Файл не найден')
    # A scoped foreman must never download a whole run's unfiltered JSON report.
    from .access import current_scope
    scope=current_scope.get()
    if scope and scope.cameras is not None and target.suffix=='.json':raise HTTPException(403,'Отчёт целиком недоступен для доступа по камерам')
    return FileResponse(target)
