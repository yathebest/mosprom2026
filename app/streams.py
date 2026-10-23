"""Bounded live-camera sessions. URLs exist only in process memory."""
from datetime import datetime,timezone
from pathlib import Path
import threading
from urllib.parse import urlsplit
import uuid

import cv2
from fastapi import HTTPException
from pydantic import BaseModel,Field,model_validator

from . import storage as db

SOURCES={}
SOURCE_LOCK=threading.Lock()


class StreamRequest(BaseModel):
    url: str = Field(min_length=8,max_length=2048)
    duration_s: int = Field(default=60,ge=5,le=300)

    @model_validator(mode='after')
    def supported_url(self):
        parsed=urlsplit(self.url)
        if parsed.scheme not in {'rtsp','http','https'} or not parsed.hostname:
            raise ValueError('Нужен RTSP, HTTP или HTTPS видеопоток')
        return self


def open_capture(url):
    # Bound network waits. Credentials are never copied to metadata or exports.
    cap=cv2.VideoCapture(url,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,
                                          cv2.CAP_PROP_READ_TIMEOUT_MSEC,3000])
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Источник не отвечает или формат потока не поддерживается')
    return cap


def prepare(request):
    with SOURCE_LOCK:
        if len(SOURCES)>=8:raise HTTPException(409,'Сначала запустите подготовленные подключения или перезапустите сервер')
    cap=None
    try:
        cap=open_capture(request.url)
        ok,frame=cap.read()
        if not ok:raise ValueError('Источник не передал первый кадр')
        h,w=frame.shape[:2]
        if w*h>4096*2160:raise ValueError('Разрешение потока должно быть не выше 4K')
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if not 1<=fps<=60:fps=25.
        job_id=uuid.uuid4().hex;folder=db.DATA/'runs'/job_id;folder.mkdir(parents=True)
        cv2.imwrite(str(folder/'preview.jpg'),frame)
        meta={'width':w,'height':h,'fps':fps,'frames':round(fps*request.duration_s),
              'duration_s':request.duration_s,'live':True,'constant_fps_required':True}
        db.add_job(job_id,'Прямой видеопоток',meta)
        with SOURCE_LOCK:SOURCES[job_id]=request.url
        return db.job(job_id)
    except HTTPException:raise
    except Exception as exc:
        # Never include the URL in errors: it can contain a camera password.
        raise HTTPException(400,'Не удалось получить кадр. Проверьте адрес, доступность и формат камеры.') from None
    finally:
        if cap is not None:cap.release()
