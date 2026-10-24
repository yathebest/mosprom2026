"""Small atomic progress records and bounded local staging for prepared FPI data."""
from concurrent.futures import FIRST_COMPLETED,ThreadPoolExecutor,wait
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time


def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    # Windows readers can briefly hold a non-delete-shared handle. Linux/Kaggle
    # replaces immediately; bounded retries keep local monitoring atomic too.
    for attempt in range(8):
        try:os.replace(temporary,path);break
        except PermissionError:
            if attempt==7:raise
            time.sleep(min(.01*2**attempt,.1))


class Progress:
    """Heartbeat means the Python process is alive, not that GPU work advances."""
    def __init__(self,path,interval=10):
        self.path=Path(path);self.interval=interval;self.lock=threading.RLock()
        self.started=time.monotonic();self.work_at=self.started;self.phase_at=self.started;self.written=0
        self.payload={'stage':'starting','pid':os.getpid(),'sequence':0};self.stop=threading.Event()
        self.thread=threading.Thread(target=self._heartbeat,name='training-heartbeat',daemon=True)
        self.update('starting',force=True);self.thread.start()
    def _write(self):
        now=time.monotonic();self.payload['sequence']+=1
        atomic_json(self.path,{**self.payload,'heartbeat_utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_s':round(now-self.started,1),'stage_elapsed_s':round(now-self.phase_at,1),
            'last_work_age_s':round(now-self.work_at,1)})
        self.written=now
    def update(self,stage=None,force=False,**fields):
        with self.lock:
            changed=stage is not None and stage!=self.payload['stage']
            if changed:
                self.phase_at=time.monotonic()
                self.payload={key:self.payload[key] for key in ['stage','pid','sequence']}
            if stage is not None:self.payload['stage']=stage
            self.payload.update(fields);self.work_at=time.monotonic()
            if changed or force or self.work_at-self.written>=2:self._write()
            if changed:print('PHONE_HEAD_STAGE',self.payload['stage'],json.dumps(fields),flush=True)
    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:self._write()
    def close(self):
        self.stop.set();self.thread.join(timeout=2)
        with self.lock:self._write()


def remap_labels(text):
    rows=[]
    for line in text.splitlines():
        if not line.strip():continue
        fields=line.split()
        if len(fields)!=5 or fields[0] not in {'0','1'}:raise ValueError('Use normalized prepared FPI labels')
        values=[float(v) for v in fields[1:]]
        if any(not math.isfinite(v) or not 0<=v<=1 for v in values) or min(values[2:])<=0:
            raise ValueError('Invalid normalized FPI box')
        fields[0]='67' if fields[0]=='0' else '0';rows.append(' '.join(fields))
    return '\n'.join(rows)+'\n'


def materialize_fpi(source,mapped,progress,workers=8,include_test=False):
    """Copy bytes locally once; workers never resolve per-file remote paths."""
    source=Path(source);mapped=Path(mapped)
    if not source.is_absolute():raise ValueError('Resolve the input root once before staging')
    mapped.mkdir(parents=True,exist_ok=True);marker=mapped/'materialization.json'
    identity=hashlib.sha256(str(source).encode())
    for name in ['FPI_PREPARED_SOURCE.json','manifest.json']:
        file=source/name
        if file.exists():identity.update(file.read_bytes())  # Two metadata reads, not one per image.
    identity=identity.hexdigest();prior={}
    if marker.exists():prior=json.loads(marker.read_text())
    reuse=prior.get('source_identity')==identity
    totals={};copied=0;skipped=0
    for split in (['train','val','test'] if include_test else ['train','val']):
        src_images=source/split/'images';src_labels=source/split/'labels'
        names=sorted(name for name in os.listdir(src_images) if name.lower().endswith('.jpg'))
        images=mapped/split/'images';labels=mapped/split/'labels'
        images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        progress.update('materialize',split=split,completed=0,total=len(names),workers=workers,force=True)
        def copy_one(name):
            target=images/name;label=labels/(Path(name).stem+'.txt')
            # is_symlink checks the local directory entry without following it
            # back to FUSE; old remote symlinks are deliberately replaced.
            if reuse and not target.is_symlink() and target.is_file() and label.is_file():return False
            text=remap_labels((src_labels/label.name).read_text(encoding='utf-8'))
            temporary=target.with_name(target.name+'.writing')
            try:
                with (src_images/name).open('rb') as incoming,temporary.open('wb') as outgoing:
                    shutil.copyfileobj(incoming,outgoing,length=1024*1024)
                os.replace(temporary,target)
                temporary_label=label.with_name(label.name+'.writing');temporary_label.write_text(text,encoding='utf-8');os.replace(temporary_label,label)
            finally:temporary.unlink(missing_ok=True)
            return True
        completed=0;iterator=iter(names)
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='fpi-local-copy') as pool:
            pending=set()
            def fill():
                while len(pending)<workers*2:
                    try:name=next(iterator)
                    except StopIteration:break
                    pending.add(pool.submit(copy_one,name))
            fill()
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    if future.result():copied+=1
                    else:skipped+=1
                    completed+=1
                progress.update('materialize',split=split,completed=completed,total=len(names),copied=copied,reused=skipped)
                fill()
        totals[split]=completed
    result={'source_identity':identity,'counts':totals,'copied':copied,'reused':skipped,
        'images':'local regular files, byte-preserving copy','test_prepared':include_test}
    atomic_json(marker,result);progress.update('materialize_complete',counts=totals,force=True)
    return result


def bounded_tail(path,limit=2048):
    """Read at most limit bytes regardless of the tqdm log's total size."""
    path=Path(path)
    if not path.exists():return ''
    with path.open('rb') as stream:
        stream.seek(0,2);stream.seek(max(0,stream.tell()-limit));return stream.read(limit).decode('utf-8',errors='replace')
