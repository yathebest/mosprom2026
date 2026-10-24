"""Optional private Kaggle Dataset backup using the official authenticated SDK.

No credentials are read or printed by this script. A lack of write authentication
is reported honestly; local epoch checkpoints remain available independently.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from training_checkpoint import atomic_copy


def snapshot(root, destination):
    files=[]
    for source in sorted(root.rglob('*')):
        if not source.is_file():continue
        if source.suffix not in {'.pt','.json','.csv','.yaml','.log'}:continue
        if source.name.endswith(('-previous.pt','resume-previous.pt')):continue
        target=destination/source.relative_to(root)
        atomic_copy(source,target)
        if target.suffix=='.pt':
            try:
                with zipfile.ZipFile(target) as archive:
                    if archive.testzip() is not None:raise ValueError('Checkpoint CRC mismatch')
            except (zipfile.BadZipFile,ValueError,EOFError):
                target.unlink();continue  # A non-atomic library best/last write may still be in progress.
        files.append({'path':str(source.relative_to(root)),
                      'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
    (destination/'backup-manifest.json').write_text(json.dumps({'created_utc':time.time(),'files':files},indent=2))
    return files


def upload_once(root, handle, probe=False):
    with tempfile.TemporaryDirectory(prefix='linq-checkpoint-backup-') as name:
        destination=Path(name)
        if probe:
            (destination/'backup-probe.json').write_text(json.dumps({'purpose':'private LinQ training checkpoint backup','time':time.time()}))
        else:snapshot(root,destination)
        # Native authentication may support downloads without write access.
        # Do not call login() or silently request/create an API credential.
        import kagglehub
        kagglehub.dataset_upload(handle,str(destination),version_notes='LinQ checkpoint snapshot; optimizer and best weights')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--handle',required=True)
    p.add_argument('--once',action='store_true');p.add_argument('--probe',action='store_true')
    p.add_argument('--interval',type=int,default=1800)
    p.add_argument('--stop-file',type=Path)
    args=p.parse_args();root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    status=root/'remote-backup-status.json'
    if args.once:
        try:
            upload_once(root,args.handle,args.probe)
            status.write_text(json.dumps({'verified_upload':True,'handle':args.handle,'time':time.time(),'probe':args.probe}))
            print('REMOTE_BACKUP_UPLOAD_CONFIRMED',args.handle,flush=True)
        except Exception as error:
            status.write_text(json.dumps({'verified_upload':False,'handle':args.handle,'time':time.time(),
                'error_type':type(error).__name__,'warning':'Local files alone do not survive a destroyed VM.'}))
            print('REMOTE_BACKUP_UNAVAILABLE',type(error).__name__,'No credential or response body logged.',flush=True)
            raise SystemExit(2)
        return
    seen=set();last_upload=0
    while True:
        if args.stop_file and args.stop_file.exists():break
        checkpoints={str(p.relative_to(root)) for p in root.rglob('*.pt')
                     if p.name=='resume-last.pt' or p.name.endswith(('classifier-last.pt','temporal-last.pt'))}
        if not checkpoints or (not(checkpoints-seen) and time.monotonic()-last_upload<args.interval):
            time.sleep(20);continue
        command=[sys.executable,__file__,'--root',str(root),'--handle',args.handle,'--once']
        try:
            result=subprocess.run(command,timeout=300,check=False)
            if result.returncode==0:seen=checkpoints;last_upload=time.monotonic()
        except subprocess.TimeoutExpired:print('REMOTE_BACKUP_TIMEOUT; training continues',flush=True)
        time.sleep(20 if last_upload else 60)


if __name__=='__main__':main()
