"""Download the authors' public CC BY 4.0 archive in resumable HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import requests


def download(target,workers=6):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    metadata=requests.get('https://api.figshare.com/v2/articles/28596332',timeout=30).json()
    entry=next(f for f in metadata['files'] if f['name']=='VideoDataset.zip')
    size=entry['size'];expected=entry['computed_md5'];partial=target.with_suffix('.part')
    if target.exists() and target.stat().st_size==size:
        with target.open('rb') as stream:actual=hashlib.file_digest(stream,'md5').hexdigest()
        if actual==expected:return target
        raise ValueError('Existing archive checksum differs; preserve it and use a new target')
    cache=target.with_suffix('.chunks');cache.mkdir(exist_ok=True)
    state_file=cache/'source.json'
    identity={'url':entry['download_url'],'size':size,'md5':expected,
              'prefix_bytes':partial.stat().st_size if partial.exists() else 0}
    if state_file.exists():
        identity=json.loads(state_file.read_text())
        if identity['md5']!=expected:raise ValueError('Source version changed')
    else:state_file.write_text(json.dumps(identity))
    chunk=16*2**20
    ranges=[(start,min(start+chunk,size)-1) for start in range(identity['prefix_bytes'],size,chunk)]
    def one(bounds):
        start,end=bounds;dest=cache/str(start)
        if dest.exists() and dest.stat().st_size==end-start+1:return dest
        for attempt in range(6):
            try:
                with requests.get(entry['download_url'],headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,45)) as response:
                    if response.status_code!=206:raise ValueError('Server did not honor range')
                    temp=dest.with_suffix('.writing')
                    with temp.open('wb') as stream:
                        for data in response.iter_content(2**20):stream.write(data)
                if temp.stat().st_size!=end-start+1:raise OSError('Truncated range')
                temp.replace(dest);return dest
            except (requests.RequestException,OSError,ValueError):
                if attempt==5:raise
                time.sleep(min(2**attempt,10))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,_ in enumerate(pool.map(one,ranges)):
            if i%5==0:print('DOWNLOAD_RANGES',i+1,'/',len(ranges),flush=True)
    if not partial.exists():partial.touch()
    with partial.open('r+b') as stream:
        stream.truncate(identity['prefix_bytes']);stream.seek(identity['prefix_bytes'])
        for start,_ in ranges:
            with (cache/str(start)).open('rb') as source:shutil.copyfileobj(source,stream)
    if partial.stat().st_size!=size:raise OSError('Archive size mismatch')
    with partial.open('rb') as stream:actual=hashlib.file_digest(stream,'md5').hexdigest()
    if actual!=expected:raise ValueError('Author-provided MD5 verification failed')
    partial.replace(target)
    target.with_suffix('.source.json').write_text(json.dumps(metadata,indent=2))
    print('AUTHOR_MD5_VERIFIED',actual,'bytes',size,flush=True)
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();download(a.output,a.workers)
