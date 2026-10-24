"""Fetch the authors' public FPI-Det archive; never execute downloaded code."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlparse
import zipfile
import requests

FILE_ID='1nObrvOSJfj2dzRcW9cGdV6yu6iurWcVo'


class DownloadForm(HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.fields={};self.inside=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form':self.action=a.get('action');self.inside=True
        if tag=='input' and self.inside and a.get('name'):self.fields[a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':self.inside=False


def main():
    root=Path(__file__).resolve().parents[1]/'data/datasets/fpi-det';root.mkdir(parents=True,exist_ok=True)
    target=root/'reorganized_phone_dataset_yolo.zip';session=requests.Session()
    if not target.exists():
        response=session.get('https://drive.google.com/uc',params={'export':'download','id':FILE_ID},stream=True,timeout=(20,60));response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type',''):
            form=DownloadForm();form.feed(response.text);response.close()
            if not form.action or urlparse(form.action).netloc!='drive.usercontent.google.com' or form.fields.get('id')!=FILE_ID:raise ValueError('Unexpected download response; no authentication bypass attempted')
            response=session.get(form.action,params=form.fields,stream=True,timeout=(20,90));response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type',''):raise ValueError('Public archive unavailable; login or quota may be required')
        temporary=target.with_suffix('.part');count=0
        with response,temporary.open('wb') as stream:
            for block in response.iter_content(1024*1024):
                stream.write(block);count+=len(block)
                if count%(64*1024*1024)<len(block):print('FPI_DOWNLOAD_MB',round(count/1e6),flush=True)
        temporary.replace(target)
    with target.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    with zipfile.ZipFile(target) as archive:
        entries=archive.infolist();total=sum(x.file_size for x in entries)
        if total>15*1024**3:raise ValueError('Unexpected extracted size')
        for entry in entries:
            if not (root/entry.filename).resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe archive path')
        archive.extractall(root)
    manifest={'source':'https://github.com/KvCgRv/FPI-Det','source_file_id':FILE_ID,'license':'MIT (authors dataset statement)',
        'sha256':sha,'archive_bytes':target.stat().st_size,'expanded_bytes':total,
        'warning':'Surveillance phone/head image annotations, not factory event timing labels; audit source splits and duplicate groups before training.'}
    (root/'source.json').write_text(json.dumps(manifest,indent=2))
    print('FPI_READY',json.dumps(manifest),flush=True)


if __name__=='__main__':main()
