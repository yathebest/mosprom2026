"""Download public training sources; preserve provenance and actual counts.

SH17 is optional (about 14 GB upstream). No credentials are stored. Archives
are data only; never run Python from a dataset archive.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

SOURCES = {
    'construction': {'url':'https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip',
                     'license':'AGPL-3.0','page':'https://docs.ultralytics.com/datasets/detect/construction-ppe'},
    'site_safety': {'url':'https://www.kaggle.com/api/v1/datasets/download/snehilsanyal/construction-site-safety-image-dataset-roboflow',
                    'license':'CC BY 4.0 (source declaration)','page':'https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow'},
    'shwd': {'url':'https://drive.google.com/uc?id=1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX',
             'license':'MIT repository; includes SCUT-HEAD-derived images; retain source notices',
             'page':'https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset'},
    'sh17': {'url':'https://www.kaggle.com/api/v1/datasets/download/mugheesahmad/sh17-dataset-for-ppe-detection',
             'license':'CC BY-NC-SA 4.0','page':'https://github.com/ahmadmughees/SH17dataset'},
    'hardhat_vest': {'url':'https://www.kaggle.com/api/v1/datasets/download/muhammetzahitaydn/hardhat-vest-dataset-v3',
                     'license':'Check archive source notices; Kaggle declaration alone is not a provenance audit',
                     'page':'https://www.kaggle.com/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3'},
}
EXPECTED_SHA256={
    'construction':'bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc',
    'site_safety':'f1acf02c7116a09cdbb2b3ace1ee9b597977c0eb0bd0025e8949f81091fb2a3d',
    'shwd':'2f6c5c347b778271bc8b27c2d61b731fc5fb8cdb4f45fce583dd2f82e6a46401',
    'hardhat_vest':'61048c7f9a52c8672e1989e338e61781515056ae3809c2fa7a40477ddf20afaf',
}


def download_source(name,root):
    spec=SOURCES[name];root=Path(root);root.mkdir(parents=True,exist_ok=True)
    target=root/(name+'.zip');partial=target.with_suffix('.zip.part')
    if not target.exists():
        if name=='shwd':
            import gdown
            if not gdown.download(spec['url'],str(partial),quiet=True):raise RuntimeError('SHWD download failed')
        else:
            offset=partial.stat().st_size if partial.exists() else 0
            headers={'User-Agent':'LinQSafety dataset preparation/1.0'}
            if offset:headers['Range']=f'bytes={offset}-'
            request=urllib.request.Request(spec['url'],headers=headers)
            with urllib.request.urlopen(request,timeout=120) as response:
                mode='ab' if response.status==206 and offset else 'wb'
                with partial.open(mode) as stream:
                    while chunk:=response.read(1024*1024):stream.write(chunk)
        if not zipfile.is_zipfile(partial):raise ValueError(f'{name}: response is not a ZIP dataset')
        partial.replace(target)
    digest=hashlib.sha256()
    with target.open('rb') as stream:
        while chunk:=stream.read(8*1024*1024):digest.update(chunk)
    if name in EXPECTED_SHA256 and digest.hexdigest()!=EXPECTED_SHA256[name]:
        raise ValueError(f'{name}: source archive changed; review its version before using new data')
    folder=root/name;marker=folder/'.extracted'
    if not marker.exists():
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(target) as archive:
            for item in archive.infolist():
                path=(folder/item.filename).resolve()
                if not path.is_relative_to(folder.resolve()):raise ValueError('Unsafe archive path')
            archive.extractall(folder)
        marker.write_text('ok',encoding='utf-8')
    report={'source':name,**spec,'archive_sha256':digest.hexdigest(),'bytes':target.stat().st_size,
            'images':sum(p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'} for p in folder.rglob('*')),
            'status':'downloaded','folder':name}
    (root/(name+'-source.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',choices=SOURCES,default=['construction','site_safety','shwd'])
    p.add_argument('--root',default='data/datasets/ppe-sources');args=p.parse_args()
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(download_source,s,args.root):s for s in args.sources}
        for f,name in [(f,futures[f]) for f in concurrent.futures.as_completed(futures)]:
            try:f.result()
            except Exception as exc:failures.append({'source':name,'status':'failed','error':str(exc)})
    Path(args.root,'download-failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    return bool(failures)


if __name__=='__main__':raise SystemExit(main())
