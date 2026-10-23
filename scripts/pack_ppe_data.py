"""Package only generated public training data and notices, never app state."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import zipfile


def pack(data,sources,output,summary=None):
    data=Path(data);roots=[Path(p) for p in sources];output=Path(output)
    if not (data/'manifest.jsonl').exists():raise ValueError('Prepared manifest missing')
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix(output.suffix+'.partial')
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        count=0
        paths=sorted(p for p in data.rglob('*') if p.is_file())
        with ThreadPoolExecutor(max_workers=8) as pool:
            for offset in range(0,len(paths),128):
                batch=paths[offset:offset+128]
                for p,content in zip(batch,pool.map(Path.read_bytes,batch)):
                    z.writestr(p.relative_to(data).as_posix(),content,
                        compress_type=zipfile.ZIP_STORED if p.suffix.lower() in {'.jpg','.png','.jpeg'} else zipfile.ZIP_DEFLATED)
                    count+=1
                    if count%10000==0:print('Packed',count,'/',len(paths),'files',flush=True)
        if summary and Path(summary).exists():z.write(summary,'dataset-summary.json')
        for sources in roots:
            for p in sources.glob('*-source.json'):z.write(p,'source-notices/'+p.name)
            for p in sources.glob('*-metadata.json'):z.write(p,'source-notices/'+p.name)
            for source in sources.iterdir():
                if not source.is_dir():continue
                for p in source.rglob('*'):
                    if p.is_file() and p.name.lower().startswith(('readme','license')) and p.stat().st_size<100000:
                        z.write(p,'source-notices/'+source.name+'/'+p.relative_to(source).as_posix())
    partial.replace(output)
    with output.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    result={'archive':str(output),'bytes':output.stat().st_size,'sha256':digest,'prepared_files':count}
    output.with_suffix('.sha256.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/datasets/ppe-prepared')
    p.add_argument('--sources',nargs='+',default=['data/datasets/ppe-sources']);p.add_argument('--output',required=True)
    p.add_argument('--summary',default='docs/ppe-dataset-summary.json')
    a=p.parse_args();pack(a.data,a.sources,a.output,a.summary)
