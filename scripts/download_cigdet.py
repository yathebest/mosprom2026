"""Download the public CC-BY-4.0 CigDet v1 archive with a pinned checksum."""
import hashlib
import json
from pathlib import Path
import zipfile
import requests

SHA256='ef2ab181b62e93198654a333ae82b80bf3e08eafcc53e3c865340665936bae41'
URL='https://data.mendeley.com/public-files/datasets/6hyrr8typ7/files/24e3327e-155e-4565-ab44-8b45dc6445ee/file_downloaded'


def main():
    root=Path(__file__).resolve().parents[1]/'data/datasets/cigdet';root.mkdir(parents=True,exist_ok=True)
    target=root/'CigDet_dataset.zip'
    if not target.exists():
        temporary=target.with_suffix('.part')
        with requests.get(URL,stream=True,timeout=(20,90)) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(1024*1024):stream.write(chunk)
        temporary.replace(target)
    with target.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=SHA256:raise ValueError('CigDet checksum differs; archive was not extracted')
    with zipfile.ZipFile(target) as archive:
        for entry in archive.infolist():
            if not (root/entry.filename).resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe archive path')
        archive.extractall(root)
    source={'url':'https://data.mendeley.com/datasets/6hyrr8typ7/1','author':'Ali Khan',
        'license':'CC-BY-4.0','sha256':actual,'scope':'Web cigarette boxes, not factory smoking-event annotations'}
    (root/'source.json').write_text(json.dumps(source,indent=2))
    print('CIGDET_VERIFIED',actual,flush=True)


if __name__=='__main__':main()
