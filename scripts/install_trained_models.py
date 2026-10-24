"""Install the six PPE runtime artifacts from our completed Kaggle output ZIP.

Only known filenames are copied; notebooks, scripts and optimizer snapshots are
never executed or installed. Existing runtime files are backed up before replace.
"""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REQUIRED=('person-yolo12m.pt','person-yolo12m.onnx','ppe-head.onnx','ppe-vest.onnx',
          'ppe-head-classifier.onnx','ppe-torso-classifier.onnx')
REPORTS=('people-metrics.json','head_ppe-metrics.json','vest_ppe-metrics.json',
         'head_ppe-classifier-metrics.json','vest_ppe-classifier-metrics.json',
         'training-progress.json','night-verification.json')


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def extract_selected(archive,stage):
    found={}
    for entry in archive.infolist():
        name=Path(entry.filename.replace('\\','/')).name
        if name not in REQUIRED+REPORTS or entry.is_dir():continue
        if entry.file_size>1024**3:raise ValueError('Unexpectedly large runtime artifact')
        target=stage/name
        # Duplicate exports in results.zip must be byte-for-byte identical.
        with tempfile.TemporaryFile() as temporary:
            with archive.open(entry) as source:shutil.copyfileobj(source,temporary)
            temporary.seek(0)
            sha=hashlib.file_digest(temporary,'sha256').hexdigest();temporary.seek(0)
            if name in found:
                if found[name]!=sha:raise ValueError(f'Ambiguous artifacts: {name}')
                continue
            with target.open('wb') as output:shutil.copyfileobj(temporary,output)
            found[name]=sha
    return found


def install(source,models):
    models.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'data/cache';cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='install-models-',dir=cache) as temporary:
        stage=Path(temporary)
        with zipfile.ZipFile(source) as archive:
            found=extract_selected(archive,stage)
            if not all(name in found for name in REQUIRED):
                nested=[x for x in archive.infolist() if Path(x.filename).name=='linq-night-trained-models.zip']
                if len(nested)==1:
                    nested_path=stage/'nested.zip'
                    with archive.open(nested[0]) as src,nested_path.open('wb') as dst:shutil.copyfileobj(src,dst)
                    with zipfile.ZipFile(nested_path) as inner:found.update(extract_selected(inner,stage))
        missing=[name for name in REQUIRED if not (stage/name).exists()]
        if missing:raise ValueError('Missing trained artifacts: '+', '.join(missing))
        import onnx
        for name in REQUIRED:
            if not name.endswith('.onnx'):continue
            graph=onnx.load(stage/name,load_external_data=False)
            onnx.checker.check_model(graph)
            meta={x.key:x.value for x in graph.metadata_props}
            if 'classifier' in name:
                labels=json.loads(meta['class_names'])
                expected={'hardhat','no_hardhat'} if 'head' in name else {'vest','no_vest'}
                if set(labels)!=expected or meta.get('crop_policy')!='proposal_context_1.3':
                    raise ValueError(f'Unexpected classifier labels/preprocessing: {name}')
            else:
                names=ast.literal_eval(meta['names'])
                names=list(names.values()) if isinstance(names,dict) else names
                expected={'person'} if name.startswith('person') else {'hardhat','no_hardhat'} if 'head' in name else {'vest','no_vest'}
                if set(names)!=expected:raise ValueError(f'Unexpected detector classes: {name}: {names}')
            del graph
        from ultralytics import YOLO
        person=YOLO(str(stage/'person-yolo12m.pt'),task='detect')
        if set(person.names.values())!={'person'}:raise ValueError('Unexpected person checkpoint classes')
        del person
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup=cache/('models-before-'+stamp);backup.mkdir()
        installed=[]
        try:
            for name in REQUIRED+REPORTS:
                candidate=stage/name
                if not candidate.exists():continue
                target=models/name
                if target.exists():shutil.copy2(target,backup/name)
                shutil.copyfile(candidate,models/(name+'.installing'))
                (models/(name+'.installing')).replace(target);installed.append(name)
        except Exception:
            for name in installed:
                if (backup/name).exists():shutil.copy2(backup/name,models/name)
                else:(models/name).unlink(missing_ok=True)
            raise
        report={'installed_utc':stamp,'source_archive':str(source.resolve()),
                'kaggle_version':361639587,'files':{name:{'sha256':digest(models/name),'bytes':(models/name).stat().st_size} for name in installed},
                'backup_directory':str(backup),'hardware':'local PyTorch/ONNX; LinQ hardware verification remains separate'}
        (models/'trained-installation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path)
    parser.add_argument('--models',type=Path,default=ROOT/'data/models')
    args=parser.parse_args();install(args.archive,args.models)
