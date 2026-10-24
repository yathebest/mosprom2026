import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from training_observability import Progress,bounded_tail,materialize_fpi


def test_local_staging_copies_bytes_remaps_classes_and_does_not_read_test(tmp_path):
    source=tmp_path/'source'
    for split in ['train','val','test']:
        (source/split/'images').mkdir(parents=True);(source/split/'labels').mkdir()
        (source/split/'images/one.jpg').write_bytes(b'original-image-bytes')
        (source/split/'labels/one.txt').write_text('0 .5 .5 .2 .2\n1 .4 .4 .1 .1\n' if split!='test' else 'deliberately invalid held-out label')
    progress=Progress(tmp_path/'progress.json',interval=.02)
    try:
        mapped=tmp_path/'mapped';report=materialize_fpi(source,mapped,progress,workers=2)
        assert report['counts']=={'train':1,'val':1} and report['copied']==2
        assert (mapped/'train/images/one.jpg').read_bytes()==b'original-image-bytes'
        assert not (mapped/'train/images/one.jpg').is_symlink()
        assert (mapped/'val/labels/one.txt').read_text().splitlines()==['67 .5 .5 .2 .2','0 .4 .4 .1 .1']
        assert not (mapped/'test').exists()
        again=materialize_fpi(source,mapped,progress,workers=2)
        assert again['copied']==0 and again['reused']==2
    finally:progress.close()


def test_heartbeat_distinguishes_process_liveness_from_work_and_clears_stale_stage_fields(tmp_path):
    target=tmp_path/'progress.json';progress=Progress(target,interval=.02)
    try:
        progress.update('materialize',completed=25,total=100,force=True)
        before=json.loads(target.read_text())['sequence']
        deadline=time.monotonic()+1
        while json.loads(target.read_text())['sequence']<=before and time.monotonic()<deadline:time.sleep(.01)
        state=json.loads(target.read_text())
        assert state['sequence']>before and state['completed']==25
        progress.update('scan',detail='loader setup',force=True)
        assert 'completed' not in json.loads(target.read_text())
    finally:progress.close()


def test_tqdm_tail_is_bounded_even_for_large_logs(tmp_path):
    file=tmp_path/'large.log';file.write_bytes(b'head-not-to-be-read\n'+b'x'*3_000_000+b'\rLAST_BATCH 17/100\n')
    tail=bounded_tail(file,2048)
    assert len(tail)<=2048 and tail.endswith('LAST_BATCH 17/100\n') and 'head-not-to-be-read' not in tail


def test_generated_notebook_and_embedded_programs_compile(tmp_path,monkeypatch):
    import ast
    import shutil
    spec=importlib.util.spec_from_file_location('factory_notebook_builder',ROOT/'scripts/build_factory_objects_notebook.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    (tmp_path/'scripts').mkdir()
    for name in ['train_fpi_objects.py','train_cigarette_candidate.py','training_checkpoint.py','training_observability.py','backup_training.py']:
        shutil.copyfile(ROOT/'scripts'/name,tmp_path/'scripts'/name)
    monkeypatch.setattr(builder,'ROOT',tmp_path);builder.main()
    book=json.loads((tmp_path/'data/cache/kaggle-factory-objects/linq-factory-objects-2026.ipynb').read_text(encoding='utf-8'))
    for cell in book['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'generated-cell','exec')
    files=ast.literal_eval(ast.parse(''.join(book['cells'][2]['source'])).body[0].value)
    assert 'training_observability.py' in files
    for name,source in files.items():compile(source,name,'exec')
    run=''.join(book['cells'][-1]['source'])
    assert 'PHONE_HEAD_ONLY=True' in run and '--resume' not in run
