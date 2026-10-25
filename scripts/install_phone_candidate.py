"""Install a separately compared phone candidate; keep a reversible local backup."""
from datetime import datetime,timezone
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def install(candidate,comparison,models=None):
    candidate=Path(candidate);comparison=Path(comparison);models=Path(models or ROOT/'data/models')
    report=json.loads(comparison.read_text(encoding='utf-8'))
    if report.get('deployment_decision')!='accept':raise ValueError('Independent comparison must approve candidate')
    if report.get('confidence')!=.3 or report.get('phone_only_decoder_enabled') is not False:
        raise ValueError('This release keeps confidence 0.3 and the original class decoder')
    expected=report['models']['candidate']['sha256']
    if digest(candidate)!=expected:raise ValueError('Candidate differs from compared weights')
    import onnx
    graph=onnx.load(str(candidate),load_external_data=False);onnx.checker.check_model(graph)
    metadata={prop.key:prop.value for prop in graph.metadata_props}
    names=ast.literal_eval(metadata.get('names','{}'))
    if not isinstance(names,dict) or names.get(0)!='head_proposal' or names.get(67)!='cell_phone' or len(names)!=80:
        raise ValueError('Expected head_proposal=0, cell_phone=67, 80-output checkpoint')
    inputs=[[dim.dim_value for dim in item.type.tensor_type.shape.dim] for item in graph.graph.input]
    outputs=[[dim.dim_value for dim in item.type.tensor_type.shape.dim] for item in graph.graph.output]
    if inputs!=[[1,3,640,640]] or outputs!=[[1,84,8400]]:
        raise ValueError('Unexpected static phone input/output shape')
    del graph
    target=models/'phone-objects.onnx';manifest=models/'scenario-manifest.json'
    baseline_sha=report['models']['runtime']['sha256']
    if not target.exists() or digest(target)!=baseline_sha:raise ValueError('Runtime baseline changed after comparison')
    rows=json.loads(manifest.read_text(encoding='utf-8'))
    matches=[row for row in rows if row.get('file')=='phone-objects.onnx']
    if len(matches)!=1:raise ValueError('Expected one phone manifest record')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=ROOT/'data/cache'/('models-before-phone-'+stamp);backup.mkdir(parents=True)
    shutil.copy2(target,backup/target.name);shutil.copy2(manifest,backup/manifest.name)
    staged=models/'phone-objects.onnx.installing'
    try:
        shutil.copyfile(candidate,staged)
        if digest(staged)!=expected:raise ValueError('Staged candidate checksum differs')
        staged.replace(target)
        row=matches[0]
        row.update(sha256=expected,bytes=target.stat().st_size,source='Locally trained FPI-Det phone/head candidate; evaluation report in docs',
            conversion={'file':target.name,'source_checkpoint':'phone-head/weights/best.pt','onnx_sha256':expected,
                'export':'static FP32 1x3x640x640; host original argmax/NMS','classes':names},
            admitted_classes={'67':'cell_phone'},ignored_classes='All other outputs, including class 0 head_proposal, are never person proposals.',
            quality={'scope':'Frozen FPI validation object boxes, not independent factory phone-use event accuracy',
                'confidence':.3,'comparison':str(comparison.relative_to(ROOT)) if comparison.is_relative_to(ROOT) else comparison.name},
            runtime='ONNX Runtime CPU/CUDA selected by environment',linq_compiled=False)
        manifest.write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
        receipt={'installed_at':stamp,'file':target.name,'sha256':expected,'previous_sha256':baseline_sha,
            'backup_directory':str(backup),'admitted_classes':{'67':'cell_phone'},'class_0':'head_proposal_not_person',
            'confidence_unchanged':.3,'decoder_unchanged':True,'event_accuracy_measured':False,'linq_compiled':False}
        save=backup/'installation.json';save.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        return receipt
    except BaseException:
        shutil.copy2(backup/target.name,target);shutil.copy2(backup/manifest.name,manifest);staged.unlink(missing_ok=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(install(args.candidate.resolve(),args.comparison.resolve()),indent=2))
