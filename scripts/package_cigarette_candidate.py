"""Package the UI-downloaded Kaggle artifacts without replacing runtime models."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'data/cache/kaggle-cigarette-completed'
manifest = {
    'source': 'https://www.kaggle.com/code/yathebest/linq-factory-objects-training-2026/output?scriptVersionId=361806401',
    'kernel_version': 361806401,
    'task': 'cigarette object detector candidate',
    'completed_epochs': 13,
    'runtime_replaced': False,
    'checkpoint_in_kaggle': {'epoch_index': 12, 'optimizer_present': True,
                            'optimizer_parameter_states': 381, 'bytes': 161970667,
                            'sha256': '9ea027ee535a80e6e1010ba15e364368d8f09e7893b336733d1c3ae11d4b8bc2'},
    'local_files': {},
    'validation_report': 'docs/cigarette-kaggle-candidate-validation.json',
    'decision': 'Keep current runtime: candidate loses recall/F1 at the same threshold on validation.',
    'limitations': ['CigDet validation object metrics, not factory smoking episode metrics.',
                    'best.pt is for inference/fine tuning; optimizer resume-last.pt remains in pinned Kaggle output.',
                    'No test data read during this comparison.'],
}
for name in ['cigarette.onnx', 'best.pt']:
    file = folder/name
    with file.open('rb') as stream:
        manifest['local_files'][name] = {'bytes': file.stat().st_size,
                                       'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
(ROOT/'docs/cigarette-kaggle-artifact.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
(folder/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
(folder/'README.txt').write_text(
    'Saved cigarette candidate from Kaggle version 361806401.\n'
    'Not installed into the application: validation recall regresses.\n'
    'cigarette.onnx: checked with ONNX checker. best.pt: best inference weights.\n'
    'Full optimizer checkpoint resume-last.pt remains in the version-pinned Kaggle output.\n'
    'See manifest.json for SHA256 and docs/cigarette-kaggle-candidate-validation.json for comparison.\n', encoding='utf-8')
archive = folder/'linq-cigarette-kaggle-361806401.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as output:
    for name in ['cigarette.onnx', 'best.pt', 'manifest.json', 'README.txt']:
        output.write(folder/name, name)
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'runtime_replaced': False}))
