"""Verify the extracted server bundle with Python's standard library only."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, quick=False):
    root = Path(root).resolve()
    manifest = json.loads((root/'SERVER_MANIFEST.json').read_text(encoding='utf-8'))
    errors = []
    for row in manifest['files']:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(row['path']+': missing or outside package')
        elif path.stat().st_size != row['bytes']:
            errors.append(row['path']+': size mismatch')
        elif not quick:
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                    errors.append(row['path']+': SHA256 mismatch')
    return manifest, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    manifest, errors = verify(args.root, args.quick)
    print(json.dumps({'files':len(manifest['files']), 'model_files':manifest['model_files'],
                      'mode':'size' if args.quick else 'SHA256', 'errors':errors}, indent=2))
    raise SystemExit(bool(errors))
