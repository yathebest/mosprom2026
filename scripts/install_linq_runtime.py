"""Install the organizer wire package from its ZIP or the integration kit.

Only the five protocol files are installed; no archived code is executed.
Existing differing files are refused rather than overwritten.
"""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ('__init__.py', '_io.py', 'headers.py', 'message.proto', 'message_pb2.py')


def install(archive_path, root=ROOT):
    root = Path(root).resolve()
    destination = root / 'tpu_pycloud_format'
    selected = {}
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename.replace('\\', '/'))
            if entry.is_dir() or path.name not in FILES:
                continue
            # Support package ZIPs with bare files and kits with nested folders.
            if len(path.parts) > 1 and path.parent.name != 'tpu_pycloud_format':
                continue
            if (path.is_absolute() or '..' in path.parts or ':' in path.as_posix()
                    or stat.S_ISLNK(entry.external_attr >> 16) or entry.file_size > 1024**2):
                raise ValueError('Unsafe protocol file: ' + entry.filename)
            data = archive.read(entry)
            if path.name in selected and selected[path.name] != data:
                raise ValueError('Conflicting protocol file: ' + path.name)
            selected[path.name] = data
    missing = set(FILES) - selected.keys()
    if missing:
        raise ValueError('Missing organizer files: ' + ', '.join(sorted(missing)))
    if destination.is_symlink() or destination.resolve().parent != root:
        raise ValueError('Protocol directory must be inside the project')
    if destination.exists():
        for name, data in selected.items():
            target = destination / name
            if target.is_symlink() or (target.exists() and target.read_bytes() != data):
                raise ValueError('Existing protocol file differs: ' + name)
        if not all((destination / name).is_file() for name in FILES):
            raise ValueError('Existing package is incomplete; move it aside before installing')
    else:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='linq-install-', dir=root) as temporary:
            stage = Path(temporary) / 'tpu_pycloud_format'
            stage.mkdir()
            for name, data in selected.items():
                (stage / name).write_bytes(data)
            shutil.move(str(stage), str(destination))
    return {name: hashlib.sha256(data).hexdigest() for name, data in selected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='tpu_pycloud_format.zip or linq-integration-kit.zip')
    args = parser.parse_args()
    install(args.archive)
    print('Organizer protocol installed: tpu_pycloud_format/ (excluded from Git).')
    print('Next: python -m pip install -r requirements-linq.txt')


if __name__ == '__main__':
    main()
