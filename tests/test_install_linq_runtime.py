import zipfile

import pytest

from scripts.install_linq_runtime import FILES, install


def make_zip(path, prefix='kit/tpu_pycloud_format/', omit=None, extra=None):
    with zipfile.ZipFile(path, 'w') as archive:
        for name in FILES:
            if name != omit:
                archive.writestr(prefix + name, 'original-' + name)
        archive.writestr('kit/unused.py', 'do not install')
        if extra:
            archive.writestr(*extra)
    return path


@pytest.mark.parametrize('prefix', ['', 'tpu_pycloud_format/', 'kit/tpu_pycloud_format/'])
def test_package_or_kit_only_installs_protocol_files_and_is_idempotent(tmp_path, prefix):
    archive = make_zip(tmp_path / 'kit.zip', prefix)
    root = tmp_path / 'clone'
    before = install(archive, root)
    assert set(p.name for p in (root / 'tpu_pycloud_format').iterdir()) == set(FILES)
    assert install(archive, root) == before
    assert not (root / 'unused.py').exists()


def test_missing_or_conflicting_archive_does_not_create_partial_package(tmp_path):
    archive = make_zip(tmp_path / 'kit.zip', omit='_io.py')
    root = tmp_path / 'clone'
    with pytest.raises(ValueError, match='Missing'):
        install(archive, root)
    assert not root.exists()
    archive = make_zip(tmp_path / 'kit2.zip', extra=('tpu_pycloud_format/_io.py', 'different'))
    with pytest.raises(ValueError, match='Conflicting'):
        install(archive, root)
    assert not root.exists()


def test_changed_local_package_is_never_overwritten_and_escape_is_rejected(tmp_path):
    archive = make_zip(tmp_path / 'kit.zip')
    root = tmp_path / 'clone'
    install(archive, root)
    target = root / 'tpu_pycloud_format/_io.py'
    target.write_text('local changes')
    with pytest.raises(ValueError, match='differs'):
        install(archive, root)
    assert target.read_text() == 'local changes'
    archive = make_zip(tmp_path / 'bad.zip', prefix='../tpu_pycloud_format/')
    with pytest.raises(ValueError, match='Unsafe'):
        install(archive, tmp_path / 'elsewhere')
