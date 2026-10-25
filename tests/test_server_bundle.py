"""Release inventory and extracted-file verification; no weights or inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def module(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def fake_source(tmp_path,monkeypatch):
    builder=module('build_server_bundle');monkeypatch.setattr(builder,'ROOT',tmp_path)
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'app').mkdir();(tmp_path/'app/tracked.py').write_text('tracked')
    (tmp_path/'.gitignore').write_text('app/ignored.py\napp/.env\n')
    subprocess.run(['git','add','app/tracked.py','.gitignore'],cwd=tmp_path,check=True)
    (tmp_path/'app/new_group.py').write_text('new helper')
    (tmp_path/'app/ignored.py').write_text('not shipped')
    (tmp_path/'app/.env').write_text('not shipped')
    return builder


def test_release_includes_current_untracked_app_but_no_ignored_local_files(tmp_path,monkeypatch):
    builder=fake_source(tmp_path,monkeypatch)
    assert {name for name,_ in builder.sources()}=={'app/tracked.py','app/new_group.py'}


def test_release_rejects_nonignored_private_files_and_unresolved_weight_pointers(tmp_path,monkeypatch):
    builder=fake_source(tmp_path,monkeypatch)
    private=tmp_path/'app/credentials.json';private.write_text('{}')
    with pytest.raises(ValueError,match='Private/local'):builder.sources()
    private.unlink()
    models=tmp_path/'data/models';models.mkdir(parents=True)
    (models/'model.pt').write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError,match='LFS pointer'):builder.sources()


def test_real_release_plan_has_all_runtime_helpers_policy_excel_template_and_quality_dependencies():
    builder=module('build_server_bundle');names={name for name,_ in builder.sources()}
    assert {'app/clip_groups.py','app/clip_metadata.py','app/pattern_policy.py','app/pattern-policy.json',
        'app/static/clip-archive.js','app/static/pattern-policy.js','app/static/pattern-ui.js',
        'app/templates/analytics.xlsx','docs/quality_report.json','docs/coverage.json',
        'scripts/server_preflight.py','scripts/verify_server_bundle.py','scripts/backfill_clip_metadata.py',
        'scripts/coalesce_legacy_clips.py','requirements-server.txt'}<=names
    assert not any(name.startswith(('data/runs/','data/cache/','data/mail-preview/')) or '.sqlite3' in name for name in names)


def test_extracted_verifier_detects_same_size_tampering_and_path_escape(tmp_path):
    verifier=module('verify_server_bundle');(tmp_path/'runtime.py').write_bytes(b'valid')
    row={'path':'runtime.py','bytes':5,'sha256':hashlib.sha256(b'valid').hexdigest()}
    manifest={'files':[row],'model_files':0};(tmp_path/'SERVER_MANIFEST.json').write_text(json.dumps(manifest))
    assert verifier.verify(tmp_path)[1]==[]
    (tmp_path/'runtime.py').write_bytes(b'other')
    assert 'SHA256 mismatch' in verifier.verify(tmp_path)[1][0]
    manifest['files']=[{**row,'path':'../outside.py'}]
    (tmp_path/'SERVER_MANIFEST.json').write_text(json.dumps(manifest))
    assert 'outside package' in verifier.verify(tmp_path)[1][0]
