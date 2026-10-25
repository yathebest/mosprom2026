"""Prepare an isolated camera-wall UI fixture; optional server, no NN auto-start.

Creates new synthetic users/DB. Production DB is never copied or modified.
Model/dataset files are linked read-only by convention for real UI inference.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]


def link_tree(source,target):
    if not source.exists():return
    if source.is_file():
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            try:os.link(source,target)
            except OSError:
                import shutil
                shutil.copy2(source,target)
        return
    target.mkdir(parents=True,exist_ok=True)
    if os.name=='nt' and source.is_dir():
        # A junction avoids enumerating/copying many GB of public samples.
        import _winapi
        if not any(target.iterdir()):
            target.rmdir()
            try:_winapi.CreateJunction(str(source),str(target));return
            except OSError:pass
            target.mkdir(exist_ok=True)
    if source.is_dir():
        for item in source.iterdir():link_tree(item,target/item.name)


def prepare(directory):
    directory=Path(directory).resolve();production=(ROOT/'data').resolve()
    if directory==production:raise ValueError('Staging directory must differ from production data')
    access=directory/'access.json';database=directory/'events.sqlite3'
    if access.exists() and database.exists():return json.loads(access.read_text(encoding='utf-8'))
    if database.exists():raise ValueError('Existing staging DB has no fixture access file; refusing to reset it')
    directory.mkdir(parents=True,exist_ok=True)
    link_tree(production/'models',directory/'models')
    # Only public sample subtrees; no production users/events/run evidence.
    for name in ['urfd','scenario-fixtures']:link_tree(production/'datasets'/name,directory/'datasets'/name)
    sys.path.insert(0,str(ROOT))
    from app import storage as db
    db.DATA=directory;db.DB=database
    from app import auth
    db.init();auth.init()
    admin_password='QaAdmin-'+secrets.token_urlsafe(15)
    operator_password='QaOperator-'+secrets.token_urlsafe(15)
    cameras=['QA Камера A','QA Камера B'];organization='camera-wall-qa'
    with db.connection() as connection:
        connection.execute('INSERT INTO organizations(id,name,status,note) VALUES(?,?,?,?)',
            (organization,'QA · Стена камер','active','Isolated synthetic UI fixture; no production data'))
        for key,login,name,role,password,scope in [
            ('qa-admin','qa-wall-admin','QA администратор','enterprise_admin',admin_password,None),
            ('qa-operator','qa-wall-operator','QA оператор','operator',operator_password,json.dumps(cameras,ensure_ascii=False))]:
            connection.execute('INSERT INTO users(id,login,name,password_hash,organization_id,role,camera_scope,must_change) VALUES(?,?,?,?,?,?,?,0)',
                (key,login,name,auth.password_hash(password),organization,role,scope))
    from app.access import Scope,current_scope
    token=current_scope.set(Scope(organization))
    try:
        from app.core import RunConfig
        for camera in cameras:
            config=RunConfig(camera=camera,fast_motion=True,fall_detection=False).model_dump();db.save_camera(config)
    finally:current_scope.reset(token)
    result={'fixture_only':True,'production_database_copied':False,'organization_id':organization,
            'data_directory':str(directory),'database':str(database),'cameras':cameras,
            'admin':{'login':'qa-wall-admin','password':admin_password},
            'operator':{'login':'qa-wall-operator','password':operator_password,'camera_scope':cameras},
            'sources':['http://127.0.0.1:8766/cam-a.mjpg','http://127.0.0.1:8766/cam-b.mjpg']}
    access.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dir',type=Path,default=ROOT/'data/cache/camera-wall-qa');parser.add_argument('--serve',action='store_true');parser.add_argument('--port',type=int,default=8767)
    args=parser.parse_args();fixture=prepare(args.dir)
    print(json.dumps(fixture,ensure_ascii=False,indent=2),flush=True)
    if args.serve:
        sys.path.insert(0,str(ROOT));from app import storage as db
        db.DATA=Path(fixture['data_directory']);db.DB=Path(fixture['database'])
        import uvicorn
        uvicorn.run('app.server:app',host='127.0.0.1',port=args.port)
