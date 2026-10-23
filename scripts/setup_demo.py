"""Explicit local demonstration seed. Never imported by the application server."""
import sys
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import auth,storage as db

LOGIN='demo-admin'
PASSWORD='LinQ-Demo-2026!'
USER_ID='local-demo-platform-admin'


def setup_demo(allow_existing=False):
    db.init(mark_interrupted=False);auth.init()
    with db.connection() as c:
        c.execute('BEGIN IMMEDIATE')
        row=c.execute('SELECT * FROM users WHERE login=? OR id=?',(LOGIN,USER_ID)).fetchone()
        if row:
            if row['id']!=USER_ID or row['login']!=LOGIN or not row['platform_admin'] or not row['active']:
                raise ValueError('Demo identity conflicts with an existing account; no accounts changed')
            return False
        if not allow_existing and c.execute('SELECT 1 FROM users WHERE platform_admin=1 AND active=1').fetchone():
            raise ValueError('A platform administrator already exists. Demo seeding is only for a new local installation')
        c.execute("INSERT OR IGNORE INTO organizations(id,name,status,note) VALUES('demo','Демонстрационное предприятие','active','Локальный демонстрационный стенд')")
        c.execute("INSERT INTO users(id,login,name,password_hash,platform_admin,role) VALUES(?,?,?,?,1,'enterprise_admin')",(USER_ID,LOGIN,'Демо · Администратор платформы',auth.password_hash(PASSWORD)))
        auth.audit({'id':USER_ID,'name':'Демо · Администратор платформы'},'platform.demo_created',USER_ID,conn=c)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alongside-existing',action='store_true',help='Explicitly add the demo administrator alongside an existing local administrator')
    args=parser.parse_args()
    try:created=setup_demo(allow_existing=args.alongside_existing)
    except ValueError as exc:raise SystemExit(str(exc)) from None
    print('Local demo administrator created.' if created else 'Demo account already exists; password and sessions preserved.')
    print('Login: '+LOGIN)
    if created:print('Password: '+PASSWORD)
    print('Credentials and clone instructions: DEMO-ACCESS.md')


if __name__=='__main__':main()
