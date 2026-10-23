"""Local, interactive bootstrap. No default passwords or web bootstrap endpoint."""
import getpass
import sys
import uuid
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import auth,storage as db


def main():
    db.init();auth.init()
    login=auth.normalize(input('Platform admin login: '))
    name=input('Display name: ').strip() or 'Администратор платформы'
    password=getpass.getpass('Password (12+ characters): ')
    if password!=getpass.getpass('Repeat password: '):raise SystemExit('Passwords differ')
    encoded=auth.password_hash(password)
    with db.connection() as c:
        c.execute('BEGIN IMMEDIATE')
        if c.execute('SELECT 1 FROM users WHERE platform_admin=1 AND active=1').fetchone():raise SystemExit('Platform administrator already exists; bootstrap refused')
        key=uuid.uuid4().hex
        c.execute("INSERT INTO users(id,login,name,password_hash,platform_admin,role) VALUES(?,?,?,?,1,'enterprise_admin')",(key,login,name,encoded))
        auth.audit({'id':key,'name':name},'platform.bootstrap',key,conn=c)
    print('Created. Open http://127.0.0.1:8765/ and sign in.')


if __name__=='__main__':main()
