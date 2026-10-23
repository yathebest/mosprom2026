import pytest
from app import auth,storage as db
from scripts.setup_demo import setup_demo,LOGIN,PASSWORD,USER_ID


def test_demo_seed_survives_clone_is_idempotent_and_preserves_running_jobs(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3')
    assert setup_demo()
    db.add_job('running','fixture.mp4',{});db.update_job('running',status='running')
    with db.connection() as c:
        user=dict(c.execute('SELECT * FROM users WHERE login=?',(LOGIN,)).fetchone())
        assert user['platform_admin']==1 and auth.verify(PASSWORD,user['password_hash'])
        c.execute("UPDATE users SET password_hash=? WHERE id=?",(auth.password_hash('ChangedByUser123!'),USER_ID))
    assert not setup_demo()
    assert db.job('running')['status']=='running'
    with db.connection() as c:
        user=c.execute('SELECT * FROM users WHERE login=?',(LOGIN,)).fetchone()
        assert auth.verify('ChangedByUser123!',user['password_hash'])
        assert c.execute('SELECT count(*) FROM users').fetchone()[0]==1


def test_demo_seed_refuses_existing_administrator_without_privilege_changes(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'events.sqlite3')
    db.init();auth.init()
    with db.connection() as c:c.execute("INSERT INTO users(id,login,name,password_hash,platform_admin,role) VALUES('custom','custom','Custom administrator',?,1,'enterprise_admin')",(auth.password_hash('PrivatePassword123!'),))
    with pytest.raises(ValueError,match='already exists'):setup_demo()
    with db.connection() as c:assert c.execute('SELECT count(*) FROM users WHERE login=?',(LOGIN,)).fetchone()[0]==0
    assert setup_demo(allow_existing=True)
    with db.connection() as c:
        assert c.execute('SELECT count(*) FROM users WHERE platform_admin=1').fetchone()[0]==2
        assert auth.verify('PrivatePassword123!',c.execute("SELECT password_hash FROM users WHERE id='custom'").fetchone()[0])
