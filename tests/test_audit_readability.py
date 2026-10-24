import sqlite3
from app.auth import audit_target_context


def connection():
    c=sqlite3.connect(':memory:');c.row_factory=sqlite3.Row
    c.executescript('''
      CREATE TABLE users(id,name,organization_id,platform_admin);
      CREATE TABLE organizations(id,name);
      CREATE TABLE admin_invitations(id,email,organization_id);
      CREATE TABLE jobs(id,organization_id);
      CREATE TABLE events(id,job_id,camera,kind);
      INSERT INTO users VALUES('one','Visible employee','a',0),('two','Other enterprise employee','b',0);
      INSERT INTO organizations VALUES('a','Enterprise A'),('b','Enterprise B');
      INSERT INTO admin_invitations VALUES('invite-a','a@example.test','a'),('invite-b','b@example.test','b');
      INSERT INTO jobs VALUES('ja','a'),('jb','b');
      INSERT INTO events VALUES('ea','ja','Camera A','danger_zone'),('eb','jb','Camera B','fire');
    ''')
    return c


def row(action,target,org='a'):
    return {'action':action,'target':target,'organization_id':org,'actor_name':'Administrator'}


def test_audit_targets_use_names_but_never_resolve_foreign_enterprise_ids():
    with connection() as c:
        assert audit_target_context(c,row('member.updated','one'))=={'type':'member','name':'Visible employee'}
        assert audit_target_context(c,row('member.updated','two'))=={'type':'member','name':None}
        assert audit_target_context(c,row('organization.active','b'))=={'type':'organization','name':None}
        assert audit_target_context(c,row('invitation.sent','invite-b'))=={'type':'invitation','email':None}


def test_audit_event_context_follows_job_enterprise_and_cabinet_has_no_target():
    with connection() as c:
        assert audit_target_context(c,row('event.reviewed','ea'))=={'type':'event','camera':'Camera A','kind':'danger_zone'}
        assert audit_target_context(c,row('event.reviewed','eb'))=={'type':'event'}
        assert audit_target_context(c,row('platform.context','',None))=={}
