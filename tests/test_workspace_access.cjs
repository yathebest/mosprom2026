const {test}=require('node:test');
const assert=require('node:assert/strict');
const access=require('../app/static/workspace-access.js');
const admin={platform_admin:1,role:'enterprise_admin',organization_id:null,organization:null,permissions:[],must_change:0};
const workspace={...admin,organization_id:'factory-a',organization:{id:'factory-a',name:'Factory A',status:'active'},permissions:['live.read','analysis.run','members.manage','audit.read','events.read']};

test('platform administrator in an enterprise opens its live view and cannot open platform management in the same shell',()=>{
  assert.equal(access.home(workspace),'live');
  assert.equal(access.visible(workspace,'platform'),false);
  assert.equal(access.canOpen(workspace,'platform'),false);
  assert.equal(access.visible(workspace,'team'),true);
  assert.equal(access.visible(workspace,'live'),true);
});
test('returning to the cabinet hides enterprise tools and opens enterprise administration',()=>{
  assert.equal(access.home(admin),'platform');
  assert.equal(access.visible(admin,'platform'),true);
  for(const page of ['live','overview','journal','clips','team','analytics','quality'])assert.equal(access.visible(admin,page),false,page);
  assert.equal(access.visible(admin,'audit'),true);
});
test('staff for a selected enterprise can be managed from the cabinet without selecting its workspace',()=>{
  assert.equal(access.canOpen(admin,'team'),false);
  assert.equal(access.canOpen(admin,'team','factory-a'),true);
  assert.equal(access.home(admin),'platform');
  assert.equal(access.canOpen({...workspace,platform_admin:0,permissions:[]},'team','factory-b'),false);
});
test('local roles keep their own workspace start pages',()=>{
  assert.equal(access.home({...workspace,platform_admin:0,role:'worker',permissions:[]}),'safety');
  assert.equal(access.home({...workspace,platform_admin:0,role:'management',permissions:['analytics.read']}),'analytics');
  assert.equal(access.home({...workspace,platform_admin:0,role:'safety_legal',permissions:['events.read']}),'journal');
});
test('suspended enterprises and temporary passwords do not open a workspace or cabinet',()=>{
  const suspended={...workspace,organization:{...workspace.organization,status:'suspended'},permissions:[]};
  assert.equal(access.home(suspended),'account');
  assert.equal(access.visible(suspended,'live'),false);
  assert.equal(access.home({...workspace,must_change:1}),'account');
  assert.equal(access.visible({...workspace,must_change:1},'team'),false);
});
