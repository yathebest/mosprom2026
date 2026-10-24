const {test}=require('node:test');
const assert=require('node:assert/strict');
const zones=require('../app/static/zone-navigation.js');

test('deleting an earlier default zone renumbers defaults but preserves named areas and geometry',()=>{
  const points=[[0,0],[1,0],[1,1]],remaining=[{name:'Опасная зона 2',points,kind:'danger_zone'},{name:'Склад сырья',points,kind:'danger_zone'},{name:'Тестовая зона 7',points}];
  const result=zones.renumber(remaining);
  assert.deepEqual(result.map(z=>z.name),['Опасная зона 1','Склад сырья','Тестовая зона 1']);
  assert.equal(result[0].points,points);assert.equal(remaining[0].name,'Опасная зона 2');
  assert.equal(zones.nextName(result),'Опасная зона 2');
});

test('a point can be removed from the middle without losing other vertices',()=>{
  const points=[[.1,.1],[.5,.2],[.9,.8],[.2,.9]];
  assert.deepEqual(zones.removePoint(points,1),[points[0],points[2],points[3]]);
  assert.equal(points.length,4);
  assert.equal(zones.nearestPoint(points,[.51,.2],500,1000),1);
  assert.equal(zones.nearestPoint(points,[.51,.6],500,1000),-1);
});

test('forward wizard navigation rechecks source, scenarios and required context',()=>{
  const state={camera:'Цех',anyScenario:true,zoneEnabled:true,zones:[{kind:'danger_zone'}],calibrationEnabled:false,groundPoints:0,harnessEnabled:true};
  assert.equal(zones.wizardIssue(0,state),'');assert.equal(zones.wizardIssue(1,state),'');
  assert.match(zones.wizardIssue(2,state),/Выбран сценарий «Работа на высоте без привязи»/);
  assert.equal(zones.wizardIssue(2,{...state,zones:[{kind:'height'}]}),'');
  assert.match(zones.wizardIssue(0,{...state,camera:' '}),/Источник/);
  assert.match(zones.wizardIssue(1,{...state,anyScenario:false}),/Сценарии/);
  assert.match(zones.wizardIssue(2,{...state,harnessEnabled:false,zones:[]}),/Сохраните полигон/);
});
