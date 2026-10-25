const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'../app/static/live.js'),'utf8');
const handler=source.split('\n').find(line=>line.startsWith("$('live-new').onclick="));
function setup(sessions){
 const nodes=new Map();function $(id){if(!nodes.has(id))nodes.set(id,{textContent:'old telemetry',className:'pill red',hidden:false,disabled:true,value:'old',width:640,height:480,classList:{remove(){}},removeAttribute(){},getContext(){return {clearRect(){}}}});return nodes.get(id)}
 const L={sessions,current:{id:'old'},events:[{id:'event'}],points:[1],zones:[1],image:{},currentAlertEventId:'old'};const messages=[];let renders=0;
 vm.runInNewContext(handler,{L,$,can:()=>true,toast:message=>messages.push(message),renderEvents(){renders++}});return {L,$,messages,get renders(){return renders}};
}
test('new camera clears old telemetry and never claims a stopped stream is running',()=>{
 const s=setup([{state:'stopped'},{state:'failed'}]);s.$('live-new').onclick();assert.equal(s.L.current,null);assert.equal(s.L.drafting,true);assert.equal(s.L.currentAlertEventId,null);assert.equal(s.$('live-state').className,'pill');assert.equal(s.$('live-model-latency').textContent,'Измерения появятся после запуска');assert.equal(s.$('live-events-total').textContent,'0 событий · 0 клипов в работе');assert.doesNotMatch(s.$('live-frames').textContent,/old/);assert.equal(s.messages[0],'Настройте новый источник и начните эфир.');assert.equal(s.renders,1);
});
test('new camera has no implicit fall or motion analysis and never mutates another session',()=>{
 const running={id:'A',state:'running',config:{fall_detection:true,helmet_detection:false,zones:[{name:'saved'}]}};
 const s=setup([running]);s.$('live-fall').checked=true;s.$('live-motion').checked=true;s.$('live-new').onclick();
 assert.equal(s.$('live-fall').checked,false);assert.equal(s.$('live-motion').checked,false);
 assert.equal(running.config.fall_detection,true);assert.equal(running.config.helmet_detection,false);assert.equal(running.config.zones[0].name,'saved');
});
test('clearing a template in a new draft resets defaults but cannot rewrite a running camera',()=>{
 const line=source.split('\n').find(row=>row.startsWith("$('live-template').onchange=")),nodes=new Map();
 const $=id=>{if(!nodes.has(id))nodes.set(id,{value:'',checked:true});return nodes.get(id)};
 const running={id:'A',state:'running',config:{fall_detection:true}},L={drafting:true,current:null,zones:[{}],points:[1]},S={cameras:[],defaults:{fall_detection:true}};
 const context={L,S,$,ScenarioCatalog:require('../app/static/scenario-catalog.js'),ZoneNavigation:{nextName:()=>''},paintLiveZones(){},toast(){},structuredClone};
 vm.runInNewContext(line,context);$('live-template').onchange();assert.equal($('live-fall').checked,false);assert.equal($('live-motion').checked,false);assert.equal(L.zones.length,0);
 L.drafting=false;L.current=running;$('live-fall').checked=true;$('live-template').onchange();assert.equal($('live-fall').checked,true);assert.equal(running.config.fall_detection,true);
});
test('new camera acknowledges only actually active streams and preserves the session limit',()=>{
 const one=setup([{state:'running'},{state:'stopped'}]);one.$('live-new').onclick();assert.match(one.messages[0],/Другой эфир продолжает/);
 const two=setup([{state:'running'},{state:'starting'}]);two.$('live-new').onclick();assert.equal(two.L.current.id,'old');assert.equal(two.renders,0);assert.match(two.messages[0],/два эфира/);
});
