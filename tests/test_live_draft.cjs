const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'../app/static/live.js'),'utf8');
const checkHelpers=fs.readFileSync(require('node:path').join(__dirname,'../app/static/app.js'),'utf8').split('\n').filter(line=>line.startsWith('const isChecked=')||line.startsWith('function setCheck(')).join('\n');
const handler=source.split('\n').find(line=>line.startsWith("$('live-new').onclick="));
const wall=require('../app/static/camera-wall.js');
function setup(sessions){
 const nodes=new Map();function $(id){if(!nodes.has(id))nodes.set(id,{textContent:'old telemetry',className:'pill red',hidden:false,disabled:true,value:'old',width:640,height:480,classList:{remove(){}},removeAttribute(){},querySelectorAll(){return []},getContext(){return {clearRect(){}}}});return nodes.get(id)}
 const L={sessions,current:{id:'old'},events:[{id:'event'}],points:[1],zones:[1],image:{},currentAlertEventId:'old'};const messages=[];let renders=0;
 let genericRenders=0;
 vm.runInNewContext(checkHelpers+'\n'+handler,{L,$,can:()=>true,toast:message=>messages.push(message),labels:{no_helmet:'Без каски'},timestamp:String,UiFormat:{clock:String},openLiveSource(){},openEvent(){},renderLiveAlerts(){},CameraWall:{renderEvents(...args){renders++;wall.renderEvents(...args)}},renderEvents(){genericRenders++;$('live-events').innerHTML='В этой записи не подтверждено…'}});return {L,$,messages,get renders(){return renders},get genericRenders(){return genericRenders}};
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

test('drafting camera B preserves camera A alerts and the common-feed cache without invoking the recording renderer',()=>{
 const running={id:'A',camera:'Camera A',state:'running'},s=setup([running]);
 const event={id:'alert-A',job_id:'A',camera:'Camera A',kind:'no_helmet',track_id:2,start_s:1,alert_s:3,created:'utc'};
 const events=[event];s.L.events=events;
 const queue=require('../app/static/live-alert-queue.js').create();queue.ingest([]);queue.ingest([event]);s.L.alertQueue=queue;
 wall.renderEvents(s.$('live-events'),s.L.sessions,events,{labels:{no_helmet:'Без каски'},time:String,clock:String});
 const original=s.$('live-events').innerHTML,signature=s.$('live-events')._wallEventHTML;
 s.$('live-new').onclick();
 assert.equal(s.L.current,null);assert.equal(s.L.drafting,true);assert.equal(s.L.events,events);assert.equal(s.L.sessions[0],running);
 assert.equal(s.genericRenders,0);assert.equal(s.$('live-events').innerHTML,original);assert.equal(s.$('live-events')._wallEventHTML,signature);
 assert.match(original,/Camera A · человек #2/);assert.doesNotMatch(original,/В этой записи/);assert.equal(queue.snapshot().active.event.id,'alert-A');
 // An unchanged subsequent poll may legitimately skip the cached render;
 // the still-correct feed must remain visible while B's form is configured.
 wall.renderEvents(s.$('live-events'),s.L.sessions,events,{labels:{no_helmet:'Без каски'},time:String,clock:String});
 assert.equal(s.$('live-events').innerHTML,original);
});
test('clearing a template in a new draft resets defaults but cannot rewrite a running camera',()=>{
 const line=source.split('\n').find(row=>row.startsWith("$('live-template').onchange=")),nodes=new Map();
 const $=id=>{if(!nodes.has(id))nodes.set(id,{value:'',checked:true});return nodes.get(id)};
 const running={id:'A',state:'running',config:{fall_detection:true}},L={drafting:true,current:null,zones:[{}],points:[1]},S={cameras:[],defaults:{fall_detection:true}};
 const context={L,S,$,ScenarioCatalog:require('../app/static/scenario-catalog.js'),ZoneNavigation:{nextName:()=>''},paintLiveZones(){},toast(){},structuredClone};
 vm.runInNewContext(checkHelpers+'\n'+line,context);$('live-template').onchange();assert.equal($('live-fall').checked,false);assert.equal($('live-motion').checked,false);assert.equal(L.zones.length,0);
 L.drafting=false;L.current=running;$('live-fall').checked=true;$('live-template').onchange();assert.equal($('live-fall').checked,true);assert.equal(running.config.fall_detection,true);
});
test('new camera acknowledges only actually active streams and preserves the session limit',()=>{
 const one=setup([{state:'running'},{state:'stopped'}]);one.$('live-new').onclick();assert.match(one.messages[0],/Другой эфир продолжает/);
 const two=setup([{state:'running'},{state:'starting'}]);two.$('live-new').onclick();assert.equal(two.L.current.id,'old');assert.equal(two.renders,0);assert.match(two.messages[0],/два эфира/);
});
