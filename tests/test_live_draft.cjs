const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'../app/static/live.js'),'utf8');
const handler=source.split('\n').find(line=>line.startsWith("$('live-new').onclick="));
function setup(sessions){
 const nodes=new Map();function $(id){if(!nodes.has(id))nodes.set(id,{textContent:'old telemetry',className:'pill red',hidden:false,disabled:true,value:'old',width:640,height:480,classList:{remove(){}},removeAttribute(){},getContext(){return {clearRect(){}}}});return nodes.get(id)}
 const L={sessions,current:{id:'old'},events:[{id:'event'}],points:[1],zones:[1],image:{},currentAlertEventId:'old'};const messages=[];let renders=0;
 vm.runInNewContext(handler,{L,$,toast:message=>messages.push(message),renderEvents(){renders++}});return {L,$,messages,get renders(){return renders}};
}
test('new camera clears old telemetry and never claims a stopped stream is running',()=>{
 const s=setup([{state:'stopped'},{state:'failed'}]);s.$('live-new').onclick();assert.equal(s.L.current,null);assert.equal(s.L.drafting,true);assert.equal(s.L.currentAlertEventId,null);assert.equal(s.$('live-state').className,'pill');assert.equal(s.$('live-model-latency').textContent,'Измерения появятся после запуска');assert.equal(s.$('live-events-total').textContent,'0 событий · 0 клипов в работе');assert.doesNotMatch(s.$('live-frames').textContent,/old/);assert.equal(s.messages[0],'Настройте новый источник и начните эфир.');assert.equal(s.renders,1);
});
test('new camera acknowledges only actually active streams and preserves the session limit',()=>{
 const one=setup([{state:'running'},{state:'stopped'}]);one.$('live-new').onclick();assert.match(one.messages[0],/Другой эфир продолжает/);
 const two=setup([{state:'running'},{state:'starting'}]);two.$('live-new').onclick();assert.equal(two.L.current.id,'old');assert.equal(two.renders,0);assert.match(two.messages[0],/два эфира/);
});
