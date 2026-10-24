const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const format=require('../app/static/ux-status.js');

test('stored UTC is formatted in the requested local time zone, with missing values explicit',()=>{
  const result=format.dateTime('2026-10-23T19:11:22Z',{timeZone:'Europe/Moscow',locale:'ru-RU'});
  assert.match(result,/23\.10\.2026/);assert.match(result,/22:11:22/);assert.doesNotMatch(result,/T19:11/);
  assert.equal(format.dateTime(null),'Время не указано');
  assert.match(format.time('2026-10-23T19:11:22Z'),/datetime="2026-10-23T19:11:22\.000Z"/);
});

test('audit details show employee, role and camera scope instead of raw JSON or opaque IDs',()=>{
  const text=format.auditDetails({action:'member.updated',target:'3cf5d127dffc4da086948fca87e26232',
    target_context:{type:'member',name:'Анна <script>alert(1)</script>'},details:'{"role":"worker","cameras":null,"active":false}'},
    [{key:'worker',name:'Работник / водитель техники'}]);
  assert.match(text,/Работник \/ водитель техники/);assert.match(text,/Все камеры предприятия/);assert.match(text,/Отключён/);
  assert.match(text,/&lt;script&gt;/);assert.doesNotMatch(text,/<script>|3cf5d127|"role"|"cameras"/);
  assert.match(format.auditDetails({details:'not valid json'}),/Действие сохранено/);
});

function browser(fetch){
  const elements={device:{textContent:'',attrs:{},setAttribute(k,v){this.attrs[k]=v},after(el){elements[el.id]=el}}};
  let timer;
  const events=[];
  const document={documentElement:{lang:'ru'},getElementById:id=>elements[id]||null,
    createElement:()=>({hidden:false}),addEventListener(){},querySelectorAll:()=>[],dispatchEvent:event=>events.push(event)};
  const window={document,location:{href:'http://localhost/',origin:'http://localhost'},fetch,
    CustomEvent:class{constructor(type,options){this.type=type;this.detail=options.detail}}};
  const context={window,URL,AbortController,setTimeout:fn=>(timer=fn,1),clearTimeout(){},Intl,Date,console};
  vm.runInNewContext(fs.readFileSync(require.resolve('../app/static/ux-status.js'),'utf8'),context);
  return {window,elements,events,timeout:()=>timer()};
}

test('platform cabinet replaces the hardware placeholder without starting a probe',()=>{
  let calls=0;const b=browser(async()=>{calls++;throw Error('should not fetch')});
  b.window.UiStatus.context({platform_admin:true},false);
  assert.match(b.elements.device.textContent,/выберите предприятие/);assert.equal(calls,0);
  assert.equal(b.elements.device.attrs['aria-busy'],'false');
});

test('a stalled health probe ends with a visible retry, rather than an infinite checking label',async()=>{
  const b=browser((url,options)=>new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(Error('aborted')))));
  b.window.UiStatus.context({platform_admin:false},true);
  const pending=b.window.fetch('/api/health');b.timeout();await assert.rejects(pending,/aborted/);
  assert.match(b.elements.device.textContent,/Не удалось/);assert.equal(b.elements.device.attrs['aria-busy'],'false');
  assert.equal(b.elements['device-retry'].hidden,false);assert.equal(b.elements['device-retry'].textContent,'Повторить проверку');
});

test('a forbidden health endpoint ends in an access message',async()=>{
  const b=browser(async()=>({ok:false,status:403}));b.window.UiStatus.context({},true);
  await b.window.fetch('/api/health');assert.match(b.elements.device.textContent,/для этой роли/);
  assert.equal(b.elements.device.attrs['aria-busy'],'false');
});

test('successful hardware response shows the actual device and finishes the busy state',async()=>{
  const response={ok:true,status:200,clone:()=>({json:async()=>({device:'RTX 4060',model_ready:true})})};
  const b=browser(async()=>response);b.window.UiStatus.context({},true);
  assert.equal(await b.window.fetch('/api/health'),response);
  assert.match(b.elements.device.textContent,/RTX 4060/);assert.doesNotMatch(b.elements.device.textContent,/Проверяем/);
  assert.equal(b.elements.device.attrs['aria-busy'],'false');
});

test('retry after a timeout publishes restored health for workspace bootstrap recovery',async()=>{
  let calls=0;
  const b=browser((url,options)=>++calls===1?
    new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(Error('aborted')))):
    Promise.resolve({ok:true,status:200,clone:()=>({json:async()=>({device:'CPU',model_ready:true})})}));
  b.window.UiStatus.context({},true);
  const first=b.window.fetch('/api/health');b.timeout();await assert.rejects(first);
  await b.elements['device-retry'].onclick();
  assert.equal(b.elements['device-retry'].hidden,true);
  assert.equal(b.events[0].type,'linq:healthready');assert.equal(b.events[0].detail.device,'CPU');
});
