const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const i18n=require('../app/static/i18n.js');
require('../app/static/i18n-extra.js');

test('all Russian static interface text has an English translation',()=>{
 const html=fs.readFileSync(path.join(__dirname,'../app/static/index-v2.html'),'utf8').replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi,'');
 const values=html.split(/<[^>]*>/g).map(s=>s.replace(/&nbsp;/g,' ').replace(/&gt;/g,'>').replace(/&amp;/g,'&').replace(/\s+/g,' ').trim());
 for(const value of values){
  if(!/[А-Яа-яЁё]/.test(value)||value==='Русский')continue;
  assert.notEqual(i18n.t(value,'en'),value,`Missing static translation: ${value}`);
 }
});

test('counts and current live telemetry are translated without changing values',()=>{
 assert.equal(i18n.t('Шаг 3 из 4','en'),'Step 3 of 4');
 assert.equal(i18n.t('Видео 19.81 кадр/с · люди 1.21 · СИЗ/объекты 0.93','en'),'Video 19.81 frames/s · people 1.21 · PPE/objects 0.93');
 assert.equal(i18n.t('Анализ людей P95: 1926.79 мс · задержка СИЗ 3438.9 мс','en'),'People analysis P95: 1926.79 ms · PPE result age 3438.9 ms');
 assert.equal(i18n.t('Удалить точку 7','en'),'Remove point 7');
});

test('data literals stay intact even when they coincide with an interface label',()=>{
 const cameraNames=new Set(['Нужна проверка','Прямой эфир']);
 assert.equal(i18n.t('Прямой эфир','en',cameraNames),'Прямой эфир');
 assert.equal(i18n.t('Нужна проверка · События','en',cameraNames),'Нужна проверка · Events');
 for(const value of ['Цех №3','Петров Алексей','worker','danger_zone','person_down','operator@example.com','Запись_касса.mp4'])assert.equal(i18n.t(value,'en'),value);
});

test('language choice persists and does not require network translation',()=>{
 const saved=global.localStorage;const values=new Map();
 global.localStorage={getItem:k=>values.get(k),setItem:(k,v)=>values.set(k,v)};
 try{
  i18n.setLanguage('en');assert.equal(values.get('linq.language'),'en');
  delete require.cache[require.resolve('../app/static/i18n.js')];
  const restored=require('../app/static/i18n.js');assert.equal(restored.language(),'en');assert.equal(restored.locale(),'en-GB');
  restored.setLanguage('ru');assert.equal(restored.t('Прямой эфир'),'Прямой эфир');
 }finally{global.localStorage=saved;i18n.setLanguage('ru')}
});

test('DOM localization preserves handlers, names, inputs and restores Russian',()=>{
 const saved={document:global.document,NodeFilter:global.NodeFilter,CustomEvent:global.CustomEvent,S:global.S};
 i18n.setLanguage('en');
 function element(kind,{nav=false,ignore=false}={}){
  return {kind,onclick:null,value:'Прямой эфир',attrs:{},parentElement:null,
   closest(selector){if(selector.includes('[data-i18n-ignore]')&&ignore)return this;if(selector.includes('nav')&&nav)return this;return null},
   matches(selector){return selector==='option'&&kind==='option'},querySelectorAll(){return []},
   getAttribute(key){return this.attrs[key]??null},setAttribute(key,v){this.attrs[key]=v}};
 }
 const button=element('button',{nav:true}),user=element('strong'),ignored=element('span',{ignore:true});
 const handler=()=>42;button.onclick=handler;
 const nodes=[{data:'Прямой эфир',parentElement:button},{data:'Прямой эфир',parentElement:user},{data:'Нужна проверка',parentElement:ignored}];
 const body=element('body');const events=[];
 global.NodeFilter={SHOW_TEXT:4};global.CustomEvent=class{constructor(type,options){this.type=type;this.detail=options.detail}};
 global.S={cameras:[{name:'Прямой эфир'}]};
 global.document={body,documentElement:{lang:'en'},getElementById:()=>null,dispatchEvent:e=>events.push(e),
  createTreeWalker(){let i=-1;return {currentNode:null,nextNode(){this.currentNode=nodes[++i];return Boolean(this.currentNode)}}}};
 try{
  i18n.apply();assert.equal(nodes[0].data,'Live monitoring');assert.equal(nodes[1].data,'Прямой эфир');assert.equal(nodes[2].data,'Нужна проверка');
  assert.equal(button.onclick,handler);assert.equal(button.onclick(),42);assert.equal(button.value,'Прямой эфир');
  i18n.setLanguage('ru');assert.equal(nodes[0].data,'Прямой эфир');assert.equal(global.document.documentElement.lang,'ru');assert.equal(events.at(-1).type,'linq:languagechange');
 }finally{Object.assign(global,saved);i18n.setLanguage('ru')}
});

test('smoke and threshold copy states uncertainty rather than unmeasured accuracy',()=>{
 const ru='Возможный дым: отличие от пыли и пара не подтверждено. Оператор проверяет клип.';
 assert.match(i18n.t(ru,'en'),/unverified.*operator/i);
 assert.match(i18n.t('Ниже порог — больше кандидатов телефона, но больше ложных тревог. Выше — строже отбор, но можно пропустить маленький телефон. Значение 0,30 не означает точность 30%.','en'),/does not mean 30% accuracy/);
});
