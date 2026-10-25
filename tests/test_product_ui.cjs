const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const access=require('../app/static/workspace-access.js');
const catalog=require('../app/static/scenario-catalog.js');
const format=require('../app/static/ux-status.js');

test('source video timecodes stay separate from readable durations and Moscow midnight',()=>{
 assert.equal(format.duration(135),'2 мин 15 с');
 assert.equal(format.duration(3605),'1 ч 5 с');
 assert.equal(format.duration(0),'0 с');assert.equal(format.duration(null),'—');
 assert.equal(format.dateKey('2026-10-24T21:15:00Z'),'2026-10-25');
 assert.match(format.dateTime('2026-10-24T21:15:00Z'),/25\.10\.2026.*00:15:00.*МСК.*UTC\+3/);
 assert.match(format.clock('2026-10-24T21:15:00Z'),/00:15:00.*МСК/);
 assert.equal(format.inputDateTime('2026-10-24T21:15:00Z'),'2026-10-25T00:15');
 assert.equal(format.fromInputDateTime('2026-10-25T00:15'),'2026-10-24T21:15:00.000Z');
 assert.equal(format.fromInputDateTime('not a date'),null);
 const incident=require('../app/static/incident-ui.js');
 const text=incident.episodeMarkup({start_s:10,payload:{episode_state:'closed',end_s:145,duration_s:135,closed_at_s:146}},v=>`TC${v}`);
 assert.match(text,/TC10/);assert.match(text,/TC145/);assert.match(text,/2 мин 15 с/);
});
test('policy seconds avoid floating-point noise without losing known fractional precision, in both locales',()=>{
 const ui=require('../app/static/incident-ui.js'),saved=global.UiFormat;
 try{
  global.UiFormat={locale:()=> 'ru-RU'};
  const ru=ui.policyMarkup({window_s:1.9179996000020765,nominal_window_s:1.5,ratio:.75,min_samples:3});
  assert.match(ru,/<b>1,92<\/b>/);assert.match(ru,/<b>1,5<\/b>/);assert.doesNotMatch(ru,/917999|1\.5|1,50/);
  global.UiFormat={locale:()=> 'en-GB'};
  const en=ui.policyMarkup({window_s:1.9179996000020765,nominal_window_s:1.5});
  assert.match(en,/<b>1\.92<\/b>/);assert.match(en,/<b>1\.5<\/b>/);
 }finally{global.UiFormat=saved}
 const css=fs.readFileSync(require.resolve('../app/static/incidents.css'),'utf8');
 assert.match(css,/\.evidence-data \.episode-times\{[^}]*grid-template-columns:minmax\(0,1fr\)/);
});
test('primary catalog excludes unvalidated research while retaining both fall-related event types',()=>{
 assert.deepEqual(catalog.coreKinds,['no_helmet','danger_zone','possible_fall','person_down']);
 for(const kind of ['no_vest','possible_phone_use','possible_smoking','fire','smoke','fast_motion','under_load'])assert.equal(catalog.isPrimary(kind),false);
});
test('technical camera metrics keep their original IDs inside a closed disclosure, while event total stays visible',()=>{
 const html=fs.readFileSync(require.resolve('../app/static/index-v2.html'),'utf8');
 const details=html.match(/<details class="camera-diagnostics"[^>]*>[\s\S]*?<\/details>/)?.[0];
 assert.ok(details);assert.doesNotMatch(details,/^<details[^>]*\bopen\b/);
 for(const id of ['live-frames','live-model-latency']){assert.match(details,new RegExp(`id="${id}"`));assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1)}
 assert.doesNotMatch(details,/id="live-events-total"/);assert.match(html,/<span id="live-events-total">0 событий<\/span>/);
});
test('new-camera defaults cannot inherit fall or experimental choices and do not mutate camera A',()=>{
 const old={fall_detection:true,helmet_detection:false,phone_detection:true,vest_detection:true,fast_motion:true,zones:[{name:'A'}],ppe_hold:1.5};
 const draft=catalog.newCameraConfig(old);
 assert.equal(draft.helmet_detection,true);assert.equal(draft.fall_detection,false);assert.equal(draft.fast_motion,false);assert.equal(draft.phone_detection,false);assert.equal(draft.vest_detection,false);assert.deepEqual(draft.zones,[]);assert.equal(draft.ppe_hold,1.5);
 assert.equal(old.fall_detection,true);assert.equal(old.phone_detection,true);assert.deepEqual(old.zones,[{name:'A'}]);
});
test('saved experimental selections are retained and disclosed, while a fresh camera defaults to helmet only',()=>{
 const source=fs.readFileSync(require.resolve('../app/static/scenarios-ui.js'),'utf8');
 const code=source.slice(source.indexOf('function setScenarioValues('),source.indexOf('const scenarioParams='));
 const nodes=new Map();const element=id=>{if(!nodes.has(id))nodes.set(id,{checked:false,value:'',dataset:{},open:false});return nodes.get(id)};
 const context={$:element,scenarioOptions:[['helmet'],['phone'],['vest']],scenarioParams:[['ppe_hold','',1.5]],updateSmokeExplanation(){},setConfirmationValues(){}};
 vm.runInNewContext(code,context);
 context.setScenarioValues('extra',{});
 assert.equal(element('extra-helmet').checked,true);assert.equal(element('extra-phone').checked,false);assert.equal(element('extra-experimental').open,false);
 context.setScenarioValues('extra',{helmet_detection:false,phone_detection:true,ppe_hold:2});
 assert.equal(element('extra-helmet').checked,false);assert.equal(element('extra-phone').checked,true);assert.equal(element('extra-experimental').open,true);assert.equal(element('extra-ppe_hold').value,2);
 // A live status poll must not forcibly reopen details the operator collapsed.
 element('extra-experimental').open=false;context.setScenarioValues('extra',{helmet_detection:false,phone_detection:true});
 assert.equal(element('extra-experimental').open,false);
 context.setScenarioValues('extra',{fast_motion:true});assert.equal(element('extra-experimental').open,true);
});
test('new-camera controls unlock immediately after leaving active A, without changing A configuration',()=>{
 const source=fs.readFileSync(require.resolve('../app/static/scenarios-ui.js'),'utf8');
 const status=source.split('\n').find(line=>line.startsWith('renderLiveStatus=function'));
 const draft=source.split('\n').find(line=>line.startsWith("$('live-new').onclick="));
 const button={},controls=[{disabled:false,checked:true},{disabled:false,value:'danger_zone'}],configA={helmet_detection:true,fall_detection:false},sessionA={id:'A',state:'running',config:configA};
 const values=[];
 const context={L:{current:sessionA,events:[],drafting:false},S:{jobs:[],defaults:{helmet_detection:false,fall_detection:true}},$:()=>button,document:{querySelectorAll:()=>controls},ScenarioCatalog:catalog,
  baseLiveStatus(){},renderPpeAssessments(){},setScenarioValues(_prefix,c){values.push(c)},baseLiveNew(){context.L.current=null;context.L.drafting=true}};
 vm.runInNewContext(status,context);context.renderLiveStatus();assert.equal(controls.every(el=>el.disabled),true);
 vm.runInNewContext(draft,context);button.onclick();
 assert.equal(controls.every(el=>!el.disabled),true);assert.equal(values.at(-1).helmet_detection,true);assert.equal(values.at(-1).fall_detection,false);
 assert.deepEqual(configA,{helmet_detection:true,fall_detection:false});assert.equal(sessionA.state,'running');
});

function shell(user){
 class Element{
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.hidden=false;this.attrs={};this.classes=new Set();this.classList={toggle:(k,on)=>{if(on===undefined)on=!this.classes.has(k);on?this.classes.add(k):this.classes.delete(k)},add:k=>this.classes.add(k),contains:k=>this.classes.has(k)};}
  append(...nodes){for(const n of nodes){if(n.parent)n.parent.children=n.parent.children.filter(x=>x!==n);n.parent=this;this.children.push(n)}}
  insertBefore(n,ref){this.append(n);this.children.splice(this.children.indexOf(n),1);this.children.splice(this.children.indexOf(ref),0,n)}
  replaceChildren(...nodes){this.children=[];this.append(...nodes)}
  after(n){const i=this.parent.children.indexOf(this);n.parent=this.parent;this.parent.children.splice(i+1,0,n)}
  setAttribute(k,v){this.attrs[k]=v}
  set innerHTML(value){this.html=value;if(value.includes('product-tools-content'))this.append(new Element('div'))}
  querySelectorAll(selector){const all=this.children.flatMap(x=>[x,...x.querySelectorAll('*')]);if(selector==='*')return all;if(selector==='div')return all.filter(x=>x.tag==='div');const target=selector.match(/data-page="([^"]+)"/);return all.filter(x=>x.dataset.page&&(!target||x.dataset.page===target[1])&&(!selector.startsWith('button')||x.tag==='button'))}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null}
  addEventListener(){}
 }
 const body=new Element('body'),aside=new Element('aside'),nav=new Element('nav'),main=new Element('main'),header=new Element('header');body.append(aside,main);aside.append(nav);main.append(header);
 const buttons={};for(const page of ['live','overview','journal','clips','triage','analytics','quality','help','platform','team','audit','account','safety']){const b=new Element('button');b.dataset.page=page;buttons[page]=b;nav.append(b)}
 const badge=new Element('span');badge.id='new-badge';buttons.journal.append(badge);
 const all=()=>[body,...body.querySelectorAll('*')];
 const context={document:{querySelector:s=>s==='aside nav'?nav:s==='main header'?header:null,createElement:tag=>new Element(tag),addEventListener(){},body},
  $:id=>all().find(x=>x.id===id),WorkspaceAccess:access,A:{user},S:{page:access.home(user)},innerWidth:458,window:null};context.window=context;
 context.applyPermissions=()=>{for(const b of Object.values(buttons))b.hidden=!access.visible(context.A.user,b.dataset.page)};
 context.showPage=page=>{if(access.canOpen(context.A.user,page))context.S.page=page};
 context.applyPermissions();vm.runInNewContext(fs.readFileSync(require.resolve('../app/static/product-ui.js'),'utf8'),context);
 return {context,buttons,nav,eventButton:context.$('product-events-nav'),eventNav:context.$('event-navigation'),tools:context.$('workspace-tools'),badge};
}
const operator={role:'operator',organization_id:'a',organization:{status:'active'},permissions:['live.read','events.read','events.review','evidence.read','analytics.read']};
test('operator event hub keeps all original event controls and permissions, and highlights the whole family',()=>{
 const s=shell(operator);assert.equal(s.eventButton.hidden,false);assert.equal(s.eventNav.hidden,true);
 assert.equal(s.eventButton.attrs['aria-label'],'События');
 assert.equal(s.badge.parent,s.eventButton);s.eventButton.onclick();assert.equal(s.context.S.page,'journal');assert.equal(s.eventNav.hidden,false);
 s.context.showPage('clips');assert.equal(s.eventButton.classList.contains('on'),true);assert.equal(s.buttons.clips.classList.contains('on'),true);
 assert.equal(s.buttons.team.hidden,true);assert.equal(s.buttons.platform.hidden,true);
 s.context.showPage('platform');assert.equal(s.context.S.page,'clips');
});
test('cabinet and restricted roles retain their own tools without seeing enterprise camera/event destinations',()=>{
 const s=shell({platform_admin:true,permissions:[]});assert.equal(s.eventButton.hidden,true);assert.equal(s.buttons.live.hidden,true);assert.equal(s.buttons.platform.hidden,false);assert.equal(s.tools.hidden,false);
 const evidence=shell({...operator,permissions:['evidence.read']});evidence.eventButton.onclick();assert.equal(evidence.context.S.page,'clips');assert.equal(evidence.buttons.journal.hidden,true);
});
