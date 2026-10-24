const {test}=require('node:test');
const assert=require('node:assert/strict');
const ui=require('../app/static/incident-ui.js');
const i18n=require('../app/static/i18n.js');require('../app/static/i18n-extra.js');
const open={id:'one',start_s:10,clip_end_s:17,payload:{episode_id:'ep1',episode_state:'open',end_s:null,observed_until_s:82}};
test('excerpt bounds and last observation never imply an episode end',()=>{
 for(const state of ['open','interrupted','unknown']){
  const event={...open,payload:{...open.payload,episode_state:state}};
  const e=ui.episode(event);assert.equal(e.end,null);assert.equal(e.duration,null);assert.equal(e.observedDuration,72);
  assert.match(ui.episodeMarkup(event),/Не подтверждено завершение/);
  assert.equal(ui.closureProofMarkup({...event,payload:{...event.payload,end_image:'unverified.jpg'}}),'');
 }
});
test('confirmed closure reports the full interval and its own proof frame',()=>{
 const e={...open,payload:{...open.payload,episode_state:'closed',end_s:90,closed_at_s:93,duration_s:80,end_image:'end.jpg'}};
 assert.equal(ui.episode(e).duration,80);assert.equal(ui.episode(e).end,90);
 assert.match(ui.closureProofMarkup(e),/end\.jpg/);assert.match(ui.closureProofMarkup(e),/1:33\.0/);
 assert.equal(ui.episode({...e,payload:{...e.payload,end_s:null}}).end,null);
 assert.equal(ui.episode({...e,payload:{...e.payload,end_s:5}}).duration,null);
});
test('closure upserts the existing ID without a recovery activation',()=>{
 const closed={id:'one',payload:{episode_state:'closed',end_s:90,end_image:'end.jpg'}};
 const result=ui.mergeEvents([open],[closed]);assert.equal(result.length,1);assert.equal(result[0].id,'one');
 assert.equal(result[0].payload.episode_id,'ep1');assert.equal(result[0].start_s,10);assert.equal(ui.episode(result[0]).end,90);
 assert.equal(ui.isActivation(result[0]),false);assert.equal(ui.isActivation(open),true);
 assert.equal(ui.isActivation({payload:{transition:'clear'}}),false);
 assert.equal(ui.isActivation({payload:{episode_state:'interrupted'}}),false);
});
test('measured missing PPE cannot masquerade as a confirmed violation',()=>{
 const rows=ui.personRows({},[{track_id:7,ppe:{helmet:{state:'missing',observed_at_s:2}}}]);
 assert.equal(rows[0].patterns[0].state,'unknown');assert.equal(rows[0].patterns[0].observed_state,'violation');
 const html=ui.statesMarkup({},[{track_id:7,ppe:{helmet:{state:'missing'}}}]);assert.doesNotMatch(html,/confirmed-violation/);
});
test('latest unknown stays neutral despite a historical clear state and IDs stay independent',()=>{
 const states={7:{no_helmet:{state:'clear',observed_state:'unknown',confirmed_at_s:10,observed_at_s:11,window_s:2}},8:{no_helmet:{state:'violation',observed_state:'violation',observed_at_s:11,window_s:2}}};
 const html=ui.statesMarkup(states,[],{now:11,labels:{no_helmet:'Без каски'}});
 assert.match(html,/Человек #7/);assert.match(html,/Человек #8/);
 const first=html.match(/<article[^>]*data-person-state-id="7"[\s\S]*?<\/article>/)[0];assert.match(first,/data-current-state="unknown"/);assert.doesNotMatch(first,/class="confirmed-state confirmed-clear"/);
 assert.match(first,/Последнее подтверждённое состояние/);assert.match(first,/Каска обнаружена/);
 assert.match(html,/data-current-state="violation"/);
});
test('stale or predicted measurements cannot display a current clear state',()=>{
 const state={3:{no_vest:{state:'clear',observed_state:'clear',observed_at_s:1,window_s:1}}};
 assert.match(ui.statesMarkup(state,[],{now:20,expandedKeys:new Set(['group:history'])}),/data-current-state="unknown"/);
 assert.match(ui.statesMarkup(state,[{track_id:3,tracking:'predicted',ppe:{}}],{now:1,expandedKeys:new Set(['group:history'])}),/data-current-state="unknown"/);
 state[3].no_vest.observation_fresh=false;assert.match(ui.statesMarkup(state,[],{now:1,expandedKeys:new Set(['group:history'])}),/data-current-state="unknown"/);
});
test('configuration preserves unset windows and independent explicit overrides',()=>{
 assert.deepEqual(ui.confirmationConfig({windows:{no_helmet:'',fire:null}}),{state_confirmation_ratio:.75,state_min_samples:3,state_windows:{}});
 assert.deepEqual(ui.confirmationConfig({ratio:.8,minSamples:4,windows:{no_helmet:'2',no_vest:'3'}}).state_windows,{no_helmet:2,no_vest:3});
 for(const value of [.74,1.01,''])assert.throws(()=>ui.confirmationConfig({ratio:value}));
 for(const value of [2,3.5,31])assert.throws(()=>ui.confirmationConfig({minSamples:value}));
 for(const value of [.1,61,'abc'])assert.throws(()=>ui.confirmationConfig({windows:{no_helmet:value}}));
 assert.throws(()=>ui.confirmationConfig({windows:{invented_pattern:2}}));
});
test('adaptive policy shows effective and nominal time separately from sample minimum',()=>{
 const html=ui.policyMarkup({window_s:4,nominal_window_s:1.5,ratio:.75,min_samples:3,cadence_limited:true});
 assert.match(html,/Фактическое окно/);assert.match(html,/Заданное окно/);assert.match(html,/Доля времени/);assert.match(html,/Минимум свежих наблюдений/);
 assert.doesNotMatch(html,/Доля согласованных замеров/);
 const event={...open,payload:{...open.payload,episode_state:'closed',end_s:20,start_state_confirmation:{window_s:2},state_confirmation:{window_s:3}}};
 assert.match(ui.episodePolicyMarkup(event),/Подтверждение начала/);assert.match(ui.episodePolicyMarkup(event),/Подтверждение завершения/);
});
test('episode and confirmed-state helper text is localizable without touching user zones',()=>{
 const html=ui.episodeMarkup(open)+ui.statesMarkup({7:{'no_helmet:Цех №7':{state:'clear',observed_state:'unknown',window_s:2}}},[],{labels:{no_helmet:'Без каски'},expandedKeys:new Set(['group:history'])})+ui.policyMarkup({window_s:4,nominal_window_s:1.5,ratio:.75,min_samples:3,cadence_limited:true});
 for(const part of html.split(/<[^>]*>/).map(v=>v.trim()))if(/[А-Яа-яЁё]/.test(part)&&part!=='Цех №7')assert.notEqual(i18n.t(part,'en'),part,part);
 assert.match(html,/data-i18n-ignore>Цех №7/);
});

test('ID churn stays compact and hidden violations remain counted with lazy access',()=>{
 const states={},assessments=[];for(let id=1;id<=80;id++)states[id]={no_helmet:{state:id===3?'violation':'unknown',observed_state:'unknown',observed_at_s:1,window_s:2}};
 for(let id=81;id<=87;id++){states[id]={no_helmet:{state:id===87?'violation':'clear',observed_state:id===87?'violation':'clear',observed_at_s:100,window_s:2}};assessments.push({track_id:id,tracking:'measured',ppe:{}})}
 const view=ui.stateView(states,assessments,{now:100});assert.equal(view.primary.length,4);assert.equal(view.primary[0].id,87);assert.equal(view.more.length,3);assert.equal(view.history.length,80);assert.equal(view.violations,2);
 const html=ui.statesMarkup(states,assessments,{now:100});assert.equal((html.match(/data-person-state-id=/g)||[]).length,4);assert.match(html,/Нарушения не сняты у ID/);assert.match(html,/#3, #87/);
 assert.doesNotMatch(html,/data-person-state-id="3"/);const history=ui.stateGroupMarkup(view,'history');assert.match(history,/data-person-state-id="3"/);assert.equal((history.match(/data-person-state-id=/g)||[]).length,80);
 const expanded=ui.statesMarkup(states,assessments,{now:100,expandedKeys:new Set(['group:history','person:3'])});assert.match(expanded,/data-preserve-key="person:3" open/);
});
test('polling time changes alone do not invalidate states; stale boundaries do',()=>{
 const states={7:{no_helmet:{state:'clear',observed_state:'clear',observed_at_s:10,window_s:2}}};
 const key=now=>ui.stateRenderKey(ui.stateView(states,[],{now}),[],{});
 assert.equal(key(10),key(10.5));assert.equal(key(10.5),key(11));assert.notEqual(key(11),key(20));
});
