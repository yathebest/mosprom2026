const {test}=require('node:test');
const assert=require('node:assert/strict');
const ui=require('../app/static/incident-ui.js');
const i18n=require('../app/static/i18n.js');require('../app/static/i18n-extra.js');
const open={id:'one',start_s:10,clip_end_s:17,payload:{episode_id:'ep1',episode_state:'open',end_s:null,observed_until_s:82}};
test('ID7 broader clothing policy remains unknown when only legacy missing-vest evidence exists',()=>{
 const assessment={track_id:7,tracking:'measured',ppe:{vest:{state:'unknown',observed_state:'unknown',confirmed_state:'unknown',fresh:true,observed_at_s:3,reason:'signal_clothing_policy_unverified',policy:'vest_or_reflective_uniform'}}};
 const states={7:{no_vest:{state:'unknown',observed_state:'unknown',observation_fresh:false,observed_at_s:3,last_sample_s:3}}};
 const view=ui.stateView(states,[assessment],{now:3});const pattern=view.rows[0].patterns[0];
 assert.equal(pattern.assessment.reason,'signal_clothing_policy_unverified');assert.equal(pattern.assessment.policy,'vest_or_reflective_uniform');
 assert.equal(pattern.state,'unknown');assert.equal(pattern.historyState,'unknown');assert.equal(view.violations,0);
 const html=ui.statesMarkup(states,[assessment],{now:3,labels:{no_vest:'Сигнальная одежда'},expandedKeys:new Set(['group:history'])});
 assert.match(html,/Человек #7/);assert.match(html,/Отражающая форма не определена/);assert.match(html,/Нет подтверждённого состояния/);
 assert.doesNotMatch(html,/Сигнальная одежда отсутствует|Жилет отсутствует|Жилет обнаружен|ordinary|confirmed-violation|confirmed-clear/);
});
test('confirmed compliant reflective workwear is labelled signal clothing, not a separate vest',()=>{
 const states={7:{no_vest:{state:'clear',observed_state:'clear',observation_fresh:true,confirmed_at_s:2,observed_at_s:3,last_sample_s:3}}};
 const rows=[{track_id:7,tracking:'measured',ppe:{vest:{state:'worn',observed_state:'worn',confirmed_state:'worn',confirmed_at_s:2,fresh:true,observed_at_s:3,reason:'measured',policy:'vest_or_reflective_uniform'}}}];
 const html=ui.statesMarkup(states,rows,{now:3,labels:{no_vest:'Сигнальная одежда'}});
 assert.match(html,/Сигнальная одежда обнаружена/);assert.doesNotMatch(html,/Жилет обнаружен|Жилет отсутствует/);
 // An explicitly stricter legacy vest-only requirement keeps its precise name.
 rows[0].ppe.vest.policy='vest_only';assert.match(ui.statesMarkup(states,rows,{now:3}),/Жилет обнаружен/);
});
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

test('fresh initial PPE observations collect confirmation instead of appearing wholly unknown',()=>{
 for(const observed_state of ['violation','clear']){
  const states={7:{no_helmet:{state:'unknown',observed_state,observation_fresh:true,observed_at_s:1,window_s:1.5,state_confirmation:{samples:2,min_samples:3}}}};
  const view=ui.stateView(states,[],{now:1});assert.equal(view.rows[0].patterns[0].currentState,'pending');assert.equal(view.rows[0].patterns[0].state,'unknown');
  const html=ui.statesMarkup(states,[],{now:1,labels:{no_helmet:'Без каски'}});assert.match(html,/Накопление подтверждения/);assert.match(html,observed_state==='violation'?/Каска отсутствует/:/Каска обнаружена/);assert.doesNotMatch(html,/class="confirmed-state confirmed-violation"/);assert.doesNotMatch(html,/class="confirmed-state confirmed-clear"/);
 }
});
test('unknown after confirmation retains an explicit historical result without claiming a fresh measurement',()=>{
 const states={7:{no_helmet:{state:'violation',observed_state:'unknown',observation_fresh:false,observed_at_s:10,last_sample_s:11,window_s:1.5,confirmed_at_s:9}}};
 const view=ui.stateView(states,[],{now:11}),p=view.rows[0].patterns[0];assert.equal(p.currentState,'unknown');assert.equal(p.historyState,'violation');assert.equal(p.measurementAt,11);
 const html=ui.statesMarkup(states,[],{now:11,expandedKeys:new Set(['group:history'])});assert.match(html,/Последнее подтверждение/);assert.match(html,/Нарушение подтверждено/);assert.match(html,/Нет свежего замера/);assert.doesNotMatch(html,/Накопление подтверждения/);
});
test('finished-session unknown cannot discard explicit assessment confirmation or become a current violation',()=>{
 const states={7:{no_helmet:{state:'unknown',observed_state:'unknown',observation_fresh:false,confirmed_at_s:1.5,observed_at_s:2,last_sample_s:2,window_s:1.5}}};
 const assessments=[{track_id:7,tracking:'measured',ppe:{helmet:{state:'missing',observed_state:'missing',fresh:true,observed_at_s:2,confirmed_state:'missing',confirmed_at_s:1.5}}}];
 const view=ui.stateView(states,assessments,{now:2,historical:true}),p=view.rows[0].patterns[0];assert.equal(p.state,'unknown');assert.equal(p.historyOnly,true);assert.equal(p.historyState,'violation');assert.equal(view.violations,0);
 const html=ui.statesMarkup(states,assessments,{now:2,historical:true});assert.match(html,/Последнее подтверждение/);assert.match(html,/Нарушение подтверждено/);assert.match(html,/последние состояния сохранённого сеанса/);
 assessments[0].ppe.helmet.confirmed_state='unknown';assert.equal(ui.stateView(states,assessments,{now:2}).rows[0].patterns[0].historyState,'unknown');
});
test('cached PPE and predicted tracking cannot masquerade as a fresh worn state',()=>{
 const assessments=[{track_id:7,tracking:'measured',ppe:{helmet:{state:'worn',observed_state:'worn',fresh:false,reason:'recent_observation',observed_at_s:2,confirmed_state:'worn',confirmed_at_s:1.5}}}];
 const view=ui.stateView({},assessments,{now:2});assert.equal(view.rows[0].patterns[0].currentState,'unknown');assert.equal(view.rows[0].patterns[0].measurement,'unknown');
 const html=ui.statesMarkup({},assessments,{now:2});assert.match(html,/Нет свежего замера/);assert.doesNotMatch(html,/class="confirmed-state confirmed-clear"/);
});
test('clear harness zone condition does not fabricate observed safety equipment',()=>{
 const states={7:{'no_harness_at_height:Height':{state:'clear',observed_state:'clear',observation_fresh:true,observed_at_s:2,window_s:2}}};
 const rows=[{track_id:7,tracking:'measured',ppe:{harness:{state:'missing',observed_state:'missing',fresh:true,observed_at_s:2,confirmed_state:'unknown'}}}];
 const html=ui.statesMarkup(states,rows,{now:2});assert.match(html,/Признак нарушения в зоне снят/);assert.match(html,/Привязь отсутствует/);assert.doesNotMatch(html,/Привязь обнаружена в зоне высоты/);
});

test('FSM last-confirmed fields preserve non-PPE history after interrupt without changing current state',()=>{
 const states={7:{possible_phone_use:{state:'unknown',observed_state:'unknown',observation_fresh:false,last_confirmed_state:'violation',last_confirmed_at_s:8,observed_at_s:9,last_sample_s:9,window_s:1}}};
 const view=ui.stateView(states,[],{now:9});assert.equal(view.rows[0].patterns[0].state,'unknown');assert.equal(view.rows[0].patterns[0].historyState,'violation');assert.equal(view.rows[0].patterns[0].historyAt,8);assert.equal(view.violations,0);
 const html=ui.statesMarkup(states,[],{now:9,expandedKeys:new Set(['group:history'])});assert.match(html,/Последнее подтверждение/);assert.match(html,/Нарушение подтверждено/);assert.match(html,/0:08.0/);
});

test('legacy historical event confirms only its exact job/person/pattern and keeps current state unknown',()=>{
 const assessments=[1,2].map(track_id=>({track_id,tracking:'measured',ppe:{helmet:{state:'missing',observed_state:'missing',fresh:true,observed_at_s:2}}}));
 const event={id:'legacy',job_id:'job1',track_id:1,kind:'no_helmet',zone:'',status:'new',start_s:.2,alert_s:1.6,payload:{}};
 const options={historical:true,jobId:'job1',events:[event],now:2};const view=ui.stateView({},assessments,options),one=view.rows.find(r=>r.id===1).patterns[0],two=view.rows.find(r=>r.id===2).patterns[0];
 assert.equal(one.state,'unknown');assert.equal(one.currentState,'unknown');assert.equal(one.historyState,'violation');assert.equal(one.historyAt,1.6);assert.equal(one.eventReview,'new');assert.equal(two.currentState,'pending');assert.equal(two.historyState,'unknown');assert.equal(view.violations,0);
 const html=ui.statesMarkup({},assessments,options);assert.match(html,/Последнее подтверждение/);assert.match(html,/Нарушение подтверждено · 0:01.6/);assert.match(html,/Новое событие/);assert.match(html,/Накопление подтверждения/);
 for(const wrong of [{job_id:'other'},{track_id:9},{kind:'no_vest'},{zone:'another'}])assert.equal(ui.stateView({},assessments,{...options,events:[{...event,...wrong}]}).rows[0].patterns[0].historyState,'unknown');
 assert.equal(ui.stateView({},assessments,{...options,historical:false}).rows[0].patterns[0].historyState,'unknown');assert.equal(ui.stateView({},assessments,{...options,jobId:undefined}).rows[0].patterns[0].historyState,'unknown');
});
test('explicit later confirmation beats legacy event and closed history needs validated closure boundaries',()=>{
 const event={job_id:'j',track_id:1,kind:'possible_phone_use',zone:'',status:'new',start_s:1,alert_s:2,payload:{episode_state:'closed',end_s:3,closed_at_s:4}};
 const state={1:{possible_phone_use:{state:'unknown',observed_state:'unknown',last_confirmed_state:'violation',last_confirmed_at_s:5}}};
 assert.equal(ui.stateView(state,[],{historical:true,jobId:'j',events:[event]}).rows[0].patterns[0].historyState,'violation');
 const confirmed=ui.ruleEventHistory([event],'j',1,'possible_phone_use').confirmed;assert.equal(confirmed.state,'clear');assert.equal(confirmed.at,4);
 for(const payload of [{end_s:null,closed_at_s:4},{end_s:3,closed_at_s:2.5},{end_s:.5,closed_at_s:4}])assert.equal(ui.ruleEventHistory([{...event,payload:{episode_state:'closed',...payload}}],'j',1,'possible_phone_use').confirmed.state,'violation');
});
test('operator-rejected legacy event never promotes a confirmed or accepted state',()=>{
 const event={job_id:'j',track_id:1,kind:'no_helmet',zone:'',status:'false',start_s:.2,alert_s:1.6};
 const assessments=[{track_id:1,tracking:'measured',ppe:{helmet:{state:'missing',observed_at_s:2}}}];
 const options={historical:true,jobId:'j',events:[event],now:2},view=ui.stateView({},assessments,options);assert.equal(view.rows[0].patterns[0].historyState,'unknown');assert.equal(view.violations,0);
 const html=ui.statesMarkup({},assessments,options);assert.match(html,/Отмечено как ложное срабатывание/);assert.doesNotMatch(html,/Нарушение подтверждено/);assert.doesNotMatch(html,/Подтверждено оператором/);
});
test('freshness group title and rule observation details use unambiguous copy',()=>{
 const states={1:{no_helmet:{state:'unknown',observed_state:'unknown',observed_at_s:1,last_sample_s:2,window_s:1}}};
 const html=ui.statesMarkup(states,[],{now:20,expandedKeys:new Set(['group:history'])});assert.match(html,/ID без свежего замера/);assert.doesNotMatch(html,/Ранее наблюдавшиеся ID/);assert.match(html,/Последний признак правила/);assert.doesNotMatch(html,/<span>Последний замер<\/span>/);
});
