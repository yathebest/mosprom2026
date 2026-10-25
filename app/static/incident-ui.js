/* Confirmed state and episode presentation. Inference and lifecycle decisions
   remain on the server; clip bounds and missing people never close an event. */
(function(root,factory){const api=factory();root.IncidentUI=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
 'use strict';
 const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const finite=value=>value!==null&&value!==undefined&&value!==''&&Number.isFinite(Number(value))?Number(value):null;
 const number=value=>new Intl.NumberFormat(globalThis.UiFormat?.locale?.()||(globalThis.document?.documentElement?.lang==='en'?'en-GB':'ru-RU'),{maximumFractionDigits:2}).format(value);
 const time=value=>{const s=finite(value);return s===null?'—':`${Math.floor(s/60)}:${(s%60).toFixed(1).padStart(4,'0')}`};
 const duration=value=>{if(globalThis.UiFormat?.duration)return globalThis.UiFormat.duration(value);const s=finite(value);if(s===null||s<0)return '—';const n=Math.round(s),h=Math.floor(n/3600),m=Math.floor(n%3600/60),sec=n%60;return [h?`${h} ч`:'',m?`${m} мин`:'',sec||!(h||m)?`${sec} с`:''].filter(Boolean).join(' ')};
 const stateLabels={unknown:'Пока не определено',violation:'Нарушение подтверждено',clear:'Признак нарушения снят'};
 const clearLabels={no_helmet:'Каска обнаружена',no_vest:'Сигнальная одежда обнаружена',no_mask:'Защита лица обнаружена',no_harness_at_height:'Признак нарушения в зоне снят'};
 const clearLabel=p=>p.kind==='no_vest'&&p.assessment?.policy==='vest_only'?'Жилет обнаружен':clearLabels[p.kind]||stateLabels.clear;
 const observationLabels={unknown:'Не определено',violation:'Признак нарушения',clear:'Признака нарушения нет'};
 const normalizeState=value=>({violation:'violation',clear:'clear',unknown:'unknown',missing:'violation',worn:'clear'})[value]||'unknown';
 function episode(event){
  const p=event?.payload||{},start=finite(event?.start_s??p.start_s),end=finite(p.end_s??event?.end_s);
  const claimed=p.episode_state||event?.episode_state||'unknown';
  const closed=claimed==='closed'&&end!==null&&start!==null&&end>=start;
  const duration=closed?(finite(p.duration_s??event?.duration_s)??end-start):null;
  const observed=finite(p.observed_until_s??event?.observed_until_s);
  return {id:p.episode_id||event?.episode_id||null,state:closed?'closed':claimed==='interrupted'?'interrupted':claimed==='open'?'open':'unknown',
   start,end:closed?end:null,duration:duration!==null&&duration>=0?duration:null,
   observedUntil:observed,observedDuration:start!==null&&observed!==null&&observed>=start?observed-start:null,
   closedAt:closed?finite(p.closed_at_s??event?.closed_at_s):null,
   endImage:closed?(p.end_image||event?.end_image||null):null,
   closureReason:p.closure_reason||event?.closure_reason||null,confirmation:p.state_confirmation||{}};
 }
 function technicalEpisodeMarkup(event,format=time){
  const e=episode(event),status={closed:'Эпизод завершён',open:'Эпизод продолжается',interrupted:'Наблюдение прервано',unknown:'Завершение не зафиксировано'}[e.state];
  return `<section class="episode-summary" data-episode-state="${e.state}"><span class="episode-badge episode-${e.state}">${status}</span><dl class="episode-times"><div><dt>Начало эпизода</dt><dd>${e.start===null?'—':format(e.start)}</dd></div><div><dt>Завершение эпизода</dt><dd>${e.end===null?'Не подтверждено завершение':format(e.end)}</dd></div><div><dt>Длительность эпизода</dt><dd>${e.duration===null?'Пока неизвестна':duration(e.duration)}</dd></div></dl>${e.end===null&&e.observedDuration!==null?`<p class="episode-observed"><span>Признак наблюдался до</span> <b>${format(e.observedUntil)}</b> <span>от начала записи; это не время завершения.</span></p>`:''}${e.closedAt!==null?`<p class="episode-observed"><span>Отсутствие нарушения подтверждено в</span> <b>${format(e.closedAt)}</b></p>`:''}${e.state==='interrupted'?'<p class="episode-observed">Остановка источника или потеря наблюдения не доказывает, что нарушение закончилось.</p>':''}</section>`;
 }
 function clipFacts(event){
  const p=event?.payload||{},start=finite(p.clip_start_s),rawEnd=finite(p.clip_end_s),media=finite(p.clip_duration_s);
  return {ready:Boolean(event?.clip),pending:!event?.clip&&['capturing','exporting','waiting_for_video_end'].includes(p.clip_state),
   start:start!==null&&start>=0?start:null,end:rawEnd!==null&&rawEnd>=0&&(start===null||rawEnd>=start)?rawEnd:null,duration:media!==null&&media>0?media:null};
 }
 function eventPresentation(event){const e=episode(event);return {detected:true,resolved:e.state==='closed'&&e.end!==null&&e.closedAt!==null&&e.closedAt>=e.end,alert:finite(event?.alert_s??event?.payload?.alert_s)}}
 function primaryEventMarkup(event,format=time){
  const state=eventPresentation(event),clip=clipFacts(event),title=event?.status==='false'?'Ложное срабатывание':state.resolved?'Завершено':'Нарушение зафиксировано';
  const field=(label,value)=>`<div><dt>${label}</dt><dd>${value}</dd></div>`;
  let facts='';
  if(clip.ready){const rows=[clip.start!==null?field('Начало фрагмента',format(clip.start)):'',clip.end!==null?field('Конец фрагмента',format(clip.end)):'',clip.duration!==null?field('Длительность клипа',clip.duration<60?`${number(clip.duration)} <span>с</span>`:duration(clip.duration)):''].join('');if(rows)facts=`<dl class="episode-times clip-facts">${rows}</dl>`;}
  else if(clip.pending)facts='<p class="event-clip-pending">Записывается клип</p>';
  else if(state.alert!==null)facts=`<p class="event-recorded-at"><span>Момент фиксации</span>: <b>${format(state.alert)}</b></p>`;
  if(!facts&&state.alert!==null)facts=`<p class="event-recorded-at"><span>Момент фиксации</span>: <b>${format(state.alert)}</b></p>`;
  return `<section class="episode-summary event-recorded-summary" data-system-event="recorded"><span class="episode-badge ${state.resolved?'episode-closed':'event-recorded-badge'}">${title}</span>${facts}</section>`;
 }
 function episodeMarkup(event,format=time){return primaryEventMarkup(event,format)+`<details class="event-technical"><summary>Технические сведения</summary>${technicalEpisodeMarkup(event,format)}${episodePolicyMarkup(event)}</details>`}
 function closureProofMarkup(event,format=time){
  const e=episode(event);if(!e.endImage)return '';
  return `<section class="episode-end-proof"><h3>Кадр подтверждённого завершения</h3><img src="/media/${escape(e.endImage)}" alt="Кадр подтверждённого завершения эпизода" loading="lazy"><p class="hint"><span>Граница эпизода</span> <b>${format(e.end)}</b>${e.closedAt!==null?` · <span>кадр подтверждения</span> <b>${format(e.closedAt)}</b>`:''}</p></section>`;
 }
 function policyMarkup(policy={}){
  const window=finite(policy.window_s),nominal=finite(policy.nominal_window_s),ratio=finite(policy.ratio),samples=finite(policy.min_samples);
  if(window===null&&ratio===null&&samples===null)return '';
  return `<div class="state-policy"><b>Как подтверждено состояние</b><p>${window!==null?`<span>Фактическое окно</span>: <b>${number(window)}</b> <span>с</span>. `:''}${nominal!==null&&nominal!==window?`<span>Заданное окно</span>: <b>${number(nominal)}</b> <span>с</span>. `:''}${ratio!==null?`<span>Доля времени с согласованным признаком</span>: <b>${Math.round(ratio*100)}%</b>. `:''}${samples!==null?`<span>Минимум свежих наблюдений</span>: <b>${samples}</b>.`:''}</p>${policy.cadence_limited?'<p class="hint">При редких свежих наблюдениях окно увеличено, чтобы получить достаточно измерений.</p>':''}<p class="hint">Доля времени в окне — правило подтверждения, а не точность нейросети.</p></div>`;
 }
 function episodePolicyMarkup(event){const p=event?.payload||{},start=p.start_state_confirmation||p.state_confirmation;return `<h3>Подтверждение начала</h3>${policyMarkup(start)}${episode(event).state==='closed'?`<h3>Подтверждение завершения</h3>${policyMarkup(p.state_confirmation)}`:''}`}
 const patternKinds=['danger_zone','fast_motion','possible_fall','person_down','no_helmet','no_vest','no_mask','no_harness_at_height','possible_phone_use','possible_smoking','fire','smoke','equipment_proximity','barrier_crossing','under_load','under_conveyor','unsafe_load_posture'];
 function confirmationConfig({ratio=.75,minSamples=3,windows={}}={}){
  ratio=finite(ratio);minSamples=finite(minSamples);
  if(ratio===null||ratio<.75||ratio>1)throw new Error('Доля времени должна быть от 75 до 100%.');
  if(minSamples===null||!Number.isInteger(minSamples)||minSamples<3||minSamples>30)throw new Error('Нужно от 3 до 30 свежих наблюдений.');
  const overrides={};for(const [kind,value] of Object.entries(windows)){
   if(value===''||value===null||value===undefined)continue;
   const n=finite(value);if(!patternKinds.includes(kind)||n===null||n<.2||n>60)throw new Error('Окно сценария должно быть от 0,2 до 60 секунд.');overrides[kind]=n;
  }
  return {state_confirmation_ratio:ratio,state_min_samples:minSamples,state_windows:overrides};
 }
 function mergeEvents(current,incoming){
  const old=new Map((current||[]).map(e=>[String(e.id),e])),updates=new Map((incoming||[]).map(e=>[String(e.id),e]));
  const additions=(incoming||[]).filter(e=>!old.has(String(e.id)));
  return additions.concat((current||[]).map(e=>{const update=updates.get(String(e.id));return update?{...e,...update,payload:{...(e.payload||{}),...(update.payload||{})}}:e}));
 }
 function isActivation(event){
  const state=event?.payload?.episode_state||event?.episode_state;
  return !['closed','interrupted'].includes(state)&&event?.payload?.transition!=='clear';
 }
 function assessmentObservation(value={}){
  return {state:normalizeState(value.observed_state??value.state),fresh:value.fresh!==false&&value.reason!=='recent_observation',
   observed_at_s:finite(value.observed_at_s),confirmed_state:normalizeState(value.confirmed_state),confirmed_at_s:finite(value.confirmed_at_s),
   reason:value.reason||null,policy:value.policy||null};
 }
 function personRows(states,assessments=[]){
  const rows=new Map();
  for(const [id,patterns] of Object.entries(states||{})){
   if(!patterns||typeof patterns!=='object')continue;
   rows.set(String(id),{id:Number(id),tracking:null,patterns:Object.entries(patterns).map(([key,value])=>({key,...value,state:normalizeState(value?.state),observed_state:normalizeState(value?.observed_state)}))});
  }
  for(const a of assessments||[]){
   const id=String(a.track_id),row=rows.get(id)||{id:Number(a.track_id),tracking:a.tracking,patterns:[]};row.tracking=a.tracking;row.assessed=true;
   for(const [key,value] of Object.entries(a.ppe||{})){
    const pattern={helmet:'no_helmet',vest:'no_vest',mask:'no_mask',harness:'no_harness_at_height'}[key]||key;
    const measured=assessmentObservation(value),matches=row.patterns.filter(p=>p.key===pattern||p.key.startsWith(pattern+':'));
    if(matches.length){for(const match of matches)match.assessment=measured;continue;}
    row.patterns.push({key:pattern,state:normalizeState(value.confirmed_state),observed_state:measured.state,observation_fresh:measured.fresh,
     confirmed_at_s:value.confirmed_at_s,observed_at_s:value.observed_at_s,window_s:value.state_confirmation?.window_s,
     agreement:value.agreement??value.state_confirmation?.agreement,state_confirmation:value.state_confirmation,reason:value.reason,assessment:measured});
   }
   rows.set(id,row);
  }
  return Array.from(rows.values()).sort((a,b)=>a.id-b.id);
 }
 function ruleEventHistory(events,jobId,trackId,key){
  if(jobId===null||jobId===undefined||jobId==='')return {confirmed:null,rejected:null};
  const colon=key.indexOf(':'),kind=colon<0?key:key.slice(0,colon),zone=colon<0?'':key.slice(colon+1),confirmed=[],rejected=[];
  for(const event of events||[]){
   if(String(event.job_id)!==String(jobId)||Number(event.track_id)!==Number(trackId)||event.kind!==kind||String(event.zone||'')!==zone)continue;
   const payload=event.payload||{},start=finite(event.start_s??payload.start_s),alert=finite(event.alert_s??payload.alert_s);
   if(start===null||alert===null||start<0||alert<start)continue;
   if(event.status==='false'){rejected.push({at:alert,status:'false'});continue;}
   const end=finite(payload.end_s??event.end_s),closedAt=finite(payload.closed_at_s??event.closed_at_s),claimed=payload.episode_state||event.episode_state;
   const validClear=claimed==='closed'&&end!==null&&closedAt!==null&&closedAt>=end&&end>=start&&closedAt>=alert;
   confirmed.push({state:validClear?'clear':'violation',at:validClear?closedAt:alert,source:'event',review:event.status||'new'});
  }
  confirmed.sort((a,b)=>b.at-a.at);rejected.sort((a,b)=>b.at-a.at);return {confirmed:confirmed[0]||null,rejected:rejected[0]||null};
 }
 function stateView(states,assessments=[],options={}){
  const now=finite(options.now),rows=personRows(states,assessments).map(row=>{
   row.patterns=row.patterns.map(p=>{
    const colon=p.key.indexOf(':'),kind=colon<0?p.key:p.key.slice(0,colon),zone=colon<0?'':p.key.slice(colon+1);
    const observed=finite(p.observed_at_s??p.last_observed_s??p.last_seen_s),window=finite(p.window_s??p.state_confirmation?.window_s);
    const stale=Boolean(p.stale)||(now!==null&&observed!==null&&now-observed>Math.max(3,(window??1)*2));
    const last=stale||p.observation_fresh===false||['weak','predicted'].includes(row.tracking)?'unknown':p.observed_state;
    const pending=last!=='unknown'&&(p.state==='unknown'||last!==p.state);let currentState=last==='unknown'?'unknown':pending?'pending':p.state;
    const eventHistory=options.historical?ruleEventHistory(options.events,options.jobId,row.id,p.key):{confirmed:null,rejected:null};const explicitHistory=normalizeState(p.last_confirmed_state),historyCandidates=[{state:explicitHistory,at:finite(p.last_confirmed_at_s),source:'snapshot'},{state:p.assessment?.confirmed_state||'unknown',at:p.assessment?.confirmed_at_s??null,source:'assessment'},...(eventHistory.confirmed?[eventHistory.confirmed]:[])].filter(v=>v.state!=='unknown'&&v.at!==null).sort((a,b)=>b.at-a.at);const historyState=p.state!=='unknown'?p.state:historyCandidates[0]?.state||'unknown',historyAt=p.state!=='unknown'?finite(p.confirmed_at_s):historyCandidates[0]?.at??null,historySource=p.state!=='unknown'?'snapshot':historyCandidates[0]?.source;if(p.state==='unknown'&&historyState!=='unknown'&&options.historical)currentState='unknown';const eventReview=historySource==='event'?historyCandidates[0]?.review:null,rejectedEvent=eventHistory.rejected&&(historyAt===null||eventHistory.rejected.at>=historyAt)?eventHistory.rejected:null;
    const measuredAt=p.assessment?.observed_at_s,assessmentLatest=p.assessment&&(measuredAt===null||observed===null||measuredAt>=observed);
    const measurement=assessmentLatest?(p.assessment.fresh&&!stale&&!['weak','predicted'].includes(row.tracking)?p.assessment.state:'unknown'):last;
    return {...p,kind,zone,observed,window,stale,last,currentState,historyState,historyAt,historySource,eventReview,rejectedEvent,historyOnly:p.state==='unknown'&&historyState!=='unknown',measurement,measurementAt:measurement==='unknown'?(finite(p.last_sample_s)??observed):(assessmentLatest?measuredAt:observed)};
   });
   const recentSample=row.patterns.some(p=>{const sample=finite(p.last_sample_s??p.observed);return sample===null||now===null||now-sample<=Math.max(3,(p.window??1)*2)});row.current=Boolean(row.assessed&&row.tracking!=='predicted'&&recentSample)||row.patterns.some(p=>!p.stale&&p.observation_fresh!==false&&p.observed!==null&&row.tracking!=='predicted');
   row.violations=row.patterns.filter(p=>p.state==='violation').length;
   return row;
  });
  const current=rows.filter(r=>r.current).sort((a,b)=>Boolean(b.violations)-Boolean(a.violations)||a.id-b.id),history=rows.filter(r=>!r.current).sort((a,b)=>Boolean(b.violations)-Boolean(a.violations)||b.id-a.id);
  const limit=Number.isInteger(options.limit)?Math.max(1,options.limit):4;
  return {rows,primary:current.slice(0,limit),more:current.slice(limit),history,violations:rows.reduce((n,r)=>n+r.violations,0),violationIds:rows.filter(r=>r.violations).map(r=>r.id)};
 }
 function stateRenderKey(view,assessments=[],options={}){
  return JSON.stringify([Boolean(options.historical),options.jobId??null,options.ratio??.75,view.rows.map(r=>[r.id,r.current,r.tracking,r.patterns.map(p=>[p.key,p.state,p.observed_state,p.currentState,p.stale,p.historyState,p.historyAt,p.historySource,p.eventReview,p.rejectedEvent,p.historyOnly,p.measurement,p.measurementAt,p.confirmed_at_s,p.observed,p.window,p.clear_window_s,p.agreement,p.state_confirmation,p.episode_id,p.assessment])]),assessments.map(r=>[r.track_id,r.tracking,r.visibility?.small_objects,Object.entries(r.ppe||{}).map(([k,v])=>[k,v.observed_state||v.state,v.reason,v.observed_at_s])])]);
 }
 function stateCardMarkup(row,options={}){
  const format=options.format||time,labels=options.labels||{},expanded=options.expandedKeys||new Set(),name=row.id===0?'Сцена':`Человек #${row.id}`;
  return `<article class="person-state-card" data-person-state-id="${row.id}"><h4>${name}${row.violations?` <span class="state-violation-count"><span>Не снято</span>: ${row.violations}</span>`:''}</h4><div class="person-state-brief">${row.patterns.map(p=>{
   const history=p.historyState==='clear'?clearLabel(p):stateLabels[p.historyState];
   const accumulating=p.currentState==='pending'&&p.state==='unknown';
   const text=accumulating?'Накопление подтверждения':p.currentState==='pending'?'Проверяем изменение состояния':p.historyState!=='unknown'?history:'Нет подтверждённого состояния';
   const stateClass=p.currentState==='unknown'&&p.historyState!=='unknown'?'history':p.currentState;
   const prefix=p.historyOnly||p.currentState==='unknown'&&p.historyState!=='unknown'?'Последнее подтверждение':accumulating||p.currentState==='pending'?'Подтверждение':p.historyState==='unknown'?'Подтверждение':'Подтверждено правилом';
   const raw={no_helmet:{violation:'Каска отсутствует',clear:'Каска обнаружена'},no_vest:p.assessment?.policy==='vest_only'?{violation:'Жилет отсутствует',clear:'Жилет обнаружен'}:{violation:'Сигнальная одежда отсутствует',clear:'Сигнальная одежда обнаружена'},no_mask:{violation:'Защита лица отсутствует',clear:'Защита лица обнаружена'},no_harness_at_height:{violation:'Привязь отсутствует',clear:'Привязь обнаружена'}};
   const policyLimited=p.assessment?.reason==='signal_clothing_policy_unverified';const measured=policyLimited?'Отражающая форма не определена':p.measurement==='unknown'&&(p.stale||p.observation_fresh===false||p.assessment?.fresh===false||['weak','predicted'].includes(row.tracking))?'Нет свежего замера':raw[p.kind]?.[p.measurement]||observationLabels[p.measurement];
   return `<div class="person-state-brief-row" data-current-state="${p.currentState}" data-confirmed-state="${p.state}"><span><b>${escape(labels[p.kind]||p.kind)}</b>${p.zone?` · <span data-i18n-ignore>${escape(p.zone)}</span>`:''}</span><div class="state-layers"><span class="state-layer-label">${prefix}</span><span class="confirmed-state confirmed-${stateClass}">${text}${prefix==='Последнее подтверждение'&&p.historyAt!==null?` · ${format(p.historyAt)}`:''}</span>${p.eventReview?`<small class="state-operator-review">${{new:'Новое событие',confirmed:'Подтверждено',forwarded:'Передано ответственному'}[p.eventReview]||'Статус события'}</small>`:''}${p.rejectedEvent?`<small class="state-operator-rejected"><span>Отмечено как ложное срабатывание</span> · ${format(p.rejectedEvent.at)}</small>`:''}<small class="state-measurement-summary"><span>Замер</span>: <b>${measured}</b>${p.measurementAt!==null?` · ${format(p.measurementAt)}`:''}${p.stale?' · <span>Данные устарели</span>':''}</small>${p.currentState==='pending'&&p.state!=='unknown'?`<small class="state-previous-confirmed"><span>Последнее подтверждение</span>: ${history}</small>`:''}${accumulating&&p.window!==null?`<small class="state-confirmation-progress"><span>Окно</span>: ${p.window} <span>с</span> · <span>Свежих наблюдений</span>: ${Number(p.state_confirmation?.samples??0)} / ${Number(p.state_confirmation?.min_samples??options.minSamples??3)}</small>`:''}</div></div>`;
  }).join('')}</div><details class="person-state-measurements" data-preserve-key="person:${row.id}" ${expanded.has('person:'+row.id)?'open':''}><summary>Замеры и история подтверждения</summary>${row.patterns.map(p=>{
   const agreement=finite(p.agreement),required=finite(p.state_confirmation?.ratio??options.ratio)??.75;
   return `<div class="person-pattern-state"><b>${escape(labels[p.kind]||p.kind)}</b>${p.zone?` · <span data-i18n-ignore>${escape(p.zone)}</span>`:''}<p class="state-history"><span>Последнее подтверждённое состояние</span>: <b>${p.historyState==='clear'?clearLabel(p):stateLabels[p.historyState]}</b>${p.historyAt!==null?` · ${format(p.historyAt)}`:''}</p><p class="measurement-line"><span>Последний признак правила</span>: <b>${observationLabels[p.last]}</b>${finite(p.last_sample_s??p.observed)!==null?` · ${format(p.last_sample_s??p.observed)}`:''}${p.stale?' · <span>Данные устарели</span>':''}</p>${p.window!==null?`<p class="state-window"><span>Окно подтверждения</span>: <b>${number(p.window)}</b> <span>с</span> · <span>доля времени не менее</span> <b>${Math.round(required*100)}%</b>${agreement!==null?` · <span>сейчас</span> <b>${Math.round(agreement*100)}%</b>`:''}${finite(p.state_confirmation?.nominal_window_s)!==null&&Number(p.state_confirmation.nominal_window_s)!==p.window?` · <span>задано</span> <b>${number(p.state_confirmation.nominal_window_s)}</b> <span>с</span>`:''} · <span>минимум наблюдений</span> <b>${Number(p.state_confirmation?.min_samples??options.minSamples??3)}</b>${finite(p.clear_window_s)!==null?` · <span>окно снятия признака</span> <b>${number(p.clear_window_s)}</b> <span>с</span>`:''}</p>`:''}</div>`;
  }).join('')}</details></article>`;
 }
 function stateGroupMarkup(view,group,options={}){return (view[group]||[]).map(row=>stateCardMarkup(row,options)).join('')}
 function statesMarkup(states,assessments=[],options={}){
  const view=options.view||stateView(states,assessments,options),expanded=options.expandedKeys||new Set();if(!view.rows.length)return '';
  const group=(key,title)=>view[key].length?`<details class="state-id-group" data-state-group="${key}" data-preserve-key="group:${key}" ${expanded.has('group:'+key)?'open':''}><summary><span>${title}</span> · <b>${view[key].length}</b>${view[key].some(r=>r.violations)?` · <span class="state-violation-count"><span>Не снято</span>: ${view[key].reduce((n,r)=>n+r.violations,0)}</span>`:''}</summary><div class="state-group-body" data-state-group-body="${key}">${expanded.has('group:'+key)?stateGroupMarkup(view,key,options):''}</div></details>`:'';
  return '<div class="confirmed-states"><div class="states-panel-head"><h3>Подтверждённые состояния по ID</h3><div class="states-counts"><span>В наблюдении</span>: <b>'+view.rows.filter(r=>r.current).length+'</b> · <span>Ожидают наблюдения</span>: <b>'+view.history.length+'</b>'+ (view.violations?' · <span class="state-violation-count"><span>Не снято</span>: '+view.violations+'</span>':'')+'</div></div><p class="hint state-compact-help">Замер показывает находку на кадре. Подтверждение относится к временному правилу. Статус события сохраняется отдельно.</p>'+(view.violations?'<p class="state-pending-ids"><span>Нарушения не сняты у ID</span>: <span data-i18n-ignore>'+view.violationIds.slice(0,12).map(id=>id===0?'0':'#'+id).join(', ')+(view.violationIds.length>12?' …':'')+'</span></p>':'')+(options.historical?'<p class="hint">Это последние состояния сохранённого сеанса, а не текущее наблюдение за камерой.</p>':'')+'<div class="state-primary-grid">'+view.primary.map(row=>stateCardMarkup(row,options)).join('')+'</div>'+group('more','Остальные люди в наблюдении')+group('history','ID без свежего замера')+'<details class="state-reading-help" data-preserve-key="reading-help" '+(expanded.has('reading-help')?'open':'')+'><summary>Как читать состояния</summary><p class="hint">Подтверждённое состояние и последний замер показаны отдельно. Пропуск или неопределённый кадр не означает, что нарушение закончилось.</p><p class="hint">Доля относится ко времени в окне, а не к числу кадров. Это не точность модели и не вероятность нарушения. Неопределённые кадры не подтверждают безопасное состояние.</p></details></div>';
 }

 return {episode,clipFacts,eventPresentation,primaryEventMarkup,technicalEpisodeMarkup,episodeMarkup,closureProofMarkup,policyMarkup,episodePolicyMarkup,patternKinds,confirmationConfig,mergeEvents,isActivation,assessmentObservation,personRows,ruleEventHistory,stateView,stateRenderKey,stateCardMarkup,stateGroupMarkup,statesMarkup};
});
