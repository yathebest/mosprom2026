/* Existing-session presentation and browser device ownership; never starts NN. */
(function(root,factory){const api=factory();root.CameraWall=api;if(typeof module==='object'&&module.exports)module.exports=api;})(globalThis,function(){
 'use strict';
 const active=s=>['starting','running','stopping'].includes(s?.state);
 const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const en=()=>globalThis.document?.documentElement?.lang==='en';
 function createBrowserStreams(){
  const entries=new Map();
  function remove(id){const entry=entries.get(id);if(!entry)return false;entries.delete(id);entry.controller?.abort();entry.stream.getTracks().forEach(track=>track.stop());entry.video.pause();entry.video.srcObject=null;return true;}
  return {get:id=>entries.get(id),list:()=>Array.from(entries.values()),hasDevice:id=>Boolean(id)&&Array.from(entries.values()).some(entry=>entry.deviceId===id),
   add(id,entry){if(entries.has(id))throw Error('Device session already attached');entries.set(id,{...entry,id});return entries.get(id)},remove,stopAll(){for(const id of Array.from(entries.keys()))remove(id)}};
 }
 function freshness(session){
  if(['stopped','failed'].includes(session.state)){
   const stopped=session.stopped_at&&globalThis.UiFormat?.clock?globalThis.UiFormat.clock(session.stopped_at):null;
   return {text:(en()?(session.state==='failed'?'Last frame of unavailable stream':'Last frame of stopped stream'):(session.state==='failed'?'Последний кадр недоступного эфира':'Последний кадр остановленного эфира'))+(stopped?' · '+stopped:''),stale:false,terminal:true};
  }
  if(session.state==='stopping')return {text:en()?'Stream stopping · last received frame':'Эфир завершается · последний полученный кадр',stale:false,terminal:true};
  const ms=session.preview_frame_age_ms;
  if(!Number.isFinite(ms))return {text:en()?'No frame received yet':'Кадр ещё не получен',stale:true};
  return {text:ms>3000?(en()?`Frame stale · ${(ms/1000).toFixed(1)} s`:`Кадр устарел · ${(ms/1000).toFixed(1)} с`):(en()?`Frame ${(ms/1000).toFixed(1)} s ago`:`Кадр ${(ms/1000).toFixed(1)} с назад`),stale:ms>3000};
 }
 function activeEvents(sessions,events){const ids=new Set(sessions.filter(active).map(session=>session.id));return events.filter(event=>ids.has(event.job_id))}
 function eventFeed(sessions,events,options={}){
  const rows=activeEvents(sessions,events).slice(0,6),format=options.time||String;
  if(!rows.length)return '<div class="empty camera-alert-empty"><b>В работающих камерах тревог пока нет</b><p>Прошлые происшествия доступны в истории событий.</p></div>';
  return rows.map(event=>{const track=Number(event.track_id??event.payload?.track_id),person=Number.isSafeInteger(track)&&track>0?` · ${en()?'person':'человек'} #${track}`:'';return `<article class="camera-alert-row"><div><h3>${escape(options.labels?.[event.kind]||event.kind)}</h3><p data-i18n-ignore>${escape(event.camera)}${person}</p><p class="hint">${escape(format(event.start_s))} → ${escape(format(event.alert_s))} · ${event.payload?.occurred_at?escape(options.clock?.(event.payload.occurred_at)||event.payload.occurred_at):'обработано '+escape(options.clock?.(event.created)||event.created)}</p></div><div class="camera-wall-card-actions"><button type="button" class="btn ghost small" data-alert-source="${escape(event.job_id)}">Открыть камеру</button><button type="button" class="btn ghost small" data-alert-review="${escape(event.id)}">Разобрать</button></div></article>`}).join('');
 }
 function renderEvents(container,sessions,events,options={}){
  if(!container)return;const html=eventFeed(sessions,events,options);if(container._wallEventHTML===html)return;container._wallEventHTML=html;container.innerHTML=html;
  for(const button of container.querySelectorAll('[data-alert-source]'))button.onclick=()=>options.onSource?.(button.dataset.alertSource);
  for(const button of container.querySelectorAll('[data-alert-review]'))button.onclick=()=>options.onReview?.(button.dataset.alertReview);
 }
 function render(container,sessions,options={}){
  if(!container)return;
  if(!container._cameraCards){
   container.innerHTML='<header class="camera-wall-header"><div><h2>Камеры оператора</h2><p data-wall-access></p></div><button type="button" class="btn ghost small" data-wall-sound>Включить звук тревог</button></header><p class="hint camera-wall-limit">До 2 одновременных эфиров на локальном стенде: предел связан с ресурсами оборудования. Просмотр карточки не запускает новый анализ.</p><div class="camera-wall-grid" data-wall-grid></div><p class="hint" data-wall-empty>Нет доступных эфиров. Администратор подключает камеры, оператор наблюдает назначенные источники.</p>';
   container._cameraCards=new Map();
  }
  container.querySelector('[data-wall-access]').textContent=options.canManage?'Выбор камеры меняет только просмотр. Остальные эфиры продолжают работать.':'Только просмотр разрешённых камер. Подключение и остановка доступны администратору.';
  const sound=container.querySelector('[data-wall-sound]');sound.textContent=options.soundEnabled?'Звук тревог включён':'Включить звук тревог';sound.onclick=options.onSound;
  const cards=container._cameraCards,ids=new Set(sessions.map(s=>s.id));
  for(const [id,card] of cards)if(!ids.has(id)){card.remove();cards.delete(id)}
  container.querySelector('[data-wall-empty]').hidden=Boolean(sessions.length);
  for(const session of sessions){
   let card=cards.get(session.id);
   if(!card){card=document.createElement('article');card.className='camera-wall-card';card.dataset.cameraSession=session.id;
    card.innerHTML='<button type="button" class="camera-wall-preview" data-wall-open><img alt="Последний кадр камеры" hidden><span data-wall-placeholder>Кадр ещё не получен</span></button><div class="camera-wall-card-body"><h3 data-wall-name data-i18n-ignore></h3><p><span class="pill" data-wall-state></span></p><p data-wall-freshness></p><p data-wall-events></p><div class="camera-wall-card-actions"><button type="button" class="btn ghost small" data-wall-open>Открыть камеру</button><button type="button" class="btn ghost small" data-wall-stop>Остановить эту камеру</button></div></div>';
    container.querySelector('[data-wall-grid]').append(card);cards.set(session.id,card);
   }
   card.classList.toggle('selected',options.selectedId===session.id);const fresh=freshness(session);card.classList.toggle('stale',fresh.stale);
   card.querySelector('[data-wall-name]').textContent=session.camera;
   card.querySelector('[data-wall-state]').textContent=options.states?.[session.state]||session.state;
   card.querySelector('[data-wall-freshness]').textContent=fresh.text;
   card.querySelector('[data-wall-events]').textContent=en()?`${Number(session.events)||0} events · ${Number(session.clips_pending)||0} clips saving`:`${Number(session.events)||0} событий · ${Number(session.clips_pending)||0} клипов сохраняется`;
   for(const button of card.querySelectorAll('[data-wall-open]'))button.onclick=()=>options.onSelect?.(session.id);
   const stop=card.querySelector('[data-wall-stop]');stop.hidden=!options.canManage;stop.disabled=!['starting','running'].includes(session.state)||options.stopping?.has(session.id);stop.onclick=()=>options.onStop?.(session.id);
   const image=card.querySelector('img'),now=Date.now();
   if(options.visible!==false&&session.preview_version>0&&image.dataset.version!==String(session.preview_version)&&now-Number(image.dataset.updated||0)>=1000){
    image.dataset.version=String(session.preview_version);image.dataset.updated=String(now);
    image.onload=()=>{image.hidden=false;card.querySelector('[data-wall-placeholder]').hidden=true};
    image.onerror=()=>{image.hidden=true;card.querySelector('[data-wall-placeholder]').hidden=false};
    image.src=session.snapshot_url+'?v='+encodeURIComponent(session.preview_version);
   }
  }
 }
 return {active,activeEvents,eventFeed,createBrowserStreams,freshness,render,renderEvents};
});
