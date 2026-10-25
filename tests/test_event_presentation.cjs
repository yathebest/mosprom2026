const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const ui=require('../app/static/incident-ui.js');
const tc=v=>`TC${v}`;

test('clip source bounds and actual media duration are distinct from episode duration',()=>{
 const event={id:'a',status:'new',clip:'clip.mp4',start_s:11,alert_s:12,payload:{episode_state:'interrupted',clip_start_s:10,clip_end_s:17,clip_duration_s:6.91,observed_until_s:16}};
 const html=ui.primaryEventMarkup(event,tc);
 assert.match(html,/Нарушение зафиксировано/);assert.match(html,/Начало фрагмента.*TC10/);assert.match(html,/Конец фрагмента.*TC17/);assert.match(html,/Длительность клипа.*6,91/);
 assert.doesNotMatch(html,/Наблюдение прервано|неизвест|Не подтверждено|Длительность эпизода/);
 assert.equal(ui.episode(event).end,null);assert.equal(ui.episode(event).duration,null);
 assert.match(ui.episodeMarkup(event),/<details class="event-technical"><summary>Технические сведения<\/summary>/);
 assert.match(ui.technicalEpisodeMarkup(event),/Наблюдение прервано/);
 assert.equal(event.status,'new');assert.equal(event.payload.episode_state,'interrupted');
});

test('missing clip metadata is omitted, pending export is explicit, valid closure alone shows completed',()=>{
 const event={id:'a',alert_s:12,start_s:10,status:'new',payload:{episode_state:'open'}};
 assert.match(ui.primaryEventMarkup({...event,payload:{...event.payload,clip_state:'capturing'}}),/Записывается клип/);
 const legacy=ui.primaryEventMarkup({...event,clip:'old.mp4'},tc);
 assert.match(legacy,/Момент фиксации.*TC12/);assert.doesNotMatch(legacy,/Длительность клипа|Начало фрагмента|Не подтверждено/);
 const bad={...event,payload:{episode_state:'closed',end_s:11,closed_at_s:10}};
 assert.equal(ui.eventPresentation(bad).resolved,false);assert.doesNotMatch(ui.primaryEventMarkup(bad),/>Завершено</);
 assert.equal(ui.eventPresentation({...event,payload:{episode_state:'closed',end_s:11,closed_at_s:12}}).resolved,true);
 assert.match(ui.primaryEventMarkup({...event,status:'false'}),/Ложное срабатывание/);
});

test('restored principal live controls are prominent and retain active-session locking',()=>{
 const html=fs.readFileSync(require.resolve('../app/static/index-v2.html'),'utf8');
 const source=fs.readFileSync(require.resolve('../app/static/scenarios-ui.js'),'utf8');
 const policy=JSON.parse(fs.readFileSync(require.resolve('../app/pattern-policy.json'),'utf8'));
 for(const key of ['fall','danger_zone','helmet','phone'])assert.ok(policy.enabled_patterns.includes(key));
 assert.match(html,/id="live-fall"[\s\S]*?Падение \/ человек на полу/);
 assert.match(html,/id="live-zone-on"[\s\S]*?Опасная \/ запретная зона/);
 assert.match(source,/querySelector\('\.live-checks h3'\)\.after\(\$\('live-extra-helmet'\)/);
 assert.match(source,/#live \.live-checks input/);
 const labels=source.match(/const extraLabels=([^;]+)/)[1];assert.match(labels,/possible_phone_use:'Телефон в руках'/);
});

test('event workflow retains original status values and explicit user action',()=>{
 const source=fs.readFileSync(require.resolve('../app/static/app.js'),'utf8');
 assert.match(source,/new:'Новое событие',confirmed:'Подтверждено',false:'Ложное срабатывание',forwarded:'Передано ответственному'/);
 assert.match(source,/el\.onchange=\(\)=>reviewEvent\(el\.dataset\.review,el\.value\)/);
 assert.match(source,/body:JSON\.stringify\(\{status\}\)/);
 assert.doesNotMatch(source,/Оператор проверяет каждое событие|episode-clip-note/);
});
