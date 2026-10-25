const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {create}=require('../app/static/live-alert-queue.js');
const source=fs.readFileSync(require('node:path').join(__dirname,'../app/static/live.js'),'utf8');
const event=(id,episode=id,job='a')=>({id,job_id:job,kind:'no_helmet',camera:job.toUpperCase(),track_id:1,start_s:1,alert_s:2,created:'2026-10-25T12:00:00Z',payload:{episode_id:episode,episode_state:'open',live:true}});
function queue(){const q=create();q.setScope('org');return q}
test('history bootstraps silently; new IDs with exactly the same timestamp still notify',()=>{
 const q=queue(),old=event('old');assert.equal(q.ingest([old],{scope:'org'}).added.length,0);assert.equal(q.snapshot().total,0);
 const fresh=event('fresh');assert.equal(fresh.created,old.created);assert.equal(q.ingest([fresh,old],{scope:'org'}).added.length,1);
 assert.equal(q.snapshot().active.event.id,'fresh');assert.equal(q.snapshot().total,1);
 assert.equal(q.ingest([fresh,old],{scope:'org'}).added.length,0);
});
test('each new episode becomes latest while prior unread items remain reachable and dismissable',()=>{
 const q=queue();q.ingest([]);q.ingest([event('one')]);const one=q.snapshot().active.key;
 q.ingest([event('two','two','b'),event('one')]);const two=q.snapshot().active.key;
 assert.equal(q.snapshot().total,2);assert.equal(q.snapshot().active.event.job_id,'b');
 q.next(two);assert.equal(q.snapshot().active.key,one);
 q.dismiss(one);assert.equal(q.snapshot().total,1);assert.equal(q.snapshot().active.key,two);
 assert.equal(q.dismiss(one),false);q.dismiss(two);assert.equal(q.snapshot().active,null);
 q.ingest([event('two','two','b'),event('one')]);assert.equal(q.snapshot().total,0);
});
test('clip/closure/review updates and duplicate rows for one episode never create another popup',()=>{
 const q=queue();q.ingest([]);q.ingest([event('one','episode')]);const key=q.snapshot().active.key;
 const changed={...event('one','episode'),status:'confirmed',clip:'clip.mp4',payload:{episode_id:'episode',episode_state:'closed',end_s:3,closed_at_s:4}};
 assert.equal(q.ingest([changed]).added.length,0);assert.equal(q.snapshot().active.event.clip,'clip.mp4');assert.equal(q.snapshot().active.event.payload.episode_state,'closed');
 assert.equal(q.ingest([{...changed,id:'duplicate-row'}]).added.length,0);assert.equal(q.snapshot().active.key,key);assert.equal(q.snapshot().total,1);
 q.dismiss(key);q.ingest([changed]);assert.equal(q.snapshot().total,0);
});
test('legacy identity upgraded to episode metadata is an update, genuinely new episode is new',()=>{
 const q=queue();q.ingest([]);const legacy=event('one');delete legacy.payload.episode_id;q.ingest([legacy]);
 assert.equal(q.ingest([event('one','resolved-identity')]).added.length,0);
 assert.equal(q.ingest([event('one','next-episode')]).added.length,1);assert.equal(q.snapshot().total,2);
});
test('an unseen episode already closed between polls is still a new incident',()=>{
 const q=queue();q.ingest([]);
 const closed={...event('short'),payload:{episode_id:'short',episode_state:'closed',end_s:2.5,closed_at_s:3}};
 assert.equal(q.ingest([closed]).added.length,1);assert.equal(q.snapshot().active.event.payload.episode_state,'closed');
 const clear={...event('clear'),payload:{transition:'clear'}};assert.equal(q.ingest([clear]).added.length,0);
});
test('scope resets isolate unread/seen history and discard late responses from previous company',()=>{
 const q=queue();q.ingest([],{scope:'org'});q.ingest([event('one')],{scope:'org'});q.setScope('other');
 assert.equal(q.snapshot().total,0);assert.equal(q.ingest([event('leaked')],{scope:'org'}).ignored,true);
 assert.equal(q.ingest([event('old-other')],{scope:'other'}).added.length,0);
 assert.equal(q.ingest([event('new-other')],{scope:'other'}).added.length,1);
});
test('unbounded journal refresh only updates known incidents and cannot burst older archive rows',()=>{
 const q=queue();q.ingest([]);q.ingest([event('one')]);
 assert.equal(q.ingest([event('historic'),{...event('one'),clip:'ready.mp4'}],{updatesOnly:true}).added.length,0);
 assert.equal(q.snapshot().total,1);assert.equal(q.snapshot().active.event.clip,'ready.mp4');
});
test('enabled sound schedules one distinct tone per new notification without overlaps',()=>{
 const start=source.indexOf('function beep()'),end=source.indexOf('function renderCameraWall',start),tones=[];
 const sound={currentTime:10,destination:{},createOscillator(){const tone={frequency:{},connect(){},start:at=>tones.push(at),stop(){}};return tone},createGain(){return {gain:{setValueAtTime(){},exponentialRampToValueAtTime(){}},connect(){}}}};
 const enabled={checked:true},context={L:{sound,nextBeepAt:0},$:()=>enabled,isChecked:()=>enabled.checked};vm.runInNewContext(source.slice(start,end),context);
 context.beep();context.beep();context.beep();assert.deepEqual(tones,[10,10.5,11]);enabled.checked=false;context.beep();assert.equal(tones.length,3);
});
test('UI latest/source pointer changes per episode; updates neither sound again nor force camera selection',()=>{
 const start=source.indexOf('function openNotificationSource('),end=source.indexOf("$('live-alert-clips').onclick",start);
 const q=queue();q.ingest([]);const nodes=new Map(),actions=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{hidden:false,innerHTML:'',textContent:'',querySelector:selector=>{if(!actions.has(selector))actions.set(selector,{});return actions.get(selector)}});return nodes.get(id)};
 const L={alertQueue:q,current:{id:'a'},sessions:[{id:'a'},{id:'b'}]},S={page:'analytics'},selected=[],sounds=[];
 const context={IncidentUI:require('../app/static/incident-ui.js'),L,S,$,labels:{no_helmet:'Без каски'},esc:String,timestamp:String,UiFormat:{clock:String},beep:()=>sounds.push(1),openLiveSource:id=>selected.push(id),showPage(){},renderClips(){}};
 vm.runInNewContext(source.slice(start,end),context);
 context.receiveLiveAlerts([event('one')],'org');context.receiveLiveAlerts([event('two','two','b'),event('one')],'org');
 assert.equal(sounds.length,2);assert.equal(L.current.id,'a');assert.deepEqual(selected,[]);assert.equal($('global-live-alert').hidden,false);
 $('global-live-open').onclick();assert.deepEqual(selected,['b']);
 context.receiveLiveAlerts([{...event('two','two','b'),clip:'ready.mp4'},event('one')],'org');assert.equal(sounds.length,2);
 $('global-live-close').onclick();assert.equal(q.snapshot().active.event.job_id,'a');$('global-live-open').onclick();assert.deepEqual(selected,['b','a']);
 assert.match($('live-alert-queue').innerHTML,/Непрочитано: 1/);
 $('global-live-close').onclick();assert.equal(q.snapshot().total,0);assert.equal($('live-alert-queue').innerHTML,'','An empty notification queue must not duplicate the event-feed empty state');
});
