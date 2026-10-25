const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const wall=require('../app/static/camera-wall.js');
const source=fs.readFileSync(require('node:path').join(__dirname,'../app/static/live.js'),'utf8');
function device(id){const track={stops:0,stop(){this.stops++}},video={pauses:0,pause(){this.pauses++},srcObject:{}};return {id,deviceId:id,stream:{getTracks:()=>[track]},video,track}}
test('browser camera ownership is independent per session and stopping one releases only it',()=>{
 const streams=wall.createBrowserStreams(),a=device('a'),b=device('b');streams.add('a',a);streams.add('b',b);
 assert.equal(streams.hasDevice('a'),true);assert.equal(streams.hasDevice('other'),false);
 assert.equal(streams.remove('a'),true);assert.equal(a.track.stops,1);assert.equal(a.video.srcObject,null);
 assert.equal(b.track.stops,0);assert.equal(streams.get('b').video,b.video);assert.equal(streams.remove('a'),false);
 streams.stopAll();assert.equal(b.track.stops,1);assert.equal(streams.list().length,0);
});
test('a new preview owns independent tracks and cannot stop an adopted clone',()=>{
 const adopted=device('active'),preview=device('setup'),streams=wall.createBrowserStreams();streams.add('active',adopted);
 preview.stream.getTracks().forEach(track=>track.stop());assert.equal(preview.track.stops,1);assert.equal(adopted.track.stops,0);
 assert.equal(streams.get('active').id,'active');
});
test('freshness describes last displayed frame, never inference confidence or promised FPS',()=>{
 assert.equal(wall.freshness({preview_frame_age_ms:100}).stale,false);
 assert.equal(wall.freshness({preview_frame_age_ms:4500}).stale,true);
 assert.match(wall.freshness({}).text,/ещё не получен/);assert.equal(wall.freshness({}).stale,true);
 for(const state of ['stopped','failed','stopping']){const last=wall.freshness({state,preview_frame_age_ms:260000});assert.equal(last.stale,false);assert.equal(last.terminal,true);assert.doesNotMatch(last.text,/устарел|260/);assert.match(last.text,/Последний кадр|последний полученный кадр/)}
});
test('common alerts show active authorized session IDs only and escape user labels',()=>{
 const sessions=[{id:'a',state:'running'},{id:'b',state:'starting'},{id:'old',state:'stopped'}];
 const events=[{id:'b1',job_id:'b',camera:'<camera B>',kind:'danger_zone',start_s:1,alert_s:2,created:'utc'},
  {id:'old1',job_id:'old',camera:'Past demo',kind:'danger_zone'}, {id:'secret',job_id:'foreign',camera:'Foreign'}];
 assert.deepEqual(wall.activeEvents(sessions,events).map(event=>event.id),['b1']);
 const html=wall.eventFeed(sessions,events,{time:String});assert.match(html,/&lt;camera B&gt;/);assert.match(html,/data-alert-source="b"/);assert.doesNotMatch(html,/Past demo|Foreign|episode-summary/);
 const empty=wall.eventFeed([],events);assert.match(empty,/Тревог пока нет/);assert.equal((empty.match(/class="camera-alert-empty"/g)||[]).length,1);assert.match(empty,/aria-hidden="true"/);assert.doesNotMatch(empty,/Новых уведомлений нет/);
});
test('otherwise identical alerts identify distinct people and localize without exposing scene or invalid IDs',()=>{
 const sessions=[{id:'a',state:'running'}],base={job_id:'a',camera:'Line A',kind:'no_helmet',start_s:1,alert_s:2,created:'utc'};
 const events=[1,2,3].map(track_id=>({...base,id:String(track_id),track_id}));
 const russian=wall.eventFeed(sessions,events);for(const id of [1,2,3])assert.match(russian,new RegExp('Line A · человек #'+id));
 const previous=globalThis.document;globalThis.document={documentElement:{lang:'en'}};
 try{const english=wall.eventFeed(sessions,events);assert.match(english,/Line A · person #2/);assert.doesNotMatch(english,/человек/)}finally{if(previous===undefined)delete globalThis.document;else globalThis.document=previous}
 for(const track_id of [null,undefined,0,-1,1.5,'<script>',Infinity])assert.doesNotMatch(wall.eventFeed(sessions,[{...base,track_id}]),/человек #|person #/);
 assert.match(wall.eventFeed(sessions,[{...base,payload:{track_id:7}}]),/человек #7/);
});

test('selection changes view only and retains independent browser sources',()=>{
 const start=source.indexOf('function selectLive('),end=source.indexOf('function renderLiveStatus',start);
 const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{dataset:{},classList:{add(){}},hidden:true});return nodes.get(id)};
 const streams=wall.createBrowserStreams(),a=device('a'),b=device('b');streams.add('a',a);streams.add('b',b);
 const L={current:null,selectionVersion:0,browserStreams:streams},context={L,$,renderLiveStatus(){},renderCameraWall(){}};
 vm.runInNewContext(source.slice(start,end),context);
 context.selectLive({id:'a',view_url:'/a'});context.selectLive({id:'b',view_url:'/b'});context.selectLive({id:'b',view_url:'/b'});
 assert.equal($('live-feed').src,'/b');assert.equal(L.current.id,'b');assert.equal(streams.list().length,2);assert.equal(a.track.stops+b.track.stops,0);
});
test('stop captures its target across awaited API response and never stops newly selected browser camera',async()=>{
 const start=source.indexOf('async function stopLiveSession('),end=source.indexOf("$('live-start').onclick=",start);let complete;
 const a={id:'a',state:'running'},b={id:'b',state:'running'},removed=[],calls=[];
 const L={current:a,sessions:[a,b],stopping:new Set(),browserStreams:{remove:id=>removed.push(id)}};
 const context={L,can:()=>true,renderCameraWall(){},renderLiveStatus(){},toast(){},post:url=>{calls.push(url);return new Promise(resolve=>complete=resolve)}};
 vm.runInNewContext(source.slice(start,end),context);
 const pending=context.stopLiveSession('a');L.current=b;complete({ok:true});await pending;
 assert.deepEqual(calls,['/api/live/a/stop']);assert.deepEqual(removed,['a']);assert.equal(L.current.id,'b');assert.equal(b.state,'running');assert.equal(a.state,'stopping');
 context.can=()=>false;await context.stopLiveSession('b');assert.equal(calls.length,1);
});
test('polling collects alerts from the second allowed camera while the first remains selected',async()=>{
 const start=source.indexOf('function liveWorkspaceKey('),end=source.indexOf('function syncEpisodeAlerts',start),calls=[],notified=[];
 let round=0;const sessions=[{id:'a',state:'running'},{id:'b',state:'running'}],event={id:'new-b',job_id:'b',kind:'no_helmet',start_s:1,alert_s:2,camera:'B',created:new Date().toISOString(),payload:{episode_state:'open'}};
 const L={alertQueue:require('../app/static/live-alert-queue.js').create(),polling:false,workspaceKey:null,sessions,current:sessions[0],events:[],notified:new Set(),seen:new Set(),openedAt:0,browserStreams:{list:()=>[],stopAll(){}}};
 const A={workspace:true,user:{id:'operator',organization_id:'org',cameras:['A','B']}},node={className:'',textContent:''};
 const context={L,A,window:{A},S:{events:[],page:'live'},$ :()=>node,can:permission=>permission==='live.read',CameraWall:{renderEvents(){}},UiFormat:{clock:String},labels:{},timestamp:String,openLiveSource(){},openEvent(){},renderCameraWall(){},renderLiveStatus(){},updateOpenEpisode(){},syncEpisodeAlerts(){},selectLive(){throw Error('Should retain selected A')},alertEvent(){throw Error('Other camera must use global alert')},globalLiveAlert:event=>notified.push(event.job_id),receiveLiveAlerts:(events,scope)=>{for(const row of L.alertQueue.ingest(events,{scope}).added)notified.push(row.event.job_id)},IncidentUI:{mergeEvents:(old,rows)=>rows,isActivation:()=>true},api:async url=>{calls.push(url);return url==='/api/live'?sessions:(round++===0?[]:[event])}};
 vm.runInNewContext(source.slice(start,end),context);await context.pollLive();
 assert.deepEqual(calls.sort(),['/api/live','/api/live-events']);assert.equal(L.current.id,'a');assert.deepEqual(notified,[]);
 await context.pollLive();assert.deepEqual(notified,['b']);await context.pollLive();assert.deepEqual(notified,['b']);
});
test('concurrent polling is coalesced and a failed request retry cannot lose or duplicate a new episode',async()=>{
 const start=source.indexOf('function liveWorkspaceKey('),end=source.indexOf('function syncEpisodeAlerts',start),q=require('../app/static/live-alert-queue.js').create(),calls=[],notifications=[];
 const sessions=[{id:'a',state:'running'}],event={id:'new',job_id:'a',kind:'no_helmet',start_s:1,alert_s:2,created:'same-second',payload:{episode_id:'ep'}};
 const L={polling:false,workspaceKey:null,sessions,current:sessions[0],events:[],alertQueue:q,browserStreams:{list:()=>[],stopAll(){}}},A={workspace:true,user:{id:'op',organization_id:'org',cameras:null}},node={};
 let resolveSessions,resolveEvents;
 const context={L,A,window:{A},S:{events:[],page:'live'},$:()=>node,can:()=>true,CameraWall:{renderEvents(){}},UiFormat:{clock:String},labels:{},timestamp:String,openLiveSource(){},openEvent(){},renderCameraWall(){},renderLiveStatus(){},updateOpenEpisode(){},syncEpisodeAlerts(){},selectLive(){},receiveLiveAlerts:(rows,scope)=>notifications.push(...q.ingest(rows,{scope}).added),IncidentUI:{mergeEvents:(old,rows)=>rows},api:url=>{calls.push(url);return new Promise(resolve=>{if(url==='/api/live')resolveSessions=resolve;else resolveEvents=resolve})}};
 vm.runInNewContext(source.slice(start,end),context);
 const pending=context.pollLive();await context.pollLive();assert.equal(calls.length,2);resolveSessions(sessions);resolveEvents([]);await pending;
 context.api=async url=>{if(url==='/api/live')throw Error('temporary disconnect');return [event]};await context.pollLive();assert.equal(notifications.length,0);
 context.api=async url=>url==='/api/live'?sessions:[event];await context.pollLive();await context.pollLive();assert.equal(notifications.length,1);assert.equal(notifications[0].event.id,'new');
});
