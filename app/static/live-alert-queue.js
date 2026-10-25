/* Local notification acknowledgement is independent of event review/lifecycle. */
(function(root,factory){const api=factory();root.LiveAlertQueue=api;if(typeof module==='object'&&module.exports)module.exports=api;})(globalThis,function(){
 'use strict';
 function incident(event){
  const start=Number(event?.start_s),alert=Number(event?.alert_s);
  return event?.id!==null&&event?.id!==undefined&&String(event.id)!==''&&Boolean(event?.job_id)&&Boolean(event?.kind)
   &&event?.start_s!==null&&event?.start_s!==undefined&&event?.alert_s!==null&&event?.alert_s!==undefined
   &&event.start_s!==''&&event.alert_s!==''&&Number.isFinite(start)&&Number.isFinite(alert)&&start>=0&&alert>=start&&event?.payload?.transition!=='clear';
 }
 function create(){
  let scope=null,initialized=false,active=null,sequence=0;
  const records=new Map(),events=new Map(),episodes=new Map(),unread=[];
  function reset(nextScope=null){scope=nextScope;initialized=false;active=null;sequence=0;records.clear();events.clear();episodes.clear();unread.length=0;}
  function setScope(next){if(scope!==next)reset(next)}
  function ingest(rows,options={}){
   if(options.scope!==undefined&&options.scope!==scope)return {added:[],ignored:true};
   const bootstrap=!initialized,added=[];
   // API order is newest first. Queue order stays chronological, and the last
   // genuinely new episode becomes visible even when older alerts are unread.
   for(const incoming of [...rows].reverse()){
    if(!incident(incoming))continue;
    const eventKey=JSON.stringify([String(incoming.job_id),String(incoming.id)]);
    const episodeId=incoming.payload?.episode_id||incoming.episode_id;
    const episodeKey=episodeId?JSON.stringify([String(incoming.job_id),String(episodeId)]):null;
    let record=episodeKey?episodes.get(episodeKey):null;
    const sameEvent=events.get(eventKey);
    if(!record&&sameEvent&&(!episodeKey||!sameEvent.episodeKey||sameEvent.episodeKey===episodeKey))record=sameEvent;
    if(record){
     record.event={...record.event,...incoming,payload:{...(record.event.payload||{}),...(incoming.payload||{})}};
     if(episodeKey){record.episodeKey=episodeKey;episodes.set(episodeKey,record)}
     events.set(eventKey,record);continue;
    }
    if(options.updatesOnly)continue;
    const key='notification-'+(++sequence);
    record={key,event:{...incoming,payload:{...(incoming.payload||{})}},episodeKey};
    records.set(key,record);events.set(eventKey,record);if(episodeKey)episodes.set(episodeKey,record);
    if(!bootstrap){unread.push(key);active=key;added.push(record)}
   }
   if(!options.updatesOnly)initialized=true;return {added,bootstrap,ignored:false};
  }
  function snapshot(){return {initialized,scope,total:unread.length,active:records.get(active)||null,pending:unread.map(key=>records.get(key))}}
  function dismiss(key=active){const index=unread.indexOf(key);if(index<0)return false;unread.splice(index,1);if(active===key)active=unread[0]||null;return true;}
  function next(expected=active){if(expected!==active||!unread.length)return false;active=unread[(unread.indexOf(active)+1)%unread.length];return true;}
  return {setScope,reset,ingest,snapshot,dismiss,next};
 }
 return {create,incident};
});
