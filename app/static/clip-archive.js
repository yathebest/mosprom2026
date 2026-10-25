/* Presentation groups use only authorized rows supplied by the current workspace. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.ClipArchive=api})(typeof globalThis!=='undefined'?globalThis:this,function(){
 const finite=v=>v!==null&&v!==undefined&&v!==''&&Number.isFinite(Number(v))?Number(v):null;
 const pending=e=>!e.clip&&['capturing','exporting','waiting_for_video_end'].includes(e.payload?.clip_state);
 const episodeKey=e=>e.payload?.episode_id?JSON.stringify([e.kind,e.zone||'',e.payload.episode_id]):'event:'+String(e.id);
 function group(events=[]){
  const rows=events.filter(e=>e.clip||e.payload?.clip_state),parents=rows.map((_,i)=>i),tokens=new Map();
  const find=i=>parents[i]===i?i:parents[i]=find(parents[i]);
  for(let i=0;i<rows.length;i++){
   const e=rows[i],p=e.payload||{},scope=e.job_id?JSON.stringify([e.job_id,e.camera||'']):null;
   if(!scope)continue;
   const keys=[];if(p.clip_group_id)keys.push('group:'+p.clip_group_id);if(e.clip)keys.push('file:'+e.clip);
   const start=finite(p.clip_start_s),end=finite(p.clip_end_s);
   if(!p.clip_group_id&&start!==null&&end!==null&&end>start)keys.push('bounds:'+start+':'+end);
   for(const key of keys){const token=scope+'|'+key;if(tokens.has(token))parents[find(i)]=find(tokens.get(token));else tokens.set(token,i)}
  }
  const bins=new Map();for(let i=0;i<rows.length;i++){const key=find(i);if(!bins.has(key))bins.set(key,[]);bins.get(key).push(rows[i])}
  return [...bins.values()].map(members=>{
   members.sort((a,b)=>(finite(b.alert_s)??0)-(finite(a.alert_s)??0)||String(a.id).localeCompare(String(b.id)));
   const representative=members.find(e=>e.clip&&e.status!=='false')||members.find(e=>e.clip)||members.find(e=>e.status!=='false')||members[0],seen=new Set(),counts={},falseCounts={};
   // A rejected duplicate cannot cancel another row's accepted episode, nor count twice.
   for(const e of [...members].sort((a,b)=>Number(a.status==='false')-Number(b.status==='false'))){const key=episodeKey(e);if(seen.has(key))continue;seen.add(key);const target=e.status==='false'?falseCounts:counts;target[e.kind]=(target[e.kind]||0)+1;}
   return {key:JSON.stringify([representative.job_id,representative.camera,members.map(e=>String(e.id)).sort()]),members,representative,counts,falseCounts,total:Object.values(counts).reduce((a,b)=>a+b,0),ready:members.some(e=>e.clip),pending:members.some(pending),alert:Math.max(...members.map(e=>finite(e.alert_s)??0)),created:Math.max(...members.map(e=>Date.parse(e.created)||0))};
  }).sort((a,b)=>b.created-a.created||b.alert-a.alert||a.key.localeCompare(b.key));
 }
 function filter(groups,filters={},dateKey=()=>null){return groups.filter(g=>{
  if(filters.state==='ready'&&!g.ready||filters.state==='pending'&&(g.ready||!g.pending))return false;
  return g.members.some(e=>(!filters.camera||e.camera===filters.camera)&&(!filters.kind||e.kind===filters.kind)&&(filters.start==null||finite(e.alert_s)!==null&&e.alert_s>=filters.start)&&(filters.end==null||finite(e.alert_s)!==null&&e.alert_s<=filters.end)&&(!filters.date||dateKey(e.payload?.occurred_at||e.created)===filters.date));
 })}
 return {group,filter,episodeKey};
});
