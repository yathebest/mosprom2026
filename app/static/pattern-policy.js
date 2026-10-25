/* New-run availability comes from the server. Historical event labels stay separate. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.PatternPolicy=api})(typeof globalThis!=='undefined'?globalThis:this,function(){
 function create(manifest={}){
  const patterns=Array.isArray(manifest.patterns)?manifest.patterns:[],byId=new Map(patterns.map(p=>[p.id,p]));
  const flags=new Set(manifest.enabled_flags||[]),disabledFlags=new Set(manifest.disabled_flags||[]),zones=new Set(manifest.enabled_zone_kinds||[]);
  const allFlags=new Set(['fast_motion','fall_detection',...['helmet','vest','mask','harness','phone','fire','smoke','smoking','proximity','load_posture'].map(key=>key+'_detection'),...patterns.map(p=>p.config_flag).filter(Boolean),...(manifest.disabled_flags||[])]);
  const eventKinds=new Set(patterns.filter(p=>p.enabled).flatMap(p=>p.event_kinds||[]));
  function sanitize(input={}){
   const c=structuredClone(input);for(const flag of allFlags)if(disabledFlags.has(flag)||!flags.has(flag))c[flag]=false;
   c.zones=(c.zones||[]).filter(z=>zones.has(z.kind||'danger_zone'));
   if(c.state_windows)c.state_windows=Object.fromEntries(Object.entries(c.state_windows).filter(([kind])=>eventKinds.has(kind)));
   return c;
  }
  function removed(input={}){return [...Array.from(allFlags).filter(flag=>!flags.has(flag)&&input[flag]).map(setting=>({setting})),...(input.zones||[]).filter(z=>!zones.has(z.kind||'danger_zone')).map(z=>({setting:'zones',name:z.name,kind:z.kind||'danger_zone'}))]}
  return Object.freeze({version:manifest.version,patterns,enabled:id=>Boolean(byId.get(id)?.enabled),allowsFlag:flag=>flags.has(flag)&&!disabledFlags.has(flag),isDisabledFlag:flag=>disabledFlags.has(flag),allowsZone:kind=>zones.has(kind),allowsEvent:kind=>eventKinds.has(kind),sanitize,removed,
   sampleAllowed:sample=>Array.isArray(sample.pattern_ids)&&sample.pattern_ids.length>0&&sample.pattern_ids.every(id=>byId.get(id)?.enabled)});
 }
 return {create};
});
