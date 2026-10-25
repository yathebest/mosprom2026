/* Capability filtering affects creation only. History, review and evidence remain intact. */
function sanitizePatternConfig(config){
 if(!S.patternPolicy)throw Error('Каталог сценариев не загружен. Обновите страницу.');
 return S.patternPolicy.sanitize(config);
}
function showPatternWarnings(hostId,warnings=[]){
 const host=$(hostId);if(!host)return;
 let note=host.querySelector('.pattern-policy-note');
 if(!warnings.length){if(note)note.remove();return}
 if(!note){note=document.createElement('p');note.className='pattern-policy-note callout compact';note.setAttribute('role','status');host.prepend(note)}
 note.textContent='Недоступные правила исключены из этой копии настроек. Исторические события сохранены.';
}
function applyPatternAvailability(manifest){
 S.patternPolicy=PatternPolicy.create(manifest);
 for(const id of ['builder','live-setup-drawer'])if($(id))$(id).inert=false;
 const p=S.patternPolicy,remove=id=>$(id)?.closest('label')?.remove();
 const zoneAllowed=(manifest.enabled_zone_kinds||[]).length>0;
 for(const [id,enabled] of [['check-fall',p.enabled('fall')],['live-fall',p.enabled('fall')],['check-motion',p.enabled('fast_motion')],['live-motion',p.enabled('fast_motion')],['check-zone',zoneAllowed],['live-zone-on',zoneAllowed]])if(!enabled){setCheck(id,false);remove(id)}
 const params={phone_confidence:p.enabled('phone'),ppe_hold:['helmet','vest','mask','harness'].some(id=>p.enabled(id)),object_hold:scenarioOptions.some(([id])=>p.enabled(id)),object_confidence:scenarioOptions.some(([id])=>p.enabled(id)),proximity_heights:p.enabled('proximity'),proximity_m:p.enabled('proximity')};
 for(const prefix of ['extra','live-extra']){
  for(const [key] of scenarioOptions)if(!p.allowsFlag(key+'_detection')){setCheck(prefix+'-'+key,false);remove(prefix+'-'+key)}
  for(const [key,enabled] of Object.entries(params))if(!enabled)remove(prefix+'-'+key);
  if(!p.enabled('phone')||p.isDisabledFlag('phone_work_context_only')){setCheck(prefix+'-phone_work_context_only',false);remove(prefix+'-phone_work_context_only');}
  if(!p.enabled('smoke'))$(prefix+'-smoke-note')?.remove();
  for(const kind of IncidentUI.patternKinds)if(!p.allowsEvent(kind))$(prefix+'-window-'+kind)?.closest('[data-state-window-row]')?.remove();
  const experimental=$(prefix+'-experimental');
  if(experimental){
   const summary=experimental.querySelector('summary');if(summary)summary.textContent='Другие сценарии';
   for(const paragraph of experimental.querySelectorAll(':scope > p'))paragraph.remove();
   if(!experimental.querySelector('input[type="checkbox"]'))experimental.remove();
  }
  updateConfirmationControls(prefix);
 }
 for(const [id,enabled] of [['body-threshold',p.enabled('fall')],['body-hold',p.enabled('fall')],['speed',p.enabled('fast_motion')],['motion-hold',p.enabled('fast_motion')],['metric-speed',p.enabled('fast_motion')],['zone-hold',['danger_zone','barrier','under_load','under_conveyor'].some(id=>p.enabled(id))]])if(!enabled)remove(id);
 if(!p.enabled('fall'))for(const paragraph of $('step1').querySelectorAll('details > p'))if(paragraph.textContent.includes('Метрики падений относятся'))paragraph.remove();
 for(const id of ['zone-kind','live-zone-kind']){
  const select=$(id);if(!select)continue;
  for(const option of Array.from(select.options))if(!p.allowsZone(option.value))option.remove();
  if(select.options.length&&!select.value)select.value=select.options[0].value;
 }
 if(!p.enabled('danger_zone')){
  const label=$('check-zone')?.closest('label');
  if(label){const title=label.querySelector('strong'),description=label.querySelector('span:not(.scenario-icon)');if(title)title.textContent='Контекст камеры';if(description)description.textContent='Разметьте доступные области для выбранных правил камеры.';}
 }
 document.dispatchEvent(new CustomEvent('linq:patternpolicychange',{detail:{version:p.version}}));
}
function patternPolicyUnavailable(){
 for(const id of ['builder','live-setup-drawer'])if($(id))$(id).inert=true;
 if($('start'))$('start').disabled=true;if($('live-start'))$('live-start').disabled=true;
 const host=$('live-setup-drawer');if(host){let note=host.querySelector('.pattern-policy-note');if(!note){note=document.createElement('p');note.className='pattern-policy-note callout';host.prepend(note)}note.textContent='Каталог сценариев не загружен. Обновите страницу перед новым запуском.';}
}
