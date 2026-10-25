// Setup controls: camera preview and the always-visible zone editor.
window.LiveLayout={
  openSetup(options={}){
    const drawer=$('live-setup-drawer');if(!drawer)return;
    drawer.open=true;$('live-config-open')?.setAttribute('aria-expanded','true');
    if(options.scroll!==false)drawer.scrollIntoView({behavior:'smooth',block:'start'});
  },
  closeSetup(){const drawer=$('live-setup-drawer');if(drawer)drawer.open=false;$('live-config-open')?.setAttribute('aria-expanded','false');}
};
$('live-config-open').onclick=()=>LiveLayout.openSetup();
$('live-config-close').onclick=()=>{LiveLayout.closeSetup();$('live-config-open').focus()};
$('live-setup-drawer').addEventListener('toggle',()=>{$('live-config-open').setAttribute('aria-expanded',String($('live-setup-drawer').open))});
// Source cards and toolbars have different heights for operators and admins.
// Size the workspace from its actual document position, rather than a fixed offset.
function fitLiveWorkspace(){
  const workspace=document.querySelector('#live .live-workspace');
  if(!workspace||!workspace.getClientRects().length)return;
  const top=workspace.getBoundingClientRect().top+window.scrollY;
  workspace.style.setProperty('--live-workspace-height',Math.max(260,Math.min(700,window.innerHeight-top-18))+'px');
}
if(typeof ResizeObserver!=='undefined'){
  const liveSizeObserver=new ResizeObserver(fitLiveWorkspace);
  [document.querySelector('#live>.heading'),$('live-camera-wall'),document.querySelector('.live-operation-actions')].filter(Boolean).forEach(el=>liveSizeObserver.observe(el));
}
window.addEventListener('resize',fitLiveWorkspace);
window.addEventListener('hashchange',fitLiveWorkspace);
requestAnimationFrame(fitLiveWorkspace);
let setupActiveSession=null;
const cameraPreview=createCameraPreview({
  getMediaDevices:()=>navigator.mediaDevices,
  video:$('webcam-preview'),
  onChange:state=>{L.media=state.stream;L.cameraPending=state.pending;renderCameraControls()}
});
function liveSetupLocked(){return !L.drafting&&Boolean(L.current&&['starting','running','stopping'].includes(L.current.state))}
function renderCameraControls(){
  const browser=$('live-source').value==='browser',opened=Boolean(L.media),pending=Boolean(L.cameraPending);
  $('webcam-preview').hidden=!opened;
  $('live-preview').disabled=liveSetupLocked()||pending;
  $('live-disconnect').hidden=!(browser||opened||pending);
  $('live-disconnect').disabled=!opened&&!pending;
  $('live-disconnect').textContent=L.browserSession?'Отключить камеру и остановить эфир':'Отключить камеру';
  $('live-device-status').textContent=pending?'Запрашиваем доступ к камере…':opened?(L.browserSession?'Камера передаёт кадры в эфир':'Камера включена для предпросмотра'):browser?'Камера выключена':'RTSP / HTTP: получаем один кадр для настройки';
  $('live-device-status').classList.toggle('camera-on',opened);
  $('live-edit-zones').disabled=!L.image;
}
stopWebcam=function(){L.browserSession=null;cameraPreview.stop()};
prepareWebcam=async function(){
  if(L.browserSession)throw Error('Веб-камера уже передаёт кадры. Сначала остановите её эфир.');
  const device=$('live-device').value,result=await cameraPreview.start(device);
  if(!result)return false;
  $('live-device').innerHTML='<option value="">Камера по умолчанию</option>'+result.devices.filter(d=>d.kind==='videoinput').map((d,i)=>`<option value="${esc(d.deviceId)}">${esc(d.label||'Камера '+(i+1))}</option>`).join('');
  $('live-device').value=device;return true;
};
$('live-disconnect').onclick=async()=>{
  const session=L.browserSession;stopWebcam();
  toast('Камера выключена. Полученный кадр и зоны сохранены.');
  if(session){
    try{await post(`/api/live/${session}/stop`);toast('Камера выключена. Завершаем эфир и сохраняем клипы.')}
    catch(e){toast('Камера выключена. Сервер недоступен; сеанс завершится по тайм-ауту.')}
  }
};
const previousCameraMode=cameraMode;
cameraMode=function(){previousCameraMode();renderCameraControls()};
function focusLiveZones(){
  LiveLayout.openSetup({scroll:false});
  $('live-zone-details').scrollIntoView({behavior:'smooth',block:'start'});
  $('live-zone-details').classList.add('zone-attention');
  setTimeout(()=>$('live-zone-details').classList.remove('zone-attention'),1800);
}
$('live-edit-zones').onclick=focusLiveZones;
$('live-zone-get-frame').onclick=()=>$('live-preview').click();
$('live-zone-on').onchange=()=>{updateLiveZoneControls();if(isChecked('live-zone-on'))focusLiveZones()};
const previousSetLiveImage=setLiveImage;
setLiveImage=async function(url){await previousSetLiveImage(url);renderCameraControls();updateLiveZoneControls()};
$('live-preview').onclick=async()=>{
  try{
    $('live-preview').disabled=true;
    if($('live-source').value==='browser'){
      if(!await prepareWebcam())return;
      const video=$('webcam-preview');
      if(video.videoWidth){
        const c=document.createElement('canvas');c.width=video.videoWidth;c.height=video.videoHeight;
        c.getContext('2d').drawImage(video,0,0);await setLiveImage(c.toDataURL('image/jpeg'));
      }
    }else{
      const response=await fetch('/api/live-preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:$('live-url').value})});
      if(!response.ok){const data=await response.json();throw Error(data.detail||'Не удалось получить кадр')}
      const url=URL.createObjectURL(await response.blob());
      try{await setLiveImage(url)}finally{URL.revokeObjectURL(url)}
    }
    focusLiveZones();toast('Кадр готов. Отметьте не менее трёх точек, выберите назначение и сохраните зону.');
  }catch(e){toast(e.name==='NotAllowedError'?'Разрешите доступ к камере или выберите RTSP / HTTP.':e.message)}
  finally{$('live-preview').disabled=liveSetupLocked();renderCameraControls()}
};
function updateLiveZoneControls(){
  const locked=liveSetupLocked(),hasFrame=Boolean(L.image),points=L.points.length;renderLivePointTools();
  $('live-zone-placeholder').hidden=hasFrame;$('live-zone-canvas').hidden=!hasFrame;
  $('live-zone-count').textContent=`${L.zones.length} / 8 зон`;
  $('live-zone-progress').textContent=locked?'Правила закреплены на время эфира':points>=3?`${points} точек · можно сохранить зону`:`${points} точек · нужно не менее 3`;
  $('live-zone-enabled').textContent=isChecked('live-zone-on')?'Размеченные зоны включены':'Зоны выключены. После сохранения первой зоны включатся автоматически.';
  $('live-save-zone').disabled=locked||!hasFrame||points<3;
  $('live-zone-undo').disabled=locked||!points;$('live-zone-clear').disabled=locked||!points;
  $('live-zone-get-frame').disabled=locked;
  if($('live-zone-kind'))$('live-zone-kind').disabled=locked;
  $('live-zone-canvas').setAttribute('aria-disabled',String(locked));
  $('live-zone-list').querySelectorAll('button').forEach(el=>el.disabled=locked);$('live-point-tools').querySelectorAll('button').forEach(el=>el.disabled=locked);$('live-save-zone').textContent=L.editingZone===null?'Сохранить зону':'Сохранить изменения зоны';
}
paintLiveZones=function(){
  const c=$('live-zone-canvas'),ctx=c.getContext('2d');
  if(L.image){
    c.width=L.image.naturalWidth;c.height=L.image.naturalHeight;ctx.drawImage(L.image,0,0);
    c.style.maxWidth=`${Math.round(520*c.width/c.height)}px`;
    function polygon(points,color,name){
      if(!points.length)return;
      ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x*c.width,y*c.height):ctx.moveTo(x*c.width,y*c.height));
      if(points.length>=3)ctx.closePath();ctx.fillStyle=color+'40';ctx.fill();ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,c.width/300);ctx.stroke();
      points.forEach(([x,y],i)=>{ctx.beginPath();ctx.arc(x*c.width,y*c.height,Math.max(4,c.width/120),0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.font=`bold ${Math.max(14,c.width/40)}px Segoe UI`;ctx.fillText(String(i+1),x*c.width+8,y*c.height-8)});
      if(name){ctx.font=`bold ${Math.max(15,c.width/35)}px Segoe UI`;ctx.fillStyle=color;ctx.fillText(name,points[0][0]*c.width+8,Math.max(24,points[0][1]*c.height-10))}
    }
    L.zones.forEach((z,i)=>{if(i!==L.editingZone)polygon(z.points,'#c7ef8a',z.name)});polygon(L.points,'#ffda7a','Новая зона');
  }
  $('live-zone-list').innerHTML=L.zones.length?L.zones.map((z,i)=>`<div class="zone-item"><span><b>${esc(z.name)}</b><small>${esc(zoneKinds[z.kind||'danger_zone'])} · ${z.points.length} точки</small></span><div class="zone-actions"><button type="button" data-live-edit="${i}" aria-label="Изменить зону ${esc(z.name)}">Изменить</button><button type="button" data-live-remove="${i}" aria-label="Удалить ${esc(z.name)}">✕</button></div></div>`).join(''):'<p class="hint zone-empty">Сохранённых зон пока нет.</p>';
  $('live-zone-list').querySelectorAll('[data-live-remove]').forEach(el=>el.onclick=()=>{if(liveSetupLocked())return;const index=Number(el.dataset.liveRemove),editingName=L.editingZone!==null?L.zones[L.editingZone]?.name:null,draftName=$('live-zone-name').value;L.zones.splice(index,1);L.zones=ZoneNavigation.renumber(L.zones);if(L.editingZone===index)resetLiveZoneDraft();else if(L.editingZone!==null&&L.editingZone>index)L.editingZone--;$('live-zone-name').value=L.editingZone!==null?(draftName===editingName?L.zones[L.editingZone].name:draftName):ZoneNavigation.nextName(L.zones);paintLiveZones()});
  $('live-zone-list').querySelectorAll('[data-live-edit]').forEach(el=>el.onclick=()=>{if(liveSetupLocked())return;const index=Number(el.dataset.liveEdit),z=L.zones[index];L.editingZone=index;L.points=structuredClone(z.points);$('live-zone-name').value=z.name;$('live-zone-kind').value=z.kind||'danger_zone';paintLiveZones();focusLiveZones();toast('Зона открыта для изменения. Удалите нужные точки по номерам, затем сохраните зону.')});
  updateLiveZoneControls();
};
$('live-zone-canvas').onclick=e=>{
  if(liveSetupLocked())return;
  if(!L.image)return toast('Сначала получите кадр камеры');
  if(L.points.length>=32)return toast('Не более 32 точек на зону');
  const r=e.target.getBoundingClientRect();L.points.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);paintLiveZones();
};
const previousSaveLiveZone=$('live-save-zone').onclick;
$('live-save-zone').onclick=()=>{const count=L.zones.length;previousSaveLiveZone();if(L.zones.length>count)toast('Зона сохранена и включена. Можно добавить ещё одну или начать эфир.')};
const previousLiveStart=$('live-start').onclick;
$('live-start').onclick=()=>{
  if(L.points.length||L.editingZone!==null){focusLiveZones();toast('Сначала сохраните новую или изменённую зону. Чтобы оставить прежние правила, отмените изменение.');return}
  if(isChecked('live-zone-on')&&!L.zones.length){focusLiveZones();toast('Ниже отметьте минимум 3 точки на кадре и нажмите «Сохранить зону».');return}
  if(isChecked('live-extra-harness')&&!L.zones.some(z=>z.kind==='height')){focusLiveZones();toast('Выбран сценарий «Работа на высоте без привязи». Добавьте зону с назначением «Работа на высоте», чтобы указать участок проверки, или отключите этот сценарий.');return}
  return previousLiveStart();
};
const previousLiveStatus=renderLiveStatus;
renderLiveStatus=function(){
  previousLiveStatus();$('live').classList.toggle('live-active',liveSetupLocked());
  const activeSession=liveSetupLocked()?L.current?.id:null;
  if(activeSession&&activeSession!==setupActiveSession)LiveLayout.closeSetup();
  setupActiveSession=activeSession;
  if(L.browserSession&&L.current?.id===L.browserSession&&['stopped','failed'].includes(L.current.state))stopWebcam();
  renderCameraControls();updateLiveZoneControls();
};
const previousLiveNewSetup=$('live-new').onclick;
$('live-new').onclick=()=>{if(!L.browserSession)stopWebcam();previousLiveNewSetup();paintLiveZones();renderCameraControls();if(L.drafting&&!L.current)LiveLayout.openSetup()};
const previousLiveStop=$('live-stop').onclick;
$('live-stop').onclick=()=>{if(L.current?.id===L.browserSession)stopWebcam();return previousLiveStop()};
window.addEventListener('pagehide',()=>stopWebcam());
cameraMode();paintLiveZones();

function renderLivePointTools(){
 if(!$('live-point-tools'))$('live-zone-canvas').closest('.live-zone-stage').insertAdjacentHTML('afterend','<div id="live-point-tools" class="zone-point-tools"><p class="hint">Любую вершину можно удалить кнопкой с её номером или правой кнопкой по точке на кадре. Для сохранённой зоны сначала выберите «Изменить».</p><div id="live-point-list" class="zone-point-list"></div><button id="live-edit-cancel" type="button" class="btn ghost small" hidden>Отменить изменение</button></div>');
 const pointSignature=String(L.points.length);if($('live-point-list').dataset.signature!==pointSignature){$('live-point-list').dataset.signature=pointSignature;$('live-point-list').innerHTML=L.points.map((_,i)=>`<button type="button" class="btn ghost small" data-live-point="${i}" aria-label="Удалить точку ${i+1}">Точка ${i+1} · удалить</button>`).join('');
 $('live-point-list').querySelectorAll('[data-live-point]').forEach(el=>el.onclick=()=>{if(liveSetupLocked())return;L.points=ZoneNavigation.removePoint(L.points,Number(el.dataset.livePoint));paintLiveZones()});}
 $('live-edit-cancel').hidden=L.editingZone===null;$('live-edit-cancel').onclick=()=>{if(liveSetupLocked())return;resetLiveZoneDraft();paintLiveZones()};
}
$('live-zone-canvas').addEventListener('contextmenu',e=>{e.preventDefault();if(liveSetupLocked())return;const r=e.target.getBoundingClientRect(),point=[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height],index=ZoneNavigation.nearestPoint(L.points,point,r.width,r.height);if(index>=0){L.points=ZoneNavigation.removePoint(L.points,index);paintLiveZones()}else toast('Нажмите на вершину незавершённого полигона. Для сохранённой зоны сначала выберите «Изменить».')});
