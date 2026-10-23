// Setup controls: camera preview and the always-visible zone editor.
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
  $('live-zone-details').scrollIntoView({behavior:'smooth',block:'start'});
  $('live-zone-details').classList.add('zone-attention');
  setTimeout(()=>$('live-zone-details').classList.remove('zone-attention'),1800);
}
$('live-edit-zones').onclick=focusLiveZones;
$('live-zone-get-frame').onclick=()=>$('live-preview').click();
$('live-zone-on').onchange=()=>{updateLiveZoneControls();if($('live-zone-on').checked)focusLiveZones()};
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
  const locked=liveSetupLocked(),hasFrame=Boolean(L.image),points=L.points.length;
  $('live-zone-placeholder').hidden=hasFrame;$('live-zone-canvas').hidden=!hasFrame;
  $('live-zone-count').textContent=`${L.zones.length} / 8 зон`;
  $('live-zone-progress').textContent=locked?'Правила закреплены на время эфира':points>=3?`${points} точек · можно сохранить зону`:`${points} точек · нужно не менее 3`;
  $('live-zone-enabled').textContent=$('live-zone-on').checked?'Размеченные зоны включены':'Зоны выключены. После сохранения первой зоны включатся автоматически.';
  $('live-save-zone').disabled=locked||!hasFrame||points<3;
  $('live-zone-undo').disabled=locked||!points;$('live-zone-clear').disabled=locked||!points;
  $('live-zone-get-frame').disabled=locked;
  if($('live-zone-kind'))$('live-zone-kind').disabled=locked;
  $('live-zone-canvas').setAttribute('aria-disabled',String(locked));
  $('live-zone-list').querySelectorAll('button').forEach(el=>el.disabled=locked);
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
    L.zones.forEach(z=>polygon(z.points,'#c7ef8a',z.name));polygon(L.points,'#ffda7a','Новая зона');
  }
  $('live-zone-list').innerHTML=L.zones.length?L.zones.map((z,i)=>`<div class="zone-item"><span><b>${esc(z.name)}</b><small>${esc(zoneKinds[z.kind||'danger_zone'])} · ${z.points.length} точки</small></span><button data-live-remove="${i}" aria-label="Удалить ${esc(z.name)}">✕</button></div>`).join(''):'<p class="hint zone-empty">Сохранённых зон пока нет.</p>';
  $('live-zone-list').querySelectorAll('[data-live-remove]').forEach(el=>el.onclick=()=>{if(liveSetupLocked())return;L.zones.splice(Number(el.dataset.liveRemove),1);paintLiveZones()});
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
  if($('live-zone-on').checked&&!L.zones.length){focusLiveZones();toast('Ниже отметьте минимум 3 точки на кадре и нажмите «Сохранить зону».');return}
  if($('live-extra-harness').checked&&!L.zones.some(z=>z.kind==='height')){focusLiveZones();toast('Добавьте зону с назначением «Работа на высоте».');return}
  return previousLiveStart();
};
const previousLiveStatus=renderLiveStatus;
renderLiveStatus=function(){
  previousLiveStatus();$('live').classList.toggle('live-active',liveSetupLocked());
  if(L.browserSession&&L.current?.id===L.browserSession&&['stopped','failed'].includes(L.current.state))stopWebcam();
  renderCameraControls();updateLiveZoneControls();
};
const previousLiveNewSetup=$('live-new').onclick;
$('live-new').onclick=()=>{if(!L.browserSession)stopWebcam();previousLiveNewSetup();paintLiveZones();renderCameraControls()};
const previousLiveStop=$('live-stop').onclick;
$('live-stop').onclick=()=>{if(L.current?.id===L.browserSession)stopWebcam();return previousLiveStop()};
window.addEventListener('pagehide',()=>stopWebcam());
cameraMode();paintLiveZones();
