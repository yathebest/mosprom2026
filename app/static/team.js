function memberEndpoint(orgId){return orgId?`/api/platform/organizations/${encodeURIComponent(orgId)}/members`:'/api/members'}
function openEnterpriseTeam(orgId){A.teamOrgId=orgId;showPage('team')}
async function renderTeam(target){
  const orgId=WorkspaceAccess.inCabinet(A.user)?A.teamOrgId:null;
  const version=++A.teamRenderId,endpoint=memberEndpoint(orgId);
  let data;
  if(orgId)data=await api(endpoint);
  else{
    const [members,cameras]=await Promise.all([api(endpoint),api('/api/cameras')]);
    const active=members.filter(m=>m.active).length;
    data={members,cameras,organization:A.user.organization,counts:{active,inactive:members.length-active,total:members.length}};
  }
  if(version!==A.teamRenderId||S.page!=='team')return;
  A.members=data.members;A.teamCameras=data.cameras;
  const context={orgId,endpoint,cameras:data.cameras};
  const heading=$('team').querySelector('.heading');
  heading.querySelector('h1').textContent=`Сотрудники · ${data.organization.name}`;
  heading.querySelector('.eyebrow').textContent=orgId?'КАБИНЕТ АДМИНИСТРАТОРА':'КОМАНДА ПРЕДПРИЯТИЯ';
  $('context').textContent=`Сотрудники · ${data.organization.name}`;
  let back=$('team-back');
  if(!back){back=document.createElement('button');back.id='team-back';back.className='btn ghost';back.textContent='← К предприятиям';heading.append(back)}
  back.hidden=!orgId;back.onclick=()=>showPage('platform');
  target.innerHTML=`${kpis([['Активные сотрудники',data.counts.active,'имеют доступ в рамках роли'],['Доступ отключён',data.counts.inactive,'можно восстановить'],['Всего сотрудников',data.counts.total,'в этом предприятии']])}<div class="security-layout"><div><article class="security-card"><h2>Команда предприятия</h2><p class="security-muted">${orgId?'Вы управляете этим предприятием из кабинета администратора платформы. ':''}Отключение доступа завершает сессии сотрудника; история его событий сохраняется.</p><div class="security-list">${data.members.map(m=>`<div class="security-member"><strong>${esc(m.name)} ${m.active?'':'· доступ отключён'}</strong><small>${esc(m.login)} · ${esc(roleName(m.role))}<br>${m.camera_scope===null?'Все камеры предприятия':'Камеры: '+esc(JSON.parse(m.camera_scope).join(', ')||'не назначены')}</small><div class="actions"><button class="btn ghost small" data-edit-member="${m.id}">Роль и доступ</button>${m.id!==A.user.id?`<button class="btn ghost small" data-reset-member="${m.id}">Сбросить пароль</button><button class="btn ghost small" data-toggle-member="${m.id}">${m.active?'Удалить из команды':'Восстановить'}</button>`:''}</div></div>`).join('')||'<div class="security-empty">Сотрудников пока нет. Добавьте первого локального администратора или оператора.</div>'}</div></article></div><div><article class="security-card"><h2>Добавить сотрудника</h2><div id="member-form"></div></article><article class="security-card"><h2>Кто за что отвечает</h2>${A.roles.map(r=>`<div class="security-role"><strong>${esc(r.name)}</strong><small>${esc(r.permissions.map(p=>permissionNames[p]).join(' · ')||'Памятка по безопасности; без доступа к видеозаписям')}</small></div>`).join('')}</article></div></div>`;
  memberForm($('member-form'),null,context);
  target.querySelectorAll('[data-edit-member]').forEach(el=>el.onclick=()=>{
    const member=data.members.find(m=>m.id===el.dataset.editMember);
    $('modal-title').textContent='Роль и доступ сотрудника';$('modal-body').innerHTML='<div class="security-card" id="edit-member-form"></div>';
    memberForm($('edit-member-form'),member,context);$('modal').showModal();
  });
  target.querySelectorAll('[data-toggle-member]').forEach(el=>el.onclick=async()=>{
    const member=data.members.find(m=>m.id===el.dataset.toggleMember);el.disabled=true;
    try{
      await api(`${endpoint}/${member.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:member.name,role:member.role,cameras:member.camera_scope===null?null:JSON.parse(member.camera_scope),active:!member.active})});
      if(S.page==='team'&&A.teamOrgId===orgId)await renderTeam(target);toast(member.active?'Сотрудник удалён из команды. Сессии отозваны.':'Доступ восстановлен.');
    }catch(error){toast(error.message);el.disabled=false}
  });
  target.querySelectorAll('[data-reset-member]').forEach(el=>el.onclick=()=>resetMemberForm(data.members.find(m=>m.id===el.dataset.resetMember),context));
  applyPermissions();
}
function memberForm(target,member,context){
  const chosen=member?.camera_scope?JSON.parse(member.camera_scope):[];
  target.innerHTML=`<form class="security-form"><label>Имя<input name="name" required minlength="2" maxlength="120" value="${esc(member?.name||'')}"></label>${member?'':'<label>Логин или email<input name="login" required minlength="3" maxlength="120" autocomplete="off"></label><label>Временный пароль, от 12 символов<input name="password" type="password" required minlength="12" maxlength="256" autocomplete="new-password"></label>'}<label>Роль<select name="role">${A.roles.map(r=>`<option value="${r.key}" ${(member?.role||'operator')===r.key?'selected':''}>${esc(r.name)}</option>`).join('')}</select></label><label>Доступ к камерам<select name="scope"><option value="all" ${member?.camera_scope===null||!member?'selected':''}>Все камеры предприятия</option><option value="selected" ${member?.camera_scope!==null&&member?'selected':''}>Только выбранные камеры</option></select></label><div class="security-camera-list">${context.cameras.map(camera=>`<label><input type="checkbox" name="camera" value="${esc(camera.name)}" ${chosen.includes(camera.name)?'checked':''}>${esc(camera.name)}</label>`).join('')||'<p class="security-muted">Сначала сохраните камеру и её правила.</p>'}</div><p class="security-muted">Мастеру назначаются камеры участка. Администратор и специалист по охране труда управляют всеми камерами предприятия. Сотрудник сменит временный пароль при первом входе.</p><div class="auth-error" hidden></div><button class="btn primary" type="submit">${member?'Сохранить изменения':'Добавить сотрудника'}</button></form>`;
  const form=target.querySelector('form');
  const sync=()=>{const role=form.elements.role.value;if(role==='foreman')form.elements.scope.value='selected';if(['enterprise_admin','safety_specialist'].includes(role))form.elements.scope.value='all';form.elements.scope.disabled=['foreman','enterprise_admin','safety_specialist'].includes(role);target.querySelector('.security-camera-list').hidden=form.elements.scope.value==='all'};
  form.elements.role.onchange=sync;form.elements.scope.onchange=sync;sync();
  form.onsubmit=async e=>{
    e.preventDefault();const fields=new FormData(form),errorBox=target.querySelector('.auth-error'),button=form.querySelector('button[type=submit]');
    const data={name:fields.get('name'),role:fields.get('role'),cameras:form.elements.scope.value==='all'?null:fields.getAll('camera')};
    button.disabled=true;errorBox.hidden=true;
    try{
      if(member){
        await api(`${context.endpoint}/${member.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({...data,active:Boolean(member.active)})});
        if(member.id===A.user.id)return location.reload();$('modal').close();
      }else await post(context.endpoint,{...data,login:fields.get('login'),password:fields.get('password')});
      toast(member?'Роль и доступ обновлены.':'Сотрудник добавлен. Передайте временный пароль лично.');
      if(S.page==='team'&&A.teamOrgId===context.orgId)await renderSecurityPage('team');
    }catch(error){errorBox.textContent=error.message;errorBox.hidden=false;button.disabled=false}
  };
}
function resetMemberForm(member,context){
  $('modal-title').textContent='Новый временный пароль';
  $('modal-body').innerHTML=`<div class="security-card"><h2>${esc(member.name)}</h2><form id="reset-form" class="security-form"><label>Временный пароль<input name="password" type="password" required minlength="12" maxlength="256" autocomplete="new-password"></label><p>Текущие сессии сотрудника будут завершены. При следующем входе потребуется сменить пароль.</p><div id="reset-error" class="auth-error" hidden></div><button class="btn primary">Сбросить пароль</button></form></div>`;
  $('modal').showModal();$('reset-form').onsubmit=async e=>{e.preventDefault();try{await post(`${context.endpoint}/${member.id}/password`,{password:new FormData(e.currentTarget).get('password')});$('modal').close();toast('Пароль сброшен. Передайте новый пароль сотруднику лично.')}catch(error){$('reset-error').textContent=error.message;$('reset-error').hidden=false}};
}
