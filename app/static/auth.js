// Permissions mirror the server catalog; hiding a control never grants access.
window.A={user:null,workspace:false,roles:[],members:[],organizations:[],teamOrgId:null,teamCameras:[],teamRenderId:0};
function can(permission){return Boolean(A.user?.permissions.includes(permission))}
document.addEventListener('linq:healthready',event=>{
  if(!A.workspace)return;
  S.health=event.detail;
  // A timed-out optional health request must be recoverable without resetting
  // an already configured camera or adding another refresh interval.
  if(!Object.keys(S.defaults||{}).length)init();
});
document.addEventListener('linq:languagechange',()=>{
  // Audit targets combine localized system labels and protected user names.
  // Rebuild this read-only view with the current locale instead of translating
  // protected spans, which would also risk changing enterprise/employee names.
  if(A.user&&S.page==='audit'&&(A.user.platform_admin||can('audit.read'))){
    renderSecurityPage('audit').catch(error=>toast(error.message));
  }
});
const nativeFetch=window.fetch.bind(window);
window.fetch=(input,options={})=>{
  const url=new URL(typeof input==='string'?input:input.url,location.href);
  if(url.origin===location.origin&&A.user&&!['GET','HEAD'].includes((options.method||'GET').toUpperCase())){
    const headers=new Headers(options.headers);headers.set('X-CSRF-Token',A.user.csrf);options={...options,headers};
  }
  return nativeFetch(input,options);
};
const ordinaryShowPage=showPage;
showPage=function(page){
  if(!WorkspaceAccess.canOpen(A.user,page,A.teamOrgId))return;
  if(page==='platform')A.teamOrgId=null;
  ordinaryShowPage(page);
  applyPermissions();
  if(['platform','team','audit','account'].includes(page))renderSecurityPage(page).catch(e=>toast(e.message));
};
function addPage(id,title,icon){
  titles[id]=title;const section=document.createElement('section');section.id=id;section.className='page';
  section.innerHTML=`<div class="heading"><div><p class="eyebrow">${id==='platform'?'КАБИНЕТ АДМИНИСТРАТОРА':id==='team'?'КОМАНДА ПРЕДПРИЯТИЯ':id==='audit'?'ИСТОРИЯ ДЕЙСТВИЙ':'ЛИЧНЫЙ АККАУНТ'}</p><h1>${title}</h1></div></div><div id="${id}-content"></div>`;
  document.querySelector('main footer').before(section);
  const button=document.createElement('button');button.dataset.page=id;button.innerHTML=`<i>${icon}</i>${title}`;button.onclick=()=>showPage(id);document.querySelector('nav').append(button);
}
addPage('platform','Предприятия','▦');addPage('team','Сотрудники и роли','♙');addPage('audit','Журнал действий','≡');addPage('account','Аккаунт','○');addPage('safety','Безопасность на участке','◇');
const authRoot=document.createElement('div');authRoot.id='auth-root';document.body.append(authRoot);
function authScreen(register=false,message=''){
  document.body.classList.remove('authenticated');authRoot.hidden=false;
  authRoot.innerHTML=`<div class="auth-intro"><p class="eyebrow">LINQ SAFETY / ПРОМЫШЛЕННАЯ БЕЗОПАСНОСТЬ</p><h1>Одно пространство.<br>Своя команда.<br>Чёткие права.</h1><p>Камеры, тревоги и доказательства доступны сотрудникам вашего предприятия в рамках их обязанностей.</p><div class="auth-points"><span>01 &nbsp; Заявка предприятия и проверка администратором</span><span>02 &nbsp; Назначение сотрудников и камер участка</span><span>03 &nbsp; Разбор происшествий с историей действий</span></div></div><div class="auth-card"><div class="auth-tabs"><button id="auth-login-tab" class="${register?'':'selected'}">Вход</button><button id="auth-register-tab" class="${register?'selected':''}">Заявка предприятия</button></div><h2>${register?'Подключить предприятие':'Войти в рабочее пространство'}</h2><p class="security-muted">${register?'Создайте аккаунт локального администратора. Камеры и архив откроются после одобрения заявки.':'Используйте аккаунт, созданный администратором вашего предприятия.'}</p><div id="auth-error" class="auth-error" ${message?'':'hidden'}>${esc(message)}</div><form id="auth-form">${register?'<label>Название предприятия<input name="enterprise" required minlength="2" maxlength="180" autocomplete="organization"></label><label>Ваше имя<input name="name" required minlength="2" maxlength="120" autocomplete="name"></label>':''}<label>Логин или email<input name="login" required minlength="3" maxlength="120" autocomplete="username"></label><label>Пароль<input name="password" type="password" required minlength="${register?12:1}" maxlength="256" autocomplete="${register?'new-password':'current-password'}"></label>${register?'<label>Описание предприятия и контакт для связи<textarea name="note" maxlength="2000"></textarea></label>':''}<button class="btn primary" type="submit">${register?'Отправить заявку →':'Войти →'}</button></form><p id="auth-bootstrap" class="security-muted" hidden>Первый запуск: создайте администратора платформы в терминале из папки проекта командой <code>python scripts/create_admin.py</code>. Пароль вводится скрыто. Затем войдите здесь.</p></div>`;
  $('auth-login-tab').onclick=()=>authScreen(false);$('auth-register-tab').onclick=()=>authScreen(true);
  $('auth-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget,button=form.querySelector('button');button.disabled=true;$('auth-error').hidden=true;try{await post(register?'/api/auth/register':'/api/auth/login',Object.fromEntries(new FormData(form)));location.reload()}catch(error){$('auth-error').textContent=error.message;$('auth-error').hidden=false;button.disabled=false}};
  api('/api/auth/status').then(s=>{if($('auth-bootstrap'))$('auth-bootstrap').hidden=s.initialized}).catch(()=>{});
}
function applyPermissions(){
  window.UiStatus?.context(A.user,A.workspace);
  document.querySelectorAll('[data-page]').forEach(el=>el.hidden=!WorkspaceAccess.visible(A.user,el.dataset.page));
  if(S.page==='team'&&WorkspaceAccess.inCabinet(A.user))document.querySelector('nav [data-page="platform"]').classList.add('on');
  document.querySelectorAll('[data-review]').forEach(el=>el.disabled=!can('events.review'));
  document.querySelectorAll('a[href^="/api/export"],a[href^="/api/review-dataset"],a[href$="/report"]').forEach(el=>el.hidden=!can('export'));
  document.querySelectorAll('.live-controls').forEach(el=>el.hidden=!can('live.manage'));
  ['live-new','live-stop','live-start','live-preview','live-disconnect'].forEach(id=>{if($(id)&&!can('live.manage'))$(id).hidden=true});
  document.querySelector('.mode-switch').hidden=!A.workspace||['platform','team','audit','account','safety'].includes(S.page)||(!can('analysis.run')&&!can('live.read'));
  document.body.classList.toggle('can-manage-live',can('live.manage'));
  $('live-idle').querySelector('p').textContent=can('live.manage')?'Подключите источник и нажмите «Начать эфир».':'Активных эфиров пока нет. Подключить источник может администратор или специалист по охране труда.';
}
const orgStatuses={pending:'На рассмотрении',active:'Подключено',rejected:'Отклонено',suspended:'Приостановлено'};
const permissionNames={'events.read':'Журнал и тревоги','live.read':'Просмотр эфиров','jobs.read':'Записи камер','analytics.read':'Статистика','evidence.read':'Кадры и клипы','export':'Экспорт доказательств','members.manage':'Сотрудники и роли','rules.manage':'Правила и зоны','analysis.run':'Анализ видео','live.manage':'Подключение и остановка камер','events.review':'Разбор событий','audit.read':'История действий'};
function roleName(key){return A.roles.find(r=>r.key===key)?.name||key}
function pendingPanel(){const org=A.user.organization;return `<article class="security-card"><span class="security-status ${esc(org?.status)}">${esc(orgStatuses[org?.status]||'Без предприятия')}</span><h2>${esc(org?.name||'Администратор платформы')}</h2><p>${org?.status==='pending'?'Заявка отправлена. Доступ к камерам и архиву появится после одобрения администратором платформы.':org?.status==='rejected'?'Заявка отклонена. Уточните причину и отправьте её повторно.':org?.status==='suspended'?'Доступ предприятия приостановлен. Обратитесь к администратору платформы.':'Выберите предприятие в разделе «Предприятия», чтобы открыть его рабочее пространство.'}</p>${org?.reason?`<p class="security-note">${esc(org.reason)}</p>`:''}${org?.status==='rejected'?'<button id="reapply" class="btn primary">Отправить повторно</button>':''}</article>`}
async function renderSecurityPage(page){
  const target=$(page+'-content');
  if(page==='platform'){
    A.organizations=await api('/api/platform/organizations');
    const pending=A.organizations.filter(o=>o.status==='pending').length;
    target.innerHTML=`${kpis([['Заявки',pending,'ожидают решения'],['Активные',A.organizations.filter(o=>o.status==='active').length,'предприятия'],['Всего',A.organizations.length,'рабочие пространства']])}<div class="security-note">Администратор платформы рассматривает заявки. Для работы с данными выберите конкретное предприятие: переход сохраняется в журнале действий.</div><br><div class="security-list">${A.organizations.map(o=>`<article class="security-card"><span class="security-status ${esc(o.status)}">${esc(orgStatuses[o.status])}</span><h2 data-i18n-ignore>${esc(o.name)}</h2><p class="security-muted">${UiFormat.time(o.created)} · Сотрудников: ${o.members} активных / ${o.members_total??o.members} всего${o.members_inactive?' · отключено: '+o.members_inactive:''}</p><p data-i18n-ignore>${esc(o.note)}</p>${o.reason?`<p class="security-note">Причина: ${esc(o.reason)}</p>`:''}<div class="actions"><button class="btn ghost" data-members="${o.id}">Сотрудники · ${o.members_total??o.members}</button>${o.status==='pending'?`<button class="btn primary" data-decision="${o.id}" data-status="active">Принять заявку</button><button class="btn ghost" data-decision="${o.id}" data-status="rejected">Отклонить</button>`:o.status==='active'?`<button class="btn primary" data-context="${o.id}">Открыть пространство →</button><button class="btn ghost" data-decision="${o.id}" data-status="suspended">Приостановить доступ</button>`:o.status==='suspended'?`<button class="btn primary" data-decision="${o.id}" data-status="active">Возобновить доступ</button>`:''}</div></article>`).join('')}</div>`;
    target.querySelectorAll('[data-context]').forEach(el=>el.onclick=()=>changeEnterpriseContext(el.dataset.context));
    target.querySelectorAll('[data-members]').forEach(el=>el.onclick=()=>openEnterpriseTeam(el.dataset.members));
    target.querySelectorAll('[data-decision]').forEach(el=>el.onclick=()=>decisionForm(el.dataset.decision,el.dataset.status));
    await renderInvitationsPanel(target);
  }else if(page==='team'){
    await renderTeam(target);
  }else if(page==='audit'){
    const rows=await api('/api/audit');target.innerHTML=`<article class="security-card"><p class="security-muted">Последние 300 действий. Смена прав, решения по заявкам и результаты проверки событий сохраняются с автором и временем.</p><div class="security-table"><table class="security-audit"><thead><tr><th>Местное время</th><th>Сотрудник</th><th>Действие</th><th>Подробности</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${UiFormat.time(r.created)}</td><td data-i18n-ignore>${esc(r.actor_name||'Система')}</td><td>${esc(auditAction(r.action))}</td><td>${UiFormat.auditDetails(r,A.roles,labels)}</td></tr>`).join('')}</tbody></table></div></article>`;
  }else if(page==='account'){
    target.innerHTML=`<div class="security-layout"><div>${!A.workspace?pendingPanel():''}<article class="security-card"><h2 data-i18n-ignore>${esc(A.user.name)}</h2><p><span data-i18n-ignore>${esc(A.user.login)}</span><br>${A.user.platform_admin?'Администратор платформы':esc(roleName(A.user.role))}<br>${esc(A.user.organization?.name||'Управление платформой')}</p><p class="security-muted">${A.user.cameras===null?'Доступ в рамках роли и предприятия.':'Назначенные камеры: '+esc(A.user.cameras.join(', ')||'нет')}</p><button id="logout" class="btn ghost">Выйти из аккаунта</button></article></div><article class="security-card"><h2>${A.user.must_change?'Задайте постоянный пароль':'Изменить пароль'}</h2><p class="security-muted">${A.user.must_change?'Администратор выдал временный пароль. Смените его, чтобы открыть рабочее пространство.':'После изменения все остальные сессии будут завершены.'}</p><form id="password-form" class="security-form"><label>Текущий пароль<input name="current" type="password" required autocomplete="current-password"></label><label>Новый пароль, от 12 символов<input name="password" type="password" required minlength="12" maxlength="256" autocomplete="new-password"></label><label>Повтор нового пароля<input name="repeat" type="password" required minlength="12" autocomplete="new-password"></label><div class="auth-error" id="password-error" hidden></div><button type="submit" class="btn primary">Сохранить пароль</button></form></article></div>`;
    $('logout').onclick=async()=>{stopWebcam();$('live-feed').removeAttribute('src');$('result-video').pause();try{await post('/api/auth/logout');location.reload()}catch(e){toast(e.message)}};
    $('password-form').onsubmit=async e=>{e.preventDefault();const f=Object.fromEntries(new FormData(e.currentTarget));try{if(f.password!==f.repeat)throw Error('Новые пароли не совпадают');await post('/api/auth/password',{current:f.current,password:f.password});location.reload()}catch(error){$('password-error').textContent=error.message;$('password-error').hidden=false}};
    if($('reapply'))$('reapply').onclick=async()=>{try{await post('/api/account/reapply');location.reload()}catch(e){toast(e.message)}};
  }
  applyPermissions();
}
function auditAction(action){return ({...invitationAuditActions,'auth.login':'Вход','auth.password_changed':'Пароль изменён','platform.bootstrap':'Создан администратор платформы','platform.context':'Открыто пространство предприятия','organization.request':'Заявка предприятия','organization.active':'Предприятие подключено','organization.rejected':'Заявка отклонена','organization.suspended':'Доступ приостановлен','organization.reapply':'Повторная заявка','member.created':'Сотрудник добавлен','member.updated':'Роль / доступ изменены','member.password_reset':'Пароль сброшен','event.reviewed':'Событие проверено'})[action]||'Другое действие'}
function decisionForm(id,status){
  const org=A.organizations.find(o=>o.id===id);$('modal-title').textContent=({active:'Подключить предприятие',rejected:'Отклонить заявку',suspended:'Приостановить доступ'})[status];
  $('modal-body').innerHTML=`<div class="security-card"><h2>${esc(org.name)}</h2><form id="decision-form" class="security-form"><label>Комментарий ${status==='active'?'(необязательно)':'и причина решения'}<textarea name="reason" maxlength="1000" ${status==='active'?'':'required'}></textarea></label><div id="decision-error" class="auth-error" hidden></div><button class="btn primary">Сохранить решение</button></form></div>`;$('modal').showModal();
  $('decision-form').onsubmit=async e=>{e.preventDefault();try{await api('/api/platform/organizations/'+id,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({status,reason:new FormData(e.currentTarget).get('reason')})});$('modal').close();await renderSecurityPage('platform')}catch(error){$('decision-error').textContent=error.message;$('decision-error').hidden=false}};
}
async function changeEnterpriseContext(orgId){
  try{stopWebcam();await post('/api/platform/context',{organization_id:orgId});location.replace('/')}
  catch(error){toast(error.message)}
}
function renderWorkspacePanel(){
  const panel=document.createElement('div');panel.className='workspace-panel';panel.id='workspace-panel';
  const cabinet=WorkspaceAccess.inCabinet(A.user);
  panel.innerHTML=`<small>${cabinet?'ПЛАТФОРМА':'ТЕКУЩЕЕ ПРЕДПРИЯТИЕ'}</small><strong>${esc(cabinet?'Кабинет администратора':A.user.organization?.name||'Аккаунт')}</strong>${A.user.platform_admin&&A.user.organization_id?'<button id="open-admin-cabinet">← Кабинет администратора</button>':''}`;
  document.querySelector('aside .brand').after(panel);
  document.querySelector('.nav-caption').textContent=cabinet?'УПРАВЛЕНИЕ ПЛАТФОРМОЙ':'РАБОЧЕЕ ПРОСТРАНСТВО';
  if($('open-admin-cabinet'))$('open-admin-cabinet').onclick=()=>changeEnterpriseContext(null);
  document.querySelector('.brand').onclick=e=>{e.preventDefault();showPage(WorkspaceAccess.home(A.user))};
}
async function startAccount(){
  if(location.hash.startsWith('#invite=')){await showInvitationRegistration(location.hash.slice(8));return}
  try{A.user=await api('/api/auth/me')}catch(e){authScreen(false,e.message==='Войдите в аккаунт'?'':e.message);return}
  A.workspace=WorkspaceAccess.inEnterprise(A.user);
  if(!A.user.must_change)A.roles=await api('/api/roles');
  authRoot.innerHTML='<div class="auth-card"><h2>Открываем рабочее пространство…</h2><p>Проверяем права доступа и загружаем данные предприятия.</p></div>';
  const chip=document.createElement('div');chip.className='account-chip';chip.innerHTML=`<b data-i18n-ignore>${esc(A.user.name)}</b><span>${A.user.platform_admin?'Администратор платформы':esc(roleName(A.user.role))}</span><span data-i18n-ignore>${esc(A.user.organization?.name||'Управление платформой')}</span><button id="open-account" class="quiet">Аккаунт и выход →</button>`;document.querySelector('aside nav').after(chip);$('open-account').onclick=()=>showPage('account');
  renderWorkspacePanel();
  $('safety-content').innerHTML='<article class="security-card"><h2>Безопасность начинается до работы</h2><p>Используйте обязательные СИЗ для своего участка: каску, сигнальный жилет и другие средства защиты по инструкции предприятия. Соблюдайте ограждения и маршруты техники, не проходите под грузом или конвейером.</p><p>Сообщайте мастеру об опасности и неисправностях. При происшествии действуйте по утверждённому плану предприятия.</p><div class="security-note">Это общая памятка. Инструктаж и локальные правила предприятия имеют приоритет. Индивидуальные видеозаписи и журнал происшествий этой роли недоступны.</div></article>';
  applyPermissions();
  if(A.workspace)await init();
  document.body.classList.add('authenticated');authRoot.hidden=true;
  const home=WorkspaceAccess.home(A.user);
  showPage(home);if(A.workspace&&can('live.read'))pollLive();
  const signature=JSON.stringify([A.user.role,A.user.permissions,A.user.organization,A.user.cameras,A.user.must_change]);
  setInterval(async()=>{try{const user=await api('/api/auth/me');if(signature!==JSON.stringify([user.role,user.permissions,user.organization,user.cameras,user.must_change])){stopWebcam();location.reload()}}catch(e){}},15000);
}
startAccount().catch(e=>authScreen(false,e.message));
