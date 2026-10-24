// Shared display formatting. UTC stays in storage; people see their local time.
(function(root){
  'use strict';
  const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const language=()=>root.document?.documentElement?.lang==='en'?'en':'ru';
  const translate=ru=>root.LinQI18n?.t?.(ru)||ru;
  const locale=()=>language()==='en'?'en-GB':'ru-RU';
  function dateTime(value,options={}){
    if(value===null||value===undefined||value==='')return translate('Время не указано');
    const date=new Date(value);
    if(!Number.isFinite(date.getTime()))return translate('Время не указано');
    return new Intl.DateTimeFormat(options.locale||locale(),{day:'2-digit',month:'2-digit',year:'numeric',
      hour:'2-digit',minute:'2-digit',second:'2-digit',timeZoneName:'short',...options}).format(date);
  }
  function time(value){
    const date=new Date(value);
    if(!value||!Number.isFinite(date.getTime()))return escape(translate('Время не указано'));
    return `<time datetime="${escape(date.toISOString())}" data-local-time data-i18n-ignore>${escape(dateTime(value))}</time>`;
  }
  const rolesEn={enterprise_admin:'Enterprise administrator',safety_specialist:'Safety specialist',foreman:'Foreman / shift supervisor',operator:'Operator / dispatcher',worker:'Worker / driver',safety_legal:'Safety office / legal',management:'Management'};
  const values={pending:'На рассмотрении',active:'Подключено',rejected:'Отклонено',suspended:'Приостановлено',new:'Нужна проверка',confirmed:'Подтверждено',false:'Ложное',forwarded:'Передано мастеру',preview:'Имитация отправки',smtp:'SMTP'};
  function auditDetails(row,roleCatalog=[],eventLabels={}){
    let details=row.details_data;
    if(!details){try{details=JSON.parse(row.details||'{}')}catch(_){details={}}}
    if(!details||typeof details!=='object'||Array.isArray(details))details={};
    const context=row.target_context||{},parts=[];
    let targetLabel='',targetValue='';
    if(context.type==='member'){targetLabel='Сотрудник';targetValue=context.name||translate('Сотрудник больше не доступен');}
    else if(context.type==='organization'){targetLabel='Предприятие';targetValue=context.name||translate('Предприятие больше не доступно');}
    else if(context.type==='invitation'){targetLabel='Приглашение';targetValue=context.email||translate('Приглашение больше не доступно');}
    else if(context.type==='event'){targetLabel='Событие';targetValue=[translate(eventLabels[context.kind]||'Событие'),context.camera].filter(Boolean).join(' · ');}
    else if(context.type==='account'){targetLabel='Аккаунт';targetValue=context.name||row.actor_name||translate('Системный аккаунт');}
    else if(row.action==='platform.context'&&!row.target){targetLabel='Рабочее пространство';targetValue=translate('Кабинет администратора');}
    if(targetLabel)parts.push(`<div class="audit-target"><b>${escape(translate(targetLabel))}</b><span data-i18n-ignore>${escape(targetValue)}</span></div>`);
    const add=(label,value,user=false,isHTML=false)=>{parts.push(`<div><dt>${escape(translate(label))}</dt><dd${user?' data-i18n-ignore':''}>${isHTML?value:escape(value)}</dd></div>`)};
    if(typeof details.role==='string')add('Роль',language()==='en'?(rolesEn[details.role]||translate(roleCatalog.find(r=>r.key===details.role)?.name||'Другая роль')):(roleCatalog.find(r=>r.key===details.role)?.name||'Другая роль'));
    if(Object.hasOwn(details,'cameras'))add('Камеры',details.cameras===null?translate('Все камеры предприятия'):Array.isArray(details.cameras)&&details.cameras.length?details.cameras.join(', '):translate('Камеры не назначены'),Array.isArray(details.cameras)&&details.cameras.length>0);
    if(typeof details.active==='boolean')add('Доступ',translate(details.active?'Разрешён':'Отключён'));
    if(typeof details.from==='string')add('Предыдущий статус',translate(values[details.from]||'Другой статус'));
    if(typeof details.status==='string')add('Результат проверки',translate(values[details.status]||'Другой статус'));
    if(details.reason)add('Причина',String(details.reason),true);
    if(details.name)add('Название',String(details.name),true);
    if(details.email)add('Email',String(details.email),true);
    if(details.expires_at)add('Ссылка действует до',time(details.expires_at),false,true);
    if(details.delivery)add('Отправка',translate(values[details.delivery]||'Другой способ'));
    if(!parts.length)return `<span class="security-muted">${escape(translate('Действие сохранено'))}</span>`;
    const target=parts[0]?.startsWith('<div class="audit-target"')?parts.shift():'';
    return target+(parts.length?`<dl class="audit-details">${parts.join('')}</dl>`:'');
  }
  const UiFormat={dateTime,time,auditDetails,locale};root.UiFormat=UiFormat;
  if(typeof module!=='undefined'&&module.exports)module.exports=UiFormat;
  if(!root.document)return;

  const status={workspace:false,user:null,state:'idle',data:null,error:null};
  function render(){
    const element=root.document.getElementById('device');if(!element)return;
    let message;
    if(!status.workspace){
      message=status.user?.platform_admin?translate('Оборудование: выберите предприятие'):translate('Оборудование недоступно в этом разделе');
    }else if(status.state==='success'){
      message=`${translate('Локальная обработка')}: ${status.data.device||'CPU'}${status.data.model_ready?'':` · ${translate('модель не установлена')}`}`;
    }else if(status.state==='forbidden')message=translate('Проверка оборудования недоступна для этой роли');
    else if(status.state==='error')message=translate('Не удалось проверить оборудование');
    else message=translate('Проверяем оборудование…');
    element.textContent=message;element.setAttribute('role','status');element.setAttribute('aria-busy',String(status.workspace&&['idle','pending'].includes(status.state)));
    let retry=root.document.getElementById('device-retry');
    if(status.workspace&&status.state==='error'){
      if(!retry){retry=root.document.createElement('button');retry.id='device-retry';retry.className='device-retry';element.after(retry);retry.onclick=()=>check();}
      retry.textContent=translate('Повторить проверку');retry.hidden=false;
    }else if(retry)retry.hidden=true;
  }
  const nativeFetch=root.fetch.bind(root);
  async function watchedFetch(input,options={}){
    const url=new URL(typeof input==='string'?input:input.url,root.location.href);
    if(url.origin!==root.location.origin||url.pathname!=='/api/health')return nativeFetch(input,options);
    status.state='pending';render();
    const controller=new AbortController(),upstream=options.signal||(typeof input==='string'?null:input.signal);
    const abort=()=>controller.abort(upstream.reason);if(upstream?.aborted)abort();else upstream?.addEventListener('abort',abort,{once:true});
    const timer=setTimeout(()=>controller.abort(),12000);
    try{
      const response=await nativeFetch(input,{...options,signal:controller.signal});
      if(!response.ok){status.state=[401,403].includes(response.status)?'forbidden':'error';render();return response;}
      try{status.data=await response.clone().json();status.state='success';}catch(_){status.state='error';}
      render();return response;
    }catch(error){status.state='error';status.error=error;render();throw error;}
    finally{clearTimeout(timer);upstream?.removeEventListener('abort',abort);}
  }
  async function check(){
    try{
      await root.fetch('/api/health');
      if(status.state==='success')root.document.dispatchEvent(new root.CustomEvent('linq:healthready',{detail:status.data}));
    }catch(_){}finally{render();}
  }
  root.fetch=watchedFetch;
  root.UiStatus={context(user,workspace){status.user=user;status.workspace=Boolean(workspace);render();},refresh:check};
  root.document.addEventListener('linq:languagechange',()=>{
    render();root.document.querySelectorAll('time[data-local-time][datetime]').forEach(el=>{el.textContent=dateTime(el.getAttribute('datetime'));});
  });
})(typeof window!=='undefined'?window:globalThis);
