/* Navigation changes presentation only. Server permissions remain authoritative. */
(function(){
 const nav=document.querySelector('aside nav');if(!nav)return;
 const primary=document.createElement('div');primary.className='product-primary-nav';
 const more=document.createElement('details');more.id='workspace-tools';more.className='product-tools';
 more.innerHTML='<summary>Управление и справка</summary><div class="product-tools-content"></div>';
 const tools=more.querySelector('div');
 const eventNav=document.createElement('nav');eventNav.id='event-navigation';eventNav.className='event-navigation';eventNav.setAttribute('aria-label','Разделы событий');
 const eventPages=['journal','triage','clips'];
 const eventButton=document.createElement('button');eventButton.id='product-events-nav';eventButton.type='button';eventButton.innerHTML='<i>☷</i><span>События</span>';
 eventButton.setAttribute('aria-label','События');
 const badge=$('new-badge');if(badge)eventButton.append(badge);
 eventButton.onclick=()=>{const page=eventPages.find(p=>WorkspaceAccess.visible(A.user,p));if(page)showPage(page)};
 for(const button of Array.from(nav.querySelectorAll('button[data-page]'))){
  const page=button.dataset.page;
  button.querySelector('i')?.setAttribute('aria-hidden','true');
  if(typeof titles!=='undefined'&&titles[page])button.setAttribute('aria-label',titles[page]);
  if(eventPages.includes(page))eventNav.append(button);
  else if(['live','analytics','safety'].includes(page))primary.append(button);
  else tools.append(button);
 }
 const analytics=primary.querySelector('[data-page="analytics"]');primary.insertBefore(eventButton,analytics);
 nav.replaceChildren(primary,more);
 document.querySelector('main header').after(eventNav);
 const primaryEvents=new Set(eventPages);
 function sync(){
  const user=window.A?.user;
  eventButton.hidden=!eventPages.some(p=>WorkspaceAccess.visible(user,p));
  const inEvents=primaryEvents.has(S.page);eventButton.classList.toggle('on',inEvents);
  eventNav.hidden=!inEvents;
  eventNav.querySelectorAll('[data-page]').forEach(el=>el.classList.toggle('on',el.dataset.page===S.page));
  more.hidden=!Array.from(tools.querySelectorAll('[data-page]')).some(el=>!el.hidden);
  more.classList.toggle('has-current-page',Boolean(tools.querySelector(`[data-page="${S.page}"]`)));
  const originalOverview=tools.querySelector('[data-page="overview"]');if(originalOverview)originalOverview.classList.add('product-recordings-link');
  document.body.dataset.productSection=inEvents?'events':S.page;
 }
 const permissionBase=applyPermissions;applyPermissions=function(){permissionBase();sync()};
 const showPageBase=showPage;showPage=function(page){showPageBase(page);sync()};
 document.addEventListener('linq:languagechange',()=>{
  sync();
  // Rebuild language-dependent durations; retain protected camera/employee data.
  lastJobSignature='';renderJobs();eventSignatures={};
  if(S.page==='overview'&&S.current&&S.current.status!=='ready')renderResult();
  if(S.page==='journal')renderJournal();if(S.page==='triage')renderTriage();
  if(typeof L!=='undefined'){L.clipRenderSignature='';if(S.page==='clips')renderClips();}
  const episode=$('modal-episode');if(episode)episode.dataset.signature='';updateOpenEpisode();
 });
 // Keyboard and screen-reader users retain the original permission-checked buttons.
 more.addEventListener('click',event=>{if(event.target.closest('button[data-page]')&&innerWidth<=850)more.open=false});
 sync();
})();
