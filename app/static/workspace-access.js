(function(root,factory){
  const access=factory();
  if(typeof module==='object'&&module.exports)module.exports=access;
  else root.WorkspaceAccess=access;
})(typeof window!=='undefined'?window:this,function(){
  const permissions={live:'live.read',overview:'analysis.run',journal:'events.read',clips:'evidence.read',triage:'events.review',analytics:'analytics.read',team:'members.manage',audit:'audit.read'};
  function inEnterprise(user){return Boolean(user?.organization?.status==='active'&&!user.must_change)}
  function inCabinet(user){return Boolean(user?.platform_admin&&!user.organization_id&&!user.must_change)}
  function has(user,permission){return Boolean(user?.permissions?.includes(permission))}
  function home(user){
    if(!user||user.must_change)return 'account';
    if(inCabinet(user))return 'platform';
    if(!inEnterprise(user))return 'account';
    if(user.role==='worker')return 'safety';
    if(has(user,'live.read'))return 'live';
    if(has(user,'analysis.run'))return 'overview';
    if(has(user,'analytics.read'))return 'analytics';
    if(has(user,'events.read'))return 'journal';
    return 'account';
  }
  function visible(user,page){
    if(page==='account')return Boolean(user);
    if(!user||user.must_change)return false;
    if(page==='platform')return inCabinet(user);
    if(inCabinet(user))return page==='audit';
    if(!inEnterprise(user))return false;
    if(page==='safety')return user.role==='worker';
    if(['quality','help'].includes(page))return user.role!=='worker';
    return permissions[page]?has(user,permissions[page]):false;
  }
  function canOpen(user,page,teamOrganization){
    return visible(user,page)||Boolean(page==='team'&&inCabinet(user)&&teamOrganization);
  }
  return {home,visible,canOpen,inCabinet,inEnterprise};
});
