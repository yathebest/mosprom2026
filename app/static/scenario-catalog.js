(function(root,factory){const catalog=factory();if(typeof module==='object'&&module.exports)module.exports=catalog;else root.ScenarioCatalog=catalog})(typeof window!=='undefined'?window:this,function(){
 const coreKinds=Object.freeze(['no_helmet','danger_zone','possible_fall','person_down']);
 const coreChoices=Object.freeze(['helmet','danger_zone','fall']);
 function newCameraConfig(defaults={}){
  const c={...defaults,zones:[],calibration:null,recorded_at:null,fall_detection:false,fast_motion:false,helmet_detection:true};
  for(const key of ['proximity','phone','fire','smoking','harness','load_posture','smoke','vest','mask'])c[key+'_detection']=false;
  return c;
 }
 return Object.freeze({coreKinds,coreChoices,newCameraConfig,isPrimary:kind=>coreKinds.includes(kind),isExperimental:kind=>!coreKinds.includes(kind)});
});
