/* Shared, DOM-independent rules for the upload and live zone editors. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.ZoneNavigation=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const defaultName=/^(Опасная зона|Тестовая зона)(?:\s+\d+)?$/;
  function renumber(zones){
    const counts={};return zones.map(zone=>{const match=String(zone.name||'').trim().match(defaultName);
      return match?{...zone,name:`${match[1]} ${counts[match[1]]=(counts[match[1]]||0)+1}`}:{...zone};});
  }
  function nextName(zones,prefix='Опасная зона'){
    const used=new Set(zones.map(z=>z.name));let index=1;while(used.has(`${prefix} ${index}`))index++;return `${prefix} ${index}`;
  }
  function removePoint(points,index){return points.filter((_,i)=>i!==index);}
  function nearestPoint(points,point,width,height,radius=14){
    let found=-1,best=radius*radius;
    points.forEach((p,i)=>{const distance=((p[0]-point[0])*width)**2+((p[1]-point[1])*height)**2;
      if(distance<=best){found=i;best=distance;}});return found;
  }
  function wizardIssue(step,state){
    if(step===0&&!String(state.camera||'').trim())return 'Введите понятное название камеры на шаге «Источник».';
    if(step===1&&!state.anyScenario)return 'На шаге «Сценарии» выберите хотя бы одно правило контроля.';
    if(step===2){
      if(state.zoneEnabled&&!state.zones.length)return 'Выбран контроль зон. Сохраните полигон на шаге «Зоны» или отключите этот сценарий.';
      if(state.calibrationEnabled&&state.groundPoints!==4)return 'Для включённой калибровки нужны четыре угла пола. Завершите разметку или отключите калибровку.';
      if(state.harnessEnabled&&!state.zones.some(z=>z.kind==='height'))return 'Выбран сценарий «Работа на высоте без привязи». Он действует только внутри зоны «Работа на высоте»: отметьте её на шаге «Зоны» или отключите этот сценарий на шаге «Сценарии».';
    }
    return '';
  }
  return {renumber,nextName,removePoint,nearestPoint,wizardIssue};
});
