const {test}=require('node:test'),assert=require('node:assert/strict');
const policy=require('../app/static/pattern-policy.js');
const fs=require('node:fs'),vm=require('node:vm');
const manifest={version:'test-policy',patterns:[{id:'helmet',enabled:true,config_flag:'helmet_detection',event_kinds:['no_helmet']},{id:'fall',enabled:false,config_flag:'fall_detection',event_kinds:['possible_fall','person_down']},{id:'phone',enabled:false,config_flag:'phone_detection',event_kinds:['possible_phone_use']},{id:'danger_zone',enabled:true,config_flag:null,event_kinds:['danger_zone']}],enabled_flags:['helmet_detection'],disabled_flags:['fall_detection','phone_detection','fast_motion'],enabled_zone_kinds:['danger_zone','reflection']};
test('saved configuration cannot silently re-enable a retired pattern, zone or state window',()=>{
 const p=policy.create(manifest),old={camera:'A',helmet_detection:true,fall_detection:true,phone_detection:true,fast_motion:true,zones:[{name:'old',kind:'under_load',points:[[0,0],[1,0],[1,1]]},{name:'mirror',kind:'reflection'},{name:'danger',kind:'danger_zone'}],state_windows:{possible_fall:.5,no_helmet:2}};
 const clean=p.sanitize(old);assert.equal(clean.fall_detection,false);assert.equal(clean.phone_detection,false);assert.equal(clean.fast_motion,false);assert.equal(clean.helmet_detection,true);assert.deepEqual(clean.zones.map(x=>x.name),['mirror','danger']);assert.deepEqual(clean.state_windows,{no_helmet:2});assert.equal(old.fall_detection,true);assert.equal(old.zones.length,3);assert.equal(p.removed(old).length,4);
});
test('sample and choice availability is explicit, and unknown manifests fail closed',()=>{
 const p=policy.create(manifest);assert.equal(p.enabled('fall'),false);assert.equal(p.allowsEvent('possible_fall'),false);assert.equal(p.sampleAllowed({pattern_ids:['fall']}),false);assert.equal(p.sampleAllowed({pattern_ids:['helmet']}),true);assert.equal(p.sampleAllowed({id:'unmapped'}),false);assert.equal(policy.create().enabled('helmet'),false);
});
test('policy changes do not remove historical event data or labels',()=>{
 const p=policy.create(manifest);const history={kind:'possible_fall',clip:'original.mp4',status:'new'};p.sanitize({fall_detection:true});assert.deepEqual(history,{kind:'possible_fall',clip:'original.mp4',status:'new'});
});

test('disabled phone work-context option is cleared in saved config while phone itself remains enabled',()=>{
 const manifest={version:'phone-without-context',patterns:[{id:'phone',enabled:true,config_flag:'phone_detection',event_kinds:['possible_phone_use']}],enabled_flags:['phone_detection'],disabled_flags:['phone_work_context_only'],enabled_zone_kinds:['reflection']};
 const p=policy.create(manifest),old={phone_detection:true,phone_work_context_only:true,zones:[]};
 const c=p.sanitize(old);assert.equal(c.phone_detection,true);assert.equal(c.phone_work_context_only,false);assert.equal(old.phone_work_context_only,true);
 assert.equal(p.enabled('phone'),true);assert.equal(p.isDisabledFlag('phone_work_context_only'),true);assert.equal(p.isDisabledFlag('phone_detection'),false);
 const nodes=new Map();for(const prefix of ['extra','live-extra'])for(const key of ['phone','phone_work_context_only']){const id=prefix+'-'+key;nodes.set(id,{checked:true,closest(){return {remove(){nodes.delete(id)}}}})}
 nodes.set('step1',{querySelectorAll:()=>[]});
 const $=id=>nodes.get(id),context={S:{},PatternPolicy:policy,$,setCheck:(id,v)=>{if($(id))$(id).checked=v},scenarioOptions:[['phone']],IncidentUI:{patternKinds:[]},updateConfirmationControls(){},document:{dispatchEvent(){}},CustomEvent:class{}};
 vm.runInNewContext(fs.readFileSync(require.resolve('../app/static/pattern-ui.js'),'utf8'),context);context.applyPatternAvailability(manifest);
 for(const prefix of ['extra','live-extra']){assert.ok($(prefix+'-phone'));assert.equal($(prefix+'-phone_work_context_only'),undefined)}
});
test('disallowed choices, numeric settings and zone options are removed rather than hidden, while history selectors stay intact',()=>{
 const nodes=new Map();function element(id){const node={id,removed:false,value:'',checked:true,options:[],querySelector:()=>null,querySelectorAll:()=>[],closest(){return {remove(){node.removed=true}}},remove(){node.removed=true}};nodes.set(id,node);return node}
 const $=id=>{const n=nodes.get(id);return n&&!n.removed?n:null};
 for(const prefix of ['extra','live-extra'])for(const id of ['helmet','phone','vest','phone_confidence','ppe_hold','object_hold','object_confidence','proximity_heights','proximity_m','phone_work_context_only','window-no_helmet','window-possible_fall','window-possible_phone_use'])element(prefix+'-'+id);
 for(const id of ['check-fall','live-fall','check-motion','live-motion','check-zone','live-zone-on','body-threshold','body-hold','speed','motion-hold','metric-speed','zone-hold','step1','filter-kind','clip-kind'])element(id);
 for(const id of ['zone-kind','live-zone-kind']){const select=element(id);select.value='height';select.options=['danger_zone','height','reflection'].map(value=>({value,remove(){select.options=select.options.filter(x=>x!==this);if(select.value===value)select.value=''}}));}
 const events=[];const context={S:{},PatternPolicy:policy,$,setCheck:(id,v)=>{if($(id))$(id).checked=v},scenarioOptions:[['helmet'],['phone'],['vest']],IncidentUI:{patternKinds:['no_helmet','possible_fall','possible_phone_use']},updateConfirmationControls(){},document:{dispatchEvent:e=>events.push(e)},CustomEvent:class{constructor(type,options){this.type=type;this.detail=options.detail}}};
 vm.runInNewContext(fs.readFileSync(require.resolve('../app/static/pattern-ui.js'),'utf8'),context);
 context.applyPatternAvailability(manifest);
 for(const id of ['check-fall','live-fall','extra-phone','live-extra-vest','extra-phone_confidence','body-hold','extra-window-possible_fall'])assert.equal($(id),null,id);
 assert.ok($('extra-helmet'));assert.ok($('filter-kind'));assert.ok($('clip-kind'));assert.deepEqual($('zone-kind').options.map(x=>x.value),['danger_zone','reflection']);assert.equal(events[0].type,'linq:patternpolicychange');
});
