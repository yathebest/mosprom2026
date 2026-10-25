const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const archive=require('../app/static/clip-archive.js');
const event=(id,options={})=>({id,job_id:'jobA',camera:'A',track_id:id,kind:'no_helmet',clip:'clipA.mp4',status:'new',start_s:1,alert_s:2,created:'2026-10-25T12:00:00Z',payload:{clip_group_id:'groupA',episode_id:'episode'+id,clip_start_s:0,clip_end_s:6,clip_duration_s:5.9},...options});

test('two and three people share one clip card with actual distinct episode counts',()=>{
 for(const n of [2,3]){const rows=Array.from({length:n},(_,i)=>event(String(i)));const grouped=archive.group(rows);assert.equal(grouped.length,1);assert.equal(grouped[0].total,n);assert.deepEqual(grouped[0].counts,{no_helmet:n});assert.equal(rows.length,n)}
 const duplicate=event('duplicate',{payload:{clip_group_id:'groupA',episode_id:'episode1',clip_group_counts:{no_helmet:999}}});
 const grouped=archive.group([event('1'),event('2'),duplicate,event('phone',{kind:'possible_phone_use'})]);
 assert.equal(grouped.length,1);assert.deepEqual(grouped[0].counts,{no_helmet:2,possible_phone_use:1});assert.equal(grouped[0].members.length,4);
});

test('same actor with a new episode counts separately; rejected episodes are explicit, not quietly accepted',()=>{
 const rows=[event('one',{track_id:1}),event('two',{track_id:1}),event('false',{status:'false'})];
 const group=archive.group(rows)[0];assert.equal(group.total,2);assert.deepEqual(group.falseCounts,{no_helmet:1});assert.notEqual(group.representative.status,'false');
});

test('group IDs, paths and legacy exact bounds never merge different jobs or cameras',()=>{
 const rows=[event('1'),event('2',{job_id:'jobB'}),event('3',{camera:'B'})];assert.equal(archive.group(rows).length,3);
 const legacy=id=>event(id,{payload:{episode_id:id,clip_start_s:1,clip_end_s:8},clip:id+'.mp4'});
 assert.equal(archive.group([legacy('one'),legacy('two')]).length,1);
 assert.equal(archive.group([legacy('one'),{...legacy('two'),payload:{clip_start_s:10,clip_end_s:17}}]).length,2);
 assert.equal(archive.group([event('1',{payload:{}}),event('2',{payload:{}})]).length,1);
 assert.equal(archive.group([event('1',{job_id:null}),event('2',{job_id:null})]).length,2);
});

test('filtering matches any authorized group member while retaining every pattern and counts in that group',()=>{
 const groups=archive.group([event('one'),event('phone',{kind:'possible_phone_use',alert_s:3}),event('old',{clip:'old.mp4',payload:{clip_group_id:'old',episode_id:'old'},created:'2026-10-24T12:00:00Z'})]);
 const rows=archive.filter(groups,{kind:'possible_phone_use',camera:'A',date:'2026-10-25',start:2.5,end:4,state:'ready'},v=>v.slice(0,10));
 assert.equal(rows.length,1);assert.deepEqual(rows[0].counts,{possible_phone_use:1,no_helmet:1});assert.equal(rows[0].members.length,2);
 assert.equal(archive.filter(groups,{camera:'other'}).length,0);assert.equal(archive.filter(groups,{state:'pending'}).length,0);
 assert.deepEqual(archive.group([...groups.flatMap(g=>g.members)].reverse()).map(g=>g.key),groups.map(g=>g.key));
});

test('archive paginates groups rather than person rows and does not change the journal event list',()=>{
 const source=fs.readFileSync(require.resolve('../app/static/live.js'),'utf8');
 const code=source.slice(source.indexOf('function clipGroupHeading('),source.indexOf("['clip-camera','clip-kind'"));
 const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{value:'',innerHTML:'',insertAdjacentHTML(_where,html){this.innerHTML+=html},querySelectorAll:()=>[]});return nodes.get(id)};
 const events=[];for(let i=0;i<13;i++)for(let person=0;person<2;person++)events.push(event(`${i}-${person}`,{clip:`clip${i}.mp4`,payload:{clip_group_id:'group'+i,episode_id:`ep${i}-${person}`}}));
 const context={S:{events},L:{clipPage:0},$,esc:String,labels:{no_helmet:'Без каски'},reviews:{new:'Новое событие'},clipStates:{ready:'Клип готов'},ClipArchive:archive,UiFormat:{dateKey:v=>v.slice(0,10),dateTime:String},timestamp:String,IncidentUI:{primaryEventMarkup:()=>'<section>clip facts</section>'}};
 vm.runInNewContext(code,context);context.renderClips();
 assert.equal(($('clips-list').innerHTML.match(/<article class="clip-row"/g)||[]).length,12);assert.match($('clips-list').innerHTML,/Найдено клипов<\/span>: 13/);assert.match($('clips-list').innerHTML,/Без каски<\/span>: <b>2/);assert.equal(events.length,26);
 $('clip-next').onclick();assert.equal(($('clips-list').innerHTML.match(/<article class="clip-row"/g)||[]).length,1);assert.equal(context.L.clipPage,1);
});
