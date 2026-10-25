const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'../app/static/analytics-ui.js'),'utf8');
function fixture(){return {generated_at:'2026-10-25T18:00:00Z',period_start:'2026-10-22T21:00:00Z',period_end:'2026-10-25T21:00:00Z',available_cameras:['A','<B>'],
 kpis:{automatic_events:4,confirmed:2,pending:2},queue:{pending:2,overdue:99},priorities:[{camera:'Never show priority queue',kind:'no_helmet'}],
 episodes:{closed_count:1,closed_interval_total_s:2},coverage:{available:false},excluded:{},data_quality:{source_purpose_events:{unclassified:4}},
 breakdowns:{by_kind:{no_helmet:3,possible_phone_use:1},by_time_day:{'2026-10-23':2,'2026-10-25':2},by_time_day_kind:{'2026-10-23':{no_helmet:2},'2026-10-25':{no_helmet:1,possible_phone_use:1}}},
 cameras:[{camera:'A',events:3,by_kind:{no_helmet:2,possible_phone_use:1}},{camera:'<B>',events:1,by_kind:{no_helmet:1}}],can_drilldown:true,
 events:[{id:'helmet',job_id:'j',camera:'A',kind:'no_helmet',status:'new',created:'2026-10-23T12:00:00Z',occurred_at:'2026-10-25T12:00:00Z',start_s:0,alert_s:1},
 {id:'phone',job_id:'j',camera:'A',kind:'possible_phone_use',status:'confirmed',created:'2026-10-25T12:00:00Z',start_s:1,alert_s:2}]};}
function boot(data=fixture(),lang='ru'){
 const requests=[],exports=[],host={_html:'',cache:new Map(),classList:{add(){}},setAttribute(){},
  set innerHTML(value){this._html=value;this.cache.clear()},get innerHTML(){return this._html},
  querySelectorAll(selector){if(this.cache.has(selector))return this.cache.get(selector);const matches=[];
   for(const tag of this._html.matchAll(/<[a-z][^>]*>/gi)){let hit=selector.startsWith('#')?tag[0].includes('id="'+selector.slice(1)+'"'):tag[0].includes(selector.slice(1,-1));if(!hit)continue;
    const node={dataset:{},focus(){},addEventListener(name,fn){this['on'+name]=fn},scrollIntoView(){}};
    for(const attribute of tag[0].matchAll(/([\w-]+)(?:="([^"]*)")?/g)){const [,key,value='']=attribute;if(key.startsWith('data-'))node.dataset[key.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=value;else if(key==='id')node.id=value;}
    matches.push(node);
   }this.cache.set(selector,matches);return matches;},querySelector(selector){return this.querySelectorAll(selector)[0]||null}};
 const document={documentElement:{lang},activeElement:null,addEventListener(){},getElementById(id){return id==='statistics'?host:host.querySelector('#'+id)}};
 const window={A:{user:{id:'u',organization_id:'org',permissions:['analytics.read','events.read']}},location:{assign:url=>exports.push(url)}};
 const context={window,document,URLSearchParams,Intl,Date,Set,can:()=>true,S:{events:[]},openEvent(){},toast(){},api:async url=>{requests.push(url);return {analytics:data}}};
 vm.runInNewContext(source.replace('window.AnalyticsUI=','window.AnalyticsTest={state,params,content,patternRows,timeSeries,eventRows};window.AnalyticsUI='),context);
 return {window,document,host,requests,exports,helpers:window.AnalyticsTest,render:()=>window.AnalyticsUI.render(host,data)};
}
test('analytics renders aggregate charts instead of any review queue, with visible camera/status and compact quality details',()=>{
 const b=boot();b.render();const html=b.host.innerHTML;
 assert.match(html,/Всего событий/);assert.match(html,/Когда возникают события/);assert.match(html,/Распределение по камерам/);
 assert.doesNotMatch(html,/Куда направить внимание|Что разобрать сейчас|Never show priority queue|analytics-priorit/);
 assert.ok(html.indexOf('analytics-camera')<html.indexOf('analytics-extra-filters'));assert.match(html,/id="analytics-status"/);assert.equal((html.match(/id="analytics-extra-filters"/g)||[]).length,1);assert.doesNotMatch(html,/class="analytics-options"/);assert.match(html,/analytics-bin-date">23.10/);assert.match(html,/23.10.2026 · Событий: 2/);
 assert.match(html,/<details class="analytics-detail analytics-data-notes"\s+data/);
 assert.match(html,/100% — все события в выборке:/);assert.match(html,/не точность модели/);assert.match(html,/&lt;B&gt;/);
});
test('pattern shares use scope counts; absolute bars rescale only geometry, not percentages or denominators',()=>{
 const b=boot(),rows=b.helpers.patternRows(fixture());assert.equal(rows[0].count,3);assert.equal(rows[0].share,75);assert.equal(rows[0].total,4);assert.equal(rows[1].share,25);
 b.helpers.state.chartMode='absolute';const counts=b.helpers.patternRows(fixture());assert.equal(counts[0].width,100);assert.equal(counts[1].share,25);assert.equal(counts[1].width,100/3);
 assert.equal(b.helpers.patternRows({...fixture(),breakdowns:{by_kind:{no_helmet:0}}})[0].share,0);
});
test('selected-basis histogram inserts zero days, respects Moscow boundaries and bounds long periods',()=>{
 const b=boot(),bins=b.helpers.timeSeries(fixture());assert.deepEqual(Array.from(bins,row=>[row.from,row.count]),[['2026-10-23',2],['2026-10-24',0],['2026-10-25',2]]);
 b.helpers.state.focus={kind:'possible_phone_use'};assert.deepEqual(Array.from(b.helpers.timeSeries(fixture()),row=>row.count),[0,0,1]);
 b.helpers.state.focus=null;const long=b.helpers.timeSeries({...fixture(),period_start:'2010-01-01T00:00:00Z'});assert.ok(long.length<=60);assert.equal(long.reduce((sum,row)=>sum+row.count,0),4);
 const unknown={...fixture(),kpis:{automatic_events:0},breakdowns:{},excluded:{unknown_time:4}};
 assert.equal(b.helpers.timeSeries(unknown).length,0);assert.match(b.helpers.content(unknown),/Неизвестные даты не отображаются как нулевые дни/);
});
test('native pattern buttons focus both camera totals and timeline; chart measure and clear buttons work',()=>{
 const b=boot();b.render();const pattern=b.host.querySelectorAll('[data-analytics-pattern]').find(n=>n.dataset.analyticsPattern==='possible_phone_use');pattern.onclick();
 assert.equal(b.helpers.state.focus.kind,'possible_phone_use');assert.match(b.host.innerHTML,/aria-pressed="true" title="Телефон в руках/);
 const cameraPart=b.host.innerHTML.split('analytics-camera-chart')[1];assert.match(cameraPart,/<strong>1 <small>100%/);assert.match(cameraPart,/<strong>0 <small>0%/);
 b.host.querySelectorAll('[data-analytics-mode]').find(n=>n.dataset.analyticsMode==='absolute').onclick();assert.equal(b.helpers.state.chartMode,'absolute');
 b.host.querySelector('[data-analytics-clear-chart]').onclick();assert.equal(b.helpers.state.focus,null);
});
test('time-bin drilldown uses the chosen source time, without exposing unavailable event details',()=>{
 const b=boot();b.helpers.state.basis='recording';b.helpers.state.focus={dayFrom:'2026-10-25',dayTo:'2026-10-25'};
 const html=b.helpers.eventRows(fixture());assert.match(html,/Без каски/);assert.doesNotMatch(html,/Возможное использование телефона/);
 const restricted=b.helpers.eventRows({...fixture(),can_drilldown:false});assert.doesNotMatch(restricted,/data-analytics-proof|helmet|phone/);assert.match(restricted,/Доступны агрегаты/);
});
test('camera/status/custom dates use the same query in summary and both exports; background refresh cannot reset active filters',async()=>{
 const b=boot();b.render();b.host.querySelector('#analytics-status').onchange({target:{value:'confirmed'}});await new Promise(setImmediate);
 assert.equal(new URL(b.requests.at(-1),'http://localhost').searchParams.get('status'),'confirmed');
 b.host.querySelector('#analytics-camera').onchange({target:{value:'A'}});await new Promise(setImmediate);
 Object.assign(b.helpers.state,{days:'custom',from:'2026-10-23',to:'2026-10-25'});await b.window.AnalyticsUI.reload();
 const expected=b.helpers.params();for(const format of ['xlsx','csv']){b.host.querySelectorAll('[data-analytics-export]').find(n=>n.dataset.analyticsExport===format).onclick();const exported=new URL(b.exports.at(-1),'http://localhost');for(const [key,value] of expected)assert.equal(exported.searchParams.get(key),value);assert.equal(exported.searchParams.get('format'),format);}
 b.render();assert.equal(b.helpers.state.status,'confirmed');assert.equal(b.helpers.state.camera,'A');
});
test('English analytics labels and empty distributions remain clear and finite',()=>{
 const b=boot({...fixture(),kpis:{},breakdowns:{},cameras:[],events:[]},'en');b.render();assert.match(b.host.innerHTML,/Total events|Event patterns/);assert.match(b.host.innerHTML,/No events to distribute/);assert.doesNotMatch(b.host.innerHTML,/NaN|Infinity|Что разобрать/);
});
