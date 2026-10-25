const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync(require.resolve('../app/static/index-v2.html'),'utf8');
function liveTree(){
 const section=html.slice(html.indexOf('<section id="live"'),html.indexOf('<section id="clips"'));
 const root={tag:'root',children:[],attrs:''},stack=[root],ids=new Map(),nodes=[];
 for(const token of section.matchAll(/<(\/?)([a-z][\w-]*)\b([^>]*)>/gi)){
  const [,closing,rawTag,attrs]=token,tag=rawTag.toLowerCase();
  if(closing){const at=stack.findLastIndex(n=>n.tag===tag);if(at>0)stack.splice(at);continue}
  const node={tag,attrs,parent:stack.at(-1),children:[],offset:token.index};node.parent.children.push(node);nodes.push(node);
  const id=attrs.match(/\bid="([^"]+)"/)?.[1];if(id){assert.equal(ids.has(id),false,`Duplicate LIVE id: ${id}`);node.id=id;ids.set(id,node)}
  if(!['img','input','br','hr','meta','link','source'].includes(tag))stack.push(node);
 }
 return {ids,nodes,inside:(node,id)=>{for(let p=node?.parent;p;p=p.parent)if(p.id===id)return true;return false}};
}
test('video and alerts share the first workspace, while all setup fields remain outside that sizing grid',()=>{
 const {ids,nodes,inside}=liveTree();
 const workspace=nodes.find(n=>/class="[^"]*live-workspace/.test(n.attrs));assert.ok(workspace);
 assert.equal(inside(ids.get('live-feed'),'live-setup-drawer'),false);
 assert.equal(ids.get('live-alert-rail').parent,workspace);
 assert.equal(inside(ids.get('live-events'),'live-alert-rail'),true);
 assert.equal(inside(ids.get('live-alert-queue'),'live-alert-rail'),true);
 assert.ok(ids.get('live-alert-queue').offset<ids.get('live-events').offset);
 assert.ok(ids.get('live-feed').offset<ids.get('live-alert-rail').offset);
 assert.ok(ids.get('live-alert-rail').offset<ids.get('live-setup-drawer').offset);
 for(const id of ['live-url','live-device','webcam-preview','live-zone-canvas','live-template','live-fall','live-zone-on'])assert.equal(inside(ids.get(id),'live-setup-drawer'),true,id);
 assert.doesNotMatch(ids.get('live-setup-drawer').attrs,/\bopen(?:\s|$|=)/);
});
test('camera actions and all alert/source IDs survive the layout move; long states are disclosed',()=>{
 const {ids,inside}=liveTree();
 for(const id of ['live-new','live-start','live-stop','live-config-open']){assert.ok(ids.has(id),id);assert.equal(inside(ids.get(id),'live-setup-drawer'),false,id)}
 for(const id of ['live-alert','live-alert-title','live-alert-time','live-alert-clips','live-frames','live-model-latency','live-events-total','live-state','live-clock','live-sessions'])assert.ok(ids.has(id),id);
 assert.equal(inside(ids.get('live-sessions'),'live-state-details'),true);
 assert.doesNotMatch(ids.get('live-state-details').attrs,/\bopen(?:\s|$|=)/);
 const css=fs.readFileSync(require.resolve('../app/static/product.css'),'utf8');
 assert.match(css,/#live \.live-layout\.live-workspace\{grid-template-columns:minmax\(0,1fr\) minmax\(280px,340px\)/);
 assert.match(css,/@media\(max-width:900px\)\{#live \.live-layout\.live-workspace\{grid-template-columns:minmax\(0,1fr\)/);
 const wall=fs.readFileSync(require.resolve('../app/static/camera-wall.css'),'utf8');
 assert.doesNotMatch(wall,/body:not\(\.can-manage-live\)\s+#live\s+\.live-layout\s*\{/,'An obsolete operator override must not force the alert rail below the video');
});
test('setup opens only on an explicit action and closing it changes no stream/config state',()=>{
 const source=fs.readFileSync(require.resolve('../app/static/live-setup.js'),'utf8');
 const initial=source.slice(0,source.indexOf('const cameraPreview='));
 const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{open:false,attrs:{},focused:false,scrolled:false,setAttribute(k,v){this.attrs[k]=v},focus(){this.focused=true},scrollIntoView(){this.scrolled=true},addEventListener(){}});return nodes.get(id)};
 const observed=[],frames=[];
 const session={id:'A',state:'running',config:{helmet_detection:true}},context={$,L:{current:session},window:null,document:{querySelector:()=>null},addEventListener(){},requestAnimationFrame:callback=>frames.push(callback),ResizeObserver:class{constructor(callback){this.callback=callback}observe(el){observed.push(el)}}};context.window=context;
 vm.runInNewContext(initial,context);assert.equal($('live-setup-drawer').open,false);
 assert.equal(frames.length,1);assert.ok(observed.includes($('live-camera-wall')));
 $('live-config-open').onclick();assert.equal($('live-setup-drawer').open,true);assert.equal($('live-config-open').attrs['aria-expanded'],'true');assert.equal($('live-setup-drawer').scrolled,true);
 $('live-config-close').onclick();assert.equal($('live-setup-drawer').open,false);assert.equal($('live-config-open').focused,true);
 assert.deepEqual(session,{id:'A',state:'running',config:{helmet_detection:true}});
});
test('workspace fits the actual document position rather than assuming equal admin/operator toolbars',()=>{
 const source=fs.readFileSync(require.resolve('../app/static/live-setup.js'),'utf8');
 const code=source.slice(source.indexOf('function fitLiveWorkspace(){'),source.indexOf("if(typeof ResizeObserver"));
 let top=356,visible=true;const values=[];
 const workspace={getClientRects:()=>visible?[{}]:[],getBoundingClientRect:()=>({top}),style:{setProperty:(key,value)=>values.push([key,value])}};
 const context={document:{querySelector:()=>workspace},window:{scrollY:0,innerHeight:712}};
 vm.runInNewContext(code,context);context.fitLiveWorkspace();assert.deepEqual(values.at(-1),['--live-workspace-height','338px']);
 top=156;context.window.scrollY=200;context.fitLiveWorkspace();assert.equal(values.at(-1)[1],'338px','Scrolling must not enlarge the original workspace');
 top=300;context.window.scrollY=0;context.window.innerHeight=1152;context.fitLiveWorkspace();assert.equal(values.at(-1)[1],'700px');
 visible=false;const n=values.length;context.fitLiveWorkspace();assert.equal(values.length,n,'A hidden page must not receive a guessed height');
});
