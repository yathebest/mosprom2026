"""Integration of real detectors with upload + live alerts/clips. No quality claims."""
from pathlib import Path
import cv2,time,json,requests,subprocess
ROOT=Path(__file__).resolve().parents[1];FIX=ROOT/'data/datasets/scenario-fixtures';URL='http://127.0.0.1:8765'
http=requests.Session()
def get(path):
    r=http.get(URL+path,timeout=30);r.raise_for_status();return r.json()
def post(path,**kw):
    r=http.post(URL+path,timeout=60,**kw);r.raise_for_status();return r.json()
report={'purpose':'Integration only: repeated still images are not independently evaluated videos.','checks':[]}
for name,flag,expected in [('no_hardhat','helmet_detection','no_helmet'),('hardhat','helmet_detection',None),('no_harness','harness_detection','no_harness_at_height'),('fire-demo-frame','fire_detection','fire'),('phone-demo','phone_detection','possible_phone_use'),('smoking-output2','smoking_detection','possible_smoking')]:
    frame=cv2.imread(str(FIX/(name+'.jpg')));h,w=frame.shape[:2];frame=frame[:h//2*2,:w//2*2]
    import imageio_ffmpeg
    video=FIX/(name+'-demo.mp4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-loop','1','-i',str(FIX/(name+'.jpg')),'-t','7','-r','10','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','0','-pix_fmt','yuv444p',str(video)],check=True)
    cfg={'camera':'Интеграционный пример · '+name,'fast_motion':False,'fall_detection':False,flag:True,'sample_fps':5}
    if flag=='harness_detection':cfg['zones']=[{'name':'Учебная область высоты','kind':'height','points':[[0,0],[1,0],[1,1],[0,1]]}]
    if flag=='phone_detection':cfg['zones']=[{'name':'Учебный контекст оборудования','kind':'equipment','points':[[0,0],[1,0],[1,1],[0,1]]}]
    with video.open('rb') as f:job=post('/api/jobs',files={'file':(video.name,f,'video/mp4')})
    post('/api/jobs/'+job['id']+'/start',json=cfg)
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        state=get('/api/jobs/'+job['id'])
        if state['status'] in ('completed','failed','interrupted'):break
        time.sleep(.5)
    events=get('/api/events?job_id='+job['id']);kinds=[e['kind'] for e in events]
    row={'sample':name,'job_id':job['id'],'status':state['status'],'events':kinds,'clips_ready':all(e['clip'] and e['payload']['clip_state']=='ready' for e in events),'stats':state.get('stats')}
    row['passed']=state['status']=='completed' and row['clips_ready'] and (expected in kinds if expected else not kinds)
    report['checks'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='stats'},ensure_ascii=False),flush=True)
# Live uses the same public JPEG API as browser cameras, without opening a physical device.
frame=cv2.imread(str(FIX/'no_hardhat.jpg'));jpeg=cv2.imencode('.jpg',frame)[1].tobytes()
session=post('/api/live',json={'source_type':'browser','config':{'camera':'Интеграционный LIVE · каска','fast_motion':False,'helmet_detection':True}})
began=time.monotonic();visible_during_live=False
while time.monotonic()-began<13:
    post('/api/live/'+session['id']+'/frame',data=jpeg,headers={'Content-Type':'image/jpeg'})
    if int((time.monotonic()-began)*10)%10==0:
        events=get('/api/events?job_id='+session['id']);visible_during_live|=any(e['clip'] for e in events)
    time.sleep(.09)
post('/api/live/'+session['id']+'/stop')
deadline=time.monotonic()+60
while time.monotonic()<deadline:
    state=get('/api/live/'+session['id'])
    if state['state'] in ('stopped','failed') and not state['clips_pending']:break
    time.sleep(.3)
events=get('/api/events?job_id='+session['id'])
row={'sample':'live-no-hardhat','job_id':session['id'],'state':state,'events':[e['kind'] for e in events],'clip_ready_before_stop':visible_during_live,'clips_ready':all(e['clip'] for e in events)}
row['passed']=state['state']=='stopped' and len(events)==1 and events[0]['kind']=='no_helmet' and row['clips_ready'] and visible_during_live
report['checks'].append(row);report['passed']=all(r['passed'] for r in report['checks'])
(ROOT/'docs/scenario_e2e.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(row,ensure_ascii=False),flush=True)
assert report['passed'],'Some integration checks failed; see docs/scenario_e2e.json'
