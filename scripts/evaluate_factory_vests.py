"""Visually reviewed garment checks; uniform compliance is not assessed."""
from pathlib import Path
import json
import sys
import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.core import RunConfig
from app.inference import create_backend
from app.video import MODEL
from app.scenarios import ScenarioRules
from app.scenario_models import iou
import app.scenario_models as scenarios


def main():
    rows=json.loads((ROOT/'data/datasets/factory-review/reviewed-head-crops.json').read_text())['rows']
    selected={str(x).zfill(4):'missing' for x in [192,193,195,196,197,199,202,208,214,215,223]}
    selected['0216']='worn'
    cfg=RunConfig(vest_detection=True,fast_motion=False,confidence=.35)
    backend=create_backend(MODEL,.35,config=cfg);original=scenarios.pose_torso_proposal;result=[]
    cache={}
    for row in rows:
        if row['id'] not in selected:continue
        key=(row['source_video'],row['frame_index'])
        if key not in cache:
            cap=cv2.VideoCapture(str(ROOT/'data/datasets/factory-expansion'/row['source_video']))
            cap.set(cv2.CAP_PROP_POS_FRAMES,row['frame_index']);ok,frame=cap.read();cap.release()
            if not ok:raise RuntimeError('Cannot decode source')
            people=[p for p in backend.predict_people(frame) if p.confidence>=cfg.confidence]
            for i,p in enumerate(people):p.track_id=i+1
            assessments={}
            for enabled in [False,True]:
                scenarios.pose_torso_proposal=original if enabled else lambda *args:None
                objects=backend.predict_scenarios(frame,people);rules=ScenarioRules(cfg)
                rules.step(people,objects,0,frame.shape[1],frame.shape[0])
                assessments[str(enabled)]={p.track_id:rules.ppe_details.get(p.track_id,{}).get('vest',{}) for p in people}
            scenarios.pose_torso_proposal=original;cache[key]=(people,assessments)
        people,assessments=cache[key];person=max(people,key=lambda p:iou(p.box,row['person_box']),default=None)
        record={'id':row['id'],'source_video':row['source_video'],'frame_index':row['frame_index'],
            'truth':selected[row['id']],'reviewer':'assistant_visual_review_2026-10-24'}
        for enabled in [False,True]:record['pose_fallback' if enabled else 'detector_only']=assessments[str(enabled)].get(person.track_id,{}) if person else {}
        result.append(record);print('VEST_CHECK',json.dumps(record),flush=True)
    report={'method':'Assistant visually reviewed whole-person crops in torso-review.jpg; not expert-certified.',
        'scope':'Visible vest garment only; factory uniforms with reflective strips are not automatically noncompliant. Not an event metric or independent safety audit.',
        'rows':result,'status':'debug_subset_not_deployment_certification'}
    (ROOT/'docs/factory-vest-debug.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
