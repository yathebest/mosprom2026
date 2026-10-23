"""Measure the actual configured backend, with no database writes or metric overwrite.

predict() must block until inference outputs have been received. Measurements include
preprocessing, the API round trip and postprocessing; they are not TPU kernel times.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from app.core import RunConfig
from app.inference import create_backend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--frames',type=int,default=100)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--require-adapter',action='store_true')
    args=parser.parse_args()
    if not 1<=args.frames<=10000 or not 0<=args.warmup<=1000:parser.error('frames: 1..10000; warmup: 0..1000')
    if args.output.resolve().is_relative_to((ROOT/'docs').resolve()):parser.error('Use a new output folder, not the frozen evaluation reports')
    if args.output.exists():parser.error('Output exists; select a new path to preserve the previous run')
    if args.require_adapter and not os.getenv('LINQ_ADAPTER'):parser.error('LINQ_ADAPTER is required; local fallback refused')
    config=RunConfig.model_validate_json(args.config.read_text(encoding='utf-8')) if args.config else RunConfig(fast_motion=False,fall_detection=True)
    cap=cv2.VideoCapture(str(args.video));samples=[]
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        stride=max(1,round(fps/config.sample_fps)) if fps>0 else 1
        index=0
        while len(samples)<args.frames:
            ok,frame=cap.read()
            if not ok:break
            if index%stride==0:samples.append((index,frame))
            index+=1
    finally:cap.release()
    if not samples:raise SystemExit('No decodable frames')
    started=time.perf_counter()
    backend=create_backend(ROOT/'data/models/yolov8n-pose.pt',config.confidence,config.fall_detection,config)
    setup=time.perf_counter()-started
    for i in range(args.warmup):backend.predict(samples[i%len(samples)][1])
    timings=[];outputs=[]
    for index,frame in samples:
        began=time.perf_counter();people=backend.predict(frame);timings.append((time.perf_counter()-began)*1000)
        state=getattr(backend,'scene_state',None)
        if config.fall_detection and not isinstance(state,dict):raise RuntimeError('Fall benchmark requires backend.scene_state; pose-only results are not comparable to RGB fall metrics')
        outputs.append({'frame':index,'time_s':index/fps if fps else None,
                        'people':[{'box':list(p.box),'confidence':float(p.confidence),'keypoints':p.keypoints.tolist() if p.keypoints is not None else None} for p in people],
                        'scene_state':state,'objects':[asdict(o) for o in getattr(backend,'objects_state',[])]})
    report={'status':'measured','backend':backend.name,'adapter':os.getenv('LINQ_ADAPTER') or None,
            'scope':'synchronous predict wall time: preprocess + inference/RPC + postprocess; excludes decoding, tracking, rules and clip export',
            'hardware_identity':'must be verified from the vendor service logs; adapter presence alone is not evidence',
            'video':args.video.name,'video_sha256':hashlib.sha256(args.video.read_bytes()).hexdigest(),
            'config':config.model_dump(),'sampled_frames':len(samples),'source_fps':fps,'stride':stride,'warmup_calls':args.warmup,'setup_s':setup,
            'predict_ms':{'p50':float(np.percentile(timings,50)),'p95':float(np.percentile(timings,95)),'mean':float(np.mean(timings))},
            'predict_throughput_fps':len(timings)/(sum(timings)/1000),'per_call_ms':timings,
            'runtime':{'python':platform.python_version(),'opencv':cv2.__version__},'outputs':outputs}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create prevents accidental overwrite even if another process races us.
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
    print(json.dumps({k:report[k] for k in ('backend','sampled_frames','predict_ms','predict_throughput_fps')},ensure_ascii=True))


if __name__=='__main__':main()
