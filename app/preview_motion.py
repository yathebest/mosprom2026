"""Optical-flow projections for the preview only, never alert evidence."""
from dataclasses import replace
import cv2
import numpy as np


class PreviewMotion:
    def __init__(self):
        self.source=None;self.gray=None;self.points=None;self.owners=[];self.people=[];self.last_time=None

    def project(self,visual,preview,now):
        if not visual or now-visual['time']>2.5 or now<visual['time']:
            self.source=None;return []
        gray=cv2.cvtColor(preview,cv2.COLOR_BGR2GRAY)
        h,w=gray.shape
        if self.source!=visual['time']:
            source=visual.get('gray')
            if source is None or source.shape!=gray.shape:return []
            scale=w/visual['width'];self.people=[];points=[];owners=[]
            for p in visual['people']:
                kp=p.keypoints.copy() if p.keypoints is not None else None
                if kp is not None:kp[:,:2]*=scale
                p=replace(p,box=tuple(v*scale for v in p.box),keypoints=kp)
                self.people.append(p)
                a,b,c,d=p.box;bw,bh=c-a,d-b
                mask=np.zeros_like(gray)
                x1,y1=max(0,int(a+bw*.15)),max(0,int(b+bh*.08))
                x2,y2=min(w,int(c-bw*.15)),min(h,int(d-bh*.08))
                if x2<=x1 or y2<=y1:continue
                mask[y1:y2,x1:x2]=255
                found=cv2.goodFeaturesToTrack(source,maxCorners=32,qualityLevel=.015,minDistance=3,mask=mask,blockSize=3)
                if found is not None:
                    points.extend(found[:,0]);owners.extend([len(self.people)-1]*len(found))
            self.source=visual['time'];self.gray=source
            self.points=np.asarray(points,np.float32).reshape(-1,1,2);self.owners=np.asarray(owners)
            self.last_time=visual['time']
        if not len(self.points):return []
        if self.gray.shape!=gray.shape or now-self.last_time>2.5:return []
        following,status,error=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None,
            winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.03))
        if following is None or status is None:return []
        back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,following,None,
            winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.03))
        if back is None or back_status is None:return []
        old=self.points[:,0];new=following[:,0]
        valid=(status[:,0]>0)&(back_status[:,0]>0)&(np.linalg.norm(back[:,0]-old,axis=1)<1.5)
        if error is not None:valid&=error[:,0]<25
        valid&=np.isfinite(new).all(1)&(new[:,0]>=0)&(new[:,0]<w)&(new[:,1]>=0)&(new[:,1]<h)
        result=[];accepted=np.zeros(len(valid),bool)
        for i,p in enumerate(self.people):
            owned=self.owners==i;good=owned&valid
            if good.sum()<4 or good.sum()<owned.sum()*.4:continue
            delta=np.median(new[good]-old[good],axis=0)
            if np.median(np.linalg.norm((new[good]-old[good])-delta,axis=1))>8:continue
            box=tuple(np.clip(np.asarray(p.box)+np.tile(delta,2),[0,0,0,0],[w,h,w,h]))
            if box[2]-box[0]<3 or box[3]-box[1]<3:continue
            kp=p.keypoints.copy() if p.keypoints is not None else None
            if kp is not None:kp[:,:2]+=delta
            projected=replace(p,box=box,keypoints=kp,observed=False,alert_eligible=False)
            self.people[i]=projected;result.append(projected)
            accepted|=good
        valid&=accepted
        self.points=following[valid];self.owners=self.owners[valid];self.gray=gray;self.last_time=now
        return result
