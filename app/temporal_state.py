"""Time-weighted confirmation of fresh, independent pattern observations.

Unknown samples and source gaps occupy the window but contribute no vote.
The last value is never extrapolated to the current wall-clock time.
"""
from collections import deque
import math
import uuid


class TemporalGate:
    def __init__(self,ratio=.75,min_samples=3):
        self.ratio=ratio;self.min_samples=min_samples;self.history=deque()
        self.state='unknown';self.observed_state='unknown';self.confirmed_at_s=None
        self.last_seen=None;self.last_true=None;self.last_observed_s=None;self.since=None
        self.sequence=0;self.samples=0;self.episode_id=None;self.episode_start=None
        self.last_transition=None;self.window_s=1.;self.clear_window_s=2.;self.gap=.8
        self.nominal_window_s=1.;self.nominal_clear_window_s=2.
        self.agreement=0.;self.coverage=0.;self.sample_count=0;self.observation_fresh=False

    @property
    def active(self):return self.state=='violation' and self.episode_id is not None

    def _support(self,now,target,window):
        cutoff=now-window;weight=0.;known=0.
        samples=[(t,s) for t,s in self.history if cutoff-1e-8<=t<=now and s is target]
        for (left,state),(right,next_state) in zip(self.history,list(self.history)[1:]):
            if state is None or next_state is None or right-left>self.gap+1e-8:continue
            duration=max(0.,min(right,now)-max(left,cutoff));known+=duration
            if state is target:weight+=duration
        return weight/window,known/window,len(samples),samples[0][0] if samples else None

    def update(self,now,condition,confirm,release,gap):
        self.last_transition=None;self.observation_fresh=False
        if not math.isfinite(now):return None
        if self.last_seen is not None and now<=self.last_seen:return None
        self.nominal_window_s=float(confirm);self.nominal_clear_window_s=float(release);self.gap=float(gap)
        self.window_s=float(confirm);self.clear_window_s=float(release)
        self.last_seen=now;condition=None if condition is None else bool(condition)
        self.observed_state='unknown' if condition is None else 'violation' if condition else 'clear'
        self.observation_fresh=condition is not None
        self.history.append((now,condition))
        horizon=max(confirm,release,gap*(self.min_samples-1))
        while len(self.history)>2 and self.history[1][0]<now-horizon:self.history.popleft()
        if condition is None:
            self.since=None;self.samples=0;self.agreement=0.;self.coverage=0.;self.sample_count=0;return None
        self.last_observed_s=now
        if condition:self.last_true=now
        window=confirm if condition else release
        # Slow LIVE still needs three independent observations. Expand a short
        # nominal window to their real source-time span, never bridge unknowns
        # or a gap, and report the effective window instead of voting twice.
        supporting=[];previous=None
        for t,value in reversed(self.history):
            if value is None or (previous is not None and previous-t>gap+1e-8):break
            if value is condition:supporting.append(t)
            previous=t
            if len(supporting)>=self.min_samples:break
        if len(supporting)>=self.min_samples:
            span=now-supporting[-1]
            if span<=gap*(self.min_samples-1)+1e-8:window=max(window,span)
        if condition:self.window_s=window
        else:self.clear_window_s=window
        agreement,coverage,count,started=self._support(now,condition,window)
        self.agreement=agreement;self.coverage=coverage;self.sample_count=count
        self.since=started;self.samples=count
        # The full source-time window, not only its known fragments, is the
        # denominator. Unknown/gaps cannot make a short burst look unanimous.
        elapsed=now-self.history[0][0]
        if elapsed+1e-8<window or agreement+1e-8<self.ratio or count<self.min_samples:return None
        target='violation' if condition else 'clear'
        if target==self.state:return None
        previous=self.state;self.state=target;self.confirmed_at_s=now
        proof=self.confirmation(window,agreement,coverage,count)
        if condition:
            self.sequence+=1;self.episode_id=uuid.uuid4().hex;self.episode_start=started
            self.last_transition={'episode_id':self.episode_id,'episode_state':'open','end_s':None,
                'state_from':previous,'state_to':target,'state_confirmation':proof}
            return started,self.sequence
        if self.episode_id is not None:
            end=max(self.episode_start,started)
            self.last_transition={'episode_id':self.episode_id,'episode_state':'closed','end_s':round(end,3),
                'closed_at_s':round(now,3),'duration_s':round(end-self.episode_start,3),
                'closure_reason':'confirmed_clear','state_from':previous,'state_to':target,'state_confirmation':proof}
            self.episode_id=None;self.episode_start=None
        return None

    def confirmation(self,window=None,agreement=None,coverage=None,count=None):
        effective=self.clear_window_s if self.observed_state=='clear' else self.window_s
        nominal=self.nominal_clear_window_s if self.observed_state=='clear' else self.nominal_window_s
        effective=float(window if window is not None else effective)
        return {'window_s':effective,'nominal_window_s':nominal,'cadence_limited':effective>nominal+1e-8,
            'ratio':self.ratio,'min_samples':self.min_samples,'agreement':round(self.agreement if agreement is None else agreement,4),
            'coverage':round(self.coverage if coverage is None else coverage,4),
            'samples':self.sample_count if count is None else count,'weighting':'elapsed_source_time'}

    def snapshot(self):
        return {'state':self.state,'observed_state':self.observed_state,'last_observed_state':self.observed_state,
            'confirmed_at_s':self.confirmed_at_s,'window_s':self.clear_window_s if self.observed_state=='clear' else self.window_s,
            'violation_window_s':self.window_s,'clear_window_s':self.clear_window_s,
            'nominal_window_s':self.nominal_window_s,'nominal_clear_window_s':self.nominal_clear_window_s,
            'agreement':round(self.agreement,4),'state_confirmation':self.confirmation(),
            'last_observed_s':self.last_observed_s,'observed_at_s':self.last_observed_s,
            'last_sample_s':self.last_seen,'observation_fresh':self.observation_fresh,'episode_id':self.episode_id}

    def interrupt(self,now,reason):
        if self.episode_id is None:return None
        patch={'episode_id':self.episode_id,'episode_state':'interrupted','end_s':None,
            'closed_at_s':round(now,3),'observed_until_s':self.last_observed_s,
            'closure_reason':reason,'state_from':self.state,'state_to':'unknown'}
        self.episode_id=None;self.episode_start=None;self.state='unknown';self.observed_state='unknown'
        self.observation_fresh=False;return patch


def _iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1])
    return inter/max(area(a)+area(b)-inter,1e-9)


class PatternStates:
    def __init__(self,config):
        self.config=config;self.gates={};self.boxes={};self.updated=set();self.event_updates=[];self.state_rows={}

    def begin(self,people,now):
        self.now=now;self.updated=set();self.event_updates=[];handoffs={}
        visible={p.track_id for p in people if p.observed and p.track_id>0}
        newcomers=[p for p in people if p.track_id>0 and p.track_id not in self.boxes and p.observed
                   and p.alert_eligible and p.confidence>=self.config.confidence]
        used=set()
        for person in people:
            tid=person.track_id
            if tid<=0 or not person.observed or not person.alert_eligible or person.confidence<self.config.confidence:continue
            if tid not in self.boxes:
                matches=sorted(((_iou(person.box,box),old) for old,(last,box) in self.boxes.items()
                    if old not in visible and old not in used and 0<=now-last<=3),reverse=True)
                if matches and matches[0][0]>=.55 and (len(matches)==1 or matches[0][0]-matches[1][0]>=.12):
                    old=matches[0][1]
                    reverse=sorted(((_iou(p.box,self.boxes[old][1]),p.track_id) for p in newcomers),reverse=True)
                    if reverse[0][1]==tid and (len(reverse)==1 or reverse[0][0]-reverse[1][0]>=.12):
                        used.add(old);handoffs[tid]=old
                        for key in [k for k in self.gates if k[0]==old]:self.gates[(tid,*key[1:])]=self.gates.pop(key)
                        self.boxes.pop(old,None)
            self.boxes[tid]=(now,tuple(person.box))
        for tid,(last,_) in list(self.boxes.items()):
            if now-last<=15:continue
            for key in [k for k in self.gates if k[0]==tid]:
                patch=self.gates[key].interrupt(now,'track_lost')
                if patch:self.event_updates.append(self._patch(key,patch))
                del self.gates[key]
            del self.boxes[tid]
        return handoffs

    @staticmethod
    def _patch(key,patch):return {**patch,'track_id':key[0],'kind':key[1],'zone':key[2],'scope':'person' if key[0] else 'scene'}

    def observe(self,tid,kind,zone,condition,hold,release,gap):
        key=(tid,kind,zone);self.updated.add(key)
        gate=self.gates.setdefault(key,TemporalGate(self.config.state_confirmation_ratio,self.config.state_min_samples))
        hold=self.config.state_windows.get(kind,hold)
        fired=gate.update(now=self.now,condition=condition,confirm=hold,release=release,gap=gap)
        if gate.last_transition and gate.last_transition['episode_state']=='closed':self.event_updates.append(self._patch(key,gate.last_transition))
        return gate,fired

    def end(self,now):
        for key,gate in self.gates.items():
            if key not in self.updated:gate.update(now,None,gate.nominal_window_s,gate.nominal_clear_window_s,gate.gap)
        rows={}
        for (tid,kind,zone),gate in self.gates.items():
            rows.setdefault(tid,{})[kind+(':'+zone if zone else '')]={**gate.snapshot(),'kind':kind,'zone':zone,'scope':'person' if tid else 'scene'}
        self.state_rows=rows;return rows

    def finish(self,now,reason):
        updates=[]
        for key,gate in self.gates.items():
            patch=gate.interrupt(now,reason)
            if patch:updates.append(self._patch(key,patch))
        self.end(now);return updates
