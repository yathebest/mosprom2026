from app.scenarios import ppe_overlay_label,annotate_objects
from app.core import Person
from app.scenario_models import Observation
import numpy as np


def test_first_missing_measurement_is_confirming_not_confirmed():
    text,color=ppe_overlay_label('helmet','missing',{'state':'missing','confirmed_state':'unknown','fresh':True})
    assert 'MISSING (CONFIRMING)' in text and 'CONFIRMED' not in text
    assert color!=(65,65,240)


def test_confirmed_missing_and_clearance_pending_match_state_engine():
    text,color=ppe_overlay_label('helmet','missing',{'state':'missing','confirmed_state':'missing','fresh':True})
    assert text=='HELMET: MISSING (CONFIRMED)' and color==(65,65,240)
    text,color=ppe_overlay_label('helmet','worn',{'state':'worn','confirmed_state':'missing','fresh':True})
    assert 'WORN (CONFIRMING)' in text and 'CONFIRMED:MISSING' in text and color==(65,65,240)


def test_unknown_measurement_does_not_erase_confirmed_violation_or_turn_green():
    text,color=ppe_overlay_label('helmet','unknown',{'state':'unknown','confirmed_state':'missing','fresh':False})
    assert 'UNKNOWN (LAST MEASURED)' in text and 'CONFIRMED:MISSING' in text
    text,color=ppe_overlay_label('helmet','worn',{'state':'worn','confirmed_state':'worn','fresh':False})
    assert 'LAST MEASURED' in text and color!=(90,220,90)


def test_raw_negative_box_is_explicitly_measured_and_not_alarm_colored(monkeypatch):
    from app import scenarios
    labels=[];boxes=[]
    monkeypatch.setattr(scenarios.cv2,'putText',lambda image,text,origin,font,size,color,*args:labels.append((text,color)))
    monkeypatch.setattr(scenarios.cv2,'rectangle',lambda image,start,end,color,*args:boxes.append(color))
    person=Person((10,10,50,100),.9,track_id=1)
    frame=np.zeros((120,160,3),np.uint8)
    annotate_objects(frame,[Observation((12,12,30,30),.99,'no_hardhat','test')],{'1':{}},[person])
    assert labels[0][0]=='MEASURED no_hardhat 0.99'
    assert boxes[0]!=(65,65,240)
