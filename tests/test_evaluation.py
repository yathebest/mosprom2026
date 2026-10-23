from scripts.evaluate_events import evaluate


def test_duplicate_alarm_is_false_positive_and_other_video_does_not_match():
    truth=[{'video':'a','kind':'danger_zone','start_s':1,'end_s':5},
           {'video':'b','kind':'danger_zone','start_s':1,'end_s':5}]
    predictions=[{'video':'a','kind':'danger_zone','alert_s':2},
                 {'video':'a','kind':'danger_zone','alert_s':3}]
    result=evaluate(truth,predictions)
    assert (result['tp'],result['fp'],result['fn'])==(1,1,1)
    assert result['precision']==result['recall']==.5
    assert result['matched_alert_delays_s']==[1]


def test_empty_sets_do_not_invent_perfect_scores():
    result=evaluate([],[])
    assert result['precision'] is None and result['recall'] is None
