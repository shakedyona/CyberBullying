import os
from source.CyberBullying.Performances import performances as per

ROOT = os.path.abspath(os.path.join(__file__, '../../../../'))


def test_correct_performance():
    y = [1, 0, 0, 1, 0, 0]
    pred = [0, 1, 0, 1, 1, 0]
    dic_per = per.get_performances(y, pred)
    if 'f-score' not in dic_per or 'precision' not in dic_per or 'recall' not in dic_per:
        assert False
    if dic_per['f-score'] != 0.4:
        assert False
    if round(dic_per['precision'], 2) != 0.33:
        assert False
    if dic_per['recall'] != 0.5:
        assert False
    assert True


test_correct_performance()
