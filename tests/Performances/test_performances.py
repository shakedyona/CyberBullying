import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from source.Preprocessing import preprocessing as pre
import pandas as pd
import source.FeatureExtraction.featureExtraction as fe
import source.TraditionalMLArchitecture.RandomForest as rf
from source import utils
from source.Performances import performances as per
import numpy as np
ROOT = os.path.abspath(os.path.join(__file__, '../../../'))


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
