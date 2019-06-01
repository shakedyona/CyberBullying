import os
import pathlib

import numpy

from source.Preprocessing import preprocessing as pre
import pandas as pd
import source.FeatureExtraction.featureExtraction as fe
import source.TraditionalMLArchitecture.RandomForest as rf
from source import utils
ROOT = os.path.abspath(os.path.join(__file__, '../../../'))


def test_correct_classification():
    model = utils.get_model(ROOT + '/source/outputs/RandomForest.pkl')
    rf_obj = rf.RandomForest()
    rf_obj.model = model
    post = 'אני אוהבת אותך'
    tagged_df = pd.DataFrame({'id': [1], 'text': [post]})
    tagged_df = pre.preprocess(tagged_df)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(tagged_df, feature_list)
    X = X.drop(columns=['id'])
    y_prob_rf = rf_obj.predict(X)
    my_prob = y_prob_rf[0]
    if my_prob is None:
        assert False
    assert isinstance(my_prob, numpy.float64)


test_correct_classification()
