import os
import numpy

from source.CyberBullying.Preprocessing import preprocessing as pre
import pandas as pd
import source.CyberBullying.FeatureExtraction.featureExtraction as fe
import source.CyberBullying.TraditionalMLArchitecture.RandomForest as rf
from source.CyberBullying import utils

ROOT = os.path.abspath(os.path.join(__file__, '../../../../'))


def test_correct_classification():
    model = utils.get_model(ROOT + '/source/CyberBullying/outputs/RandomForest.pkl')
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


def test_train_model():
    rf_obj = rf.RandomForest()
    tagged_df = utils.read_to_df(ROOT + '/source/CyberBullying/dataNew.csv')
    tagged_df = pre.preprocess(tagged_df)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(tagged_df, feature_list)
    X = X.drop(columns=['id'])
    y = (tagged_df['cb_level'] == 3).astype(int)
    rf_obj.train(X, y)
    assert rf_obj.model is not None
