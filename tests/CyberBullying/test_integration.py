import pytest
import os.path
import numpy as np
import pandas as pd
from source.CyberBullying.Preprocessing import preprocessing as pre
from source.CyberBullying.Performances import performances as per
import source.CyberBullying.FeatureExtraction.featureExtraction as fe
import source.CyberBullying.TraditionalMLArchitecture.RandomForest as rf
from source.CyberBullying.Explainability import explanation
from source.CyberBullying import utils
ROOT = os.path.abspath(os.path.join(__file__, '../../../'))
FEATURE_LIST = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']


def test_preprocessing():
    try:
        tagged_df = utils.read_to_df(ROOT + '/source/CyberBullying/dataNew.csv')
        assert isinstance(pre.preprocess(tagged_df), pd.DataFrame)
    except Exception:
        pytest.fail('Unexpected error..')


def test_feature_extraction():
    try:
        tagged_df = utils.read_to_df(ROOT + '/source/CyberBullying/dataNew.csv')
        pre.preprocess(tagged_df)
        assert isinstance(fe.extract_features(tagged_df, FEATURE_LIST), pd.DataFrame)
    except Exception:
        pytest.fail('Unexpected error..')


def test_classicifation():
    try:
        tagged_df = utils.read_to_df(ROOT + '/source/CyberBullying/dataNew.csv')
        pre.preprocess(tagged_df)
        X = fe.extract_features(tagged_df, FEATURE_LIST)
        y = (tagged_df['cb_level'] == 3).astype(int)
        X = X.drop(columns=['id'])
        rf_obj = rf.RandomForest()
        rf_obj.train(X, y)
        rf_obj.predict(X)
        assert os.path.isfile(ROOT + '/source/CyberBullying/outputs/RandomForest.pkl')
    except Exception:
        pytest.fail('Unexpected error..')


def test_performance():
    model = utils.get_model(ROOT + '/source/CyberBullying/outputs/RandomForest.pkl')
    rf_obj = rf.RandomForest()
    rf_obj.model = model
    df = utils.read_to_df(ROOT + '/source/CyberBullying/dataNew.csv')
    df = pre.preprocess(df)
    X = fe.extract_features(df, FEATURE_LIST)
    X = X.drop(columns=['id'])
    y = (df['cb_level'] == 3).astype(int)
    y_prob_rf = rf_obj.predict(X)
    pred = np.where(y_prob_rf > 0.5, 1, 0)
    performance = per.get_performances(y, pred)
    assert 'f-score' in performance.keys()
    assert 'recall' in performance.keys()
    assert 'precision' in performance.keys()


def test_explainability():
    try:
        model = utils.get_model(ROOT + '/source/CyberBullying/outputs/RandomForest.pkl')
        tagged_df = utils.read_to_df(ROOT + '/source/CyberBullying/dataNew.csv')
        tagged_df = pre.preprocess(tagged_df)
        X = fe.extract_features(tagged_df, FEATURE_LIST)
        X = X.drop(columns=['id'])
        explanation.explain_model(model, X)
        explanation.explain_class(model, X)
    except Exception:
        pytest.fail('Unexpected error..')

