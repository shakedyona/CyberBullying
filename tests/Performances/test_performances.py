import os
from source.Preprocessing import preprocessing as pre
import pandas as pd
import source.FeatureExtraction.featureExtraction as fe
import source.TraditionalMLArchitecture.RandomForest as rf
from source import utils
from source.Performances import performances as per
import numpy as np


def test_correct_performance():
    pass
    # model = utils.get_model(os.path.join('outputs', 'RandomForest.pkl'))
    # rf_obj = rf.RandomForest()
    # rf_obj.model = model
    #
    # df = utils.read_to_df(file_path)
    # df = pre.preprocess(df)
    # feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    # X = fe.extract_features(df, feature_list)
    # X = X.drop(columns=['id'])
    # y = (df['cb_level'] == 3).astype(int)
    # y_prob_rf = rf_obj.predict(X)
    # pred = np.where(y_prob_rf > 0.5, 1, 0)
    # return per.get_performances(y, pred)
    #
    #
    # post = ['מילה ועוד מילה']
    # tagged_df = pd.DataFrame({'id': [1], 'text': [post]})
    # tagged_df = pre.preprocess(tagged_df)
    # feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    # X = fe.extract_features(tagged_df, feature_list)
    # X = X.drop(columns=['id'])
    # y = (df['cb_level'] == 3).astype(int)
    # y_prob_rf = rf_obj.predict(X)
    # pred = np.where(y_prob_rf > 0.5, 1, 0)
    # per.get_performances(y, pred)
    # assert


test_correct_performance()
