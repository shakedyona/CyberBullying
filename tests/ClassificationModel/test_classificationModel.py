import os
import pathlib

from source.Preprocessing import preprocessing as pre
import pandas as pd
import source.FeatureExtraction.featureExtraction as fe
import source.TraditionalMLArchitecture.RandomForest as rf
from source import utils
HERE = pathlib.Path(__file__).parent


def test_correct_classification():
    model = utils.get_model(HERE / 'outputs/RandomForest.pkl')
    rf_obj = rf.RandomForest()
    rf_obj.model = model
    post = ['מילה ועוד מילה']
    tagged_df = pd.DataFrame({'id': [1], 'text': [post]})
    df = pre.preprocess(tagged_df)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(df, feature_list)
    X = X.drop(columns=['id'])
    y = (df['cb_level'] == 3).astype(int)
    y_prob_rf = rf_obj.predict(X)
    pred = np.where(y_prob_rf > 0.5, 1, 0)

    post = ['מילה ועוד מילה']
    tagged_df = pd.DataFrame({'id': [1], 'text': [post]})
    tagged_df = pre.preprocess(tagged_df)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(tagged_df, feature_list)
    X = X.drop(columns=['id'])
    y_prob_rf = rf_obj.predict(X)
    assert type(y_prob_rf) is float and y_prob_rf is not None


test_correct_classification()
