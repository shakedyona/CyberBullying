import os
from source.Preprocessing import preprocessing as pre
import pandas as pd
import source.FeatureExtraction.featureExtraction as fe
import source.TraditionalMLArchitecture.RandomForest as rf
from source import utils


def test_correct_classification():
    model = utils.get_model(os.path.join('outputs', 'RandomForest.pkl'))
    rf_obj = rf.RandomForest()
    rf_obj.model = model
    post = ['מילה ועוד מילה']
    tagged_df = pd.DataFrame({'id': [1], 'text': [post]})
    tagged_df = pre.preprocess(tagged_df)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(tagged_df, feature_list)
    X = X.drop(columns=['id'])
    y_prob_rf = rf_obj.predict(X)
    assert type(y_prob_rf) is float and y_prob_rf is not None


test_correct_classification()
