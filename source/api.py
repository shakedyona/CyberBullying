from source.Preprocessing import preprocessing as pre
from source.FeatureExtraction import featureExtraction as fe
from source.Performances import performances as per
from source import utils
from source.Explainability import explanation as exp
import source.TraditionalMLArchitecture.RandomForest as rf
import numpy as np
import pandas as pd
import shutil
import os
import pathlib


def train_file(file_path):
    path_object = pathlib.Path('source/outputs')
    if path_object.exists():
        shutil.rmtree('source/outputs')
        os.makedirs('source/outputs')
    tagged_df = utils.read_to_df(file_path)
    tagged_df = pre.preprocess(tagged_df)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(tagged_df, feature_list)
    y = (tagged_df['cb_level'] == 3).astype(int)
    X = X.drop(columns=['id'])
    rf_obj = rf.RandomForest()
    rf_obj.train(X, y)
    utils.save_model(rf_obj.model, os.path.join('source/outputs', 'RandomForest.pkl'))


def predict(post, explainability=True):
    model = utils.get_model(os.path.join('source/outputs', 'RandomForest.pkl'))
    rf_obj = rf.RandomForest()
    rf_obj.model = model
    tagged_df = pd.DataFrame({'id': [1], 'text': [post]})
    tagged_df = pre.preprocess(tagged_df)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(tagged_df, feature_list)
    X = X.drop(columns=['id'])
    y_prob_rf = rf_obj.predict(X)
    pred = np.where(y_prob_rf > 0.5, 1, 0)
    result = {'class': pred}
    if explainability:
        result['explain'] = exp.explain_class(post)
    return result


def get_performances(file_path):
    model = utils.get_model()
    rf_obj = rf.RandomForest()
    rf_obj.model = model
    df = utils.read_to_df(file_path)
    df = pre.preprocess(df)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(df, feature_list)
    X = X.drop(columns=['id'])
    y = (df['cb_level'] == 3).astype(int)
    y_prob_rf = rf_obj.predict(X)
    pred = np.where(y_prob_rf > 0.5, 1, 0)
    return per.get_performances(y, pred)
