from source.Preprocessing import preprocessing as pre
from source.FeatureExtraction import featureExtraction as fe
from source.Performances import performances as per
from source import utils
from source.Explainability import explanation as exp
import source.TraditionalMLArchitecture.RandomForest as rf
from source import visualization as vis
from source import Logger
import numpy as np
import pandas as pd
import shutil
import os
import pathlib


HERE = pathlib.Path(__file__).parent


def train_file(file_path):
    path_object = pathlib.Path(HERE / 'outputs')
    if path_object.exists():
        shutil.rmtree(HERE / 'outputs')
    os.makedirs(HERE / 'outputs')
    tagged_df = utils.read_to_df(file_path)
    tagged_df = pre.preprocess(tagged_df)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(tagged_df, feature_list)
    y = (tagged_df['cb_level'] == 3).astype(int)
    X = X.drop(columns=['id'])
    rf_obj = rf.RandomForest()
    rf_obj.train(X, y)
    exp.explain_model(rf_obj.model, X)
    utils.save_model(rf_obj.model, os.path.join(HERE / 'outputs', 'RandomForest.pkl'))


def predict(post, explainability=True):
    if len(os.listdir(HERE / 'outputs')) == 0:
        return {'error': "Please train the model with train data set first.."}

    model = utils.get_model(os.path.join(HERE / 'outputs', 'RandomForest.pkl'))
    rf_obj = rf.RandomForest()
    rf_obj.model = model
    post_dataframe = pd.DataFrame({'id': [1], 'text': [post]})
    post_dataframe = pre.preprocess(post_dataframe)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(post_dataframe, feature_list)
    X = X.drop(columns=['id'])
    y_prob_rf = rf_obj.predict(X)
    pred = np.where(y_prob_rf > 0.5, 1, 0)
    result = {'class': int(pred[0])}
    if explainability:
        result['explain'] = exp.explain_class(model, X)
    return result


def get_performances(file_path):
    model = utils.get_model(HERE / 'outputs/RandomForest.pkl')
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

    # roc_auc_rf, fpr_rf, tpr_rf = per.get_roc_auc(y, y_prob_rf)
    # vis.plot_roc_curve(roc_auc_rf, fpr_rf, tpr_rf, 'random forest')
    # performances_rf = per.get_performances(y, pred)
    # logger = Logger.get_logger_instance()
    # performances_list = {}
    # auc_list = {}
    # auc_list['Random forest'] = roc_auc_rf
    # performances_list['Random forest'] = performances_rf
    # logger.write_performances(auc_list, performances_list)
    return per.get_performances(y, pred)
