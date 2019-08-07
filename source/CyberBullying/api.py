from .Preprocessing import preprocessing as pre
from .FeatureExtraction import featureExtraction as fe
from .Performances import performances as per
from . import utils
from .Explainability import explanation as exp
from .TraditionalMLArchitecture import RandomForest as rf
import numpy as np
import pandas as pd
import shutil
import os
import pathlib


HERE = pathlib.Path(__file__).parent
SOURCE = os.path.abspath(os.path.join(__file__, '../'))
FEATURE_LIST = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']


def train(file_path):
    """
    get a path to a csv file with train set.
    the file should have an 'id' column, a 'text' column and a 'cb_level' column
    the 'id' is a unique identifier of the sample, the text should be in Hebrew
    the cb_level should be 1 for not offensive text and 3 for offensive text
    the function trains and builds the NLP and feature extraction models with the train set
    and saves them in the 'output' directory
    :param file_path:
    :return:
    """
    path_object = pathlib.Path(HERE / 'outputs')
    if path_object.exists():
        shutil.rmtree(HERE / 'outputs')
    os.makedirs(HERE / 'outputs')
    tagged_df = utils.read_to_df(file_path)
    tagged_df = pre.preprocess(tagged_df)
    X = fe.extract_features(tagged_df, FEATURE_LIST)
    y = (tagged_df['cb_level'] == 3).astype(int)
    X = X.drop(columns=['id'])
    rf_obj = rf.RandomForest()
    rf_obj.train(X, y)
    exp.explain_model(rf_obj.model, X)
    utils.save_model(rf_obj.model, os.path.join(HERE / 'outputs', 'RandomForest.pkl'))


def get_classification(post, explainability=True):
    """
    should be called after 'train' function is called or if all the necessary models are saved in the 'output' directory
    get a post, Hebrew text as a string
    get a boolean argument ('explainability') if an explanation of the classification is wanted - the default is True
    return class: 1 if the post is offensive and class: 0 if its not offensive
    :param post:
    :param explainability:
    :return:
    """
    if len(os.listdir(HERE / 'outputs')) == 0:
        return {'error': "Please train the model with train data set first.."}

    model = utils.get_model(os.path.join(HERE / 'outputs', 'RandomForest.pkl'))
    rf_obj = rf.RandomForest()
    rf_obj.model = model
    post_dataframe = pd.DataFrame({'id': [1], 'text': [post]})
    post_dataframe = pre.preprocess(post_dataframe)
    X = fe.extract_features(post_dataframe, FEATURE_LIST)
    X = X.drop(columns=['id'])
    y_prob_rf = rf_obj.predict(X)
    pred = np.where(y_prob_rf > 0.5, 1, 0)
    result = {'class': int(pred[0])}
    if explainability:
        exp.explain_class(model, X)
        result['explain'] = utils.get_image_string(os.path.join(SOURCE, 'outputs/force_plot_post.png'))

    return result


def get_performance(file_path):
    """
    get a path to a csv file.
    the file should have 'text' column, 'id' column and 'cb_level' column.
    'text' should be in Hebrew, cb_level should be 1 for not offensive content or 3 for offensive content
    the request should be sent after 'train' occurs
    or if all the necessary models are saved in the 'output' directory
    returns the precision, recall and f-measure and saves images of the model explanation in the 'output' directory
    :param file_path:
    :return:
    """
    model = utils.get_model(HERE / 'outputs/RandomForest.pkl')
    rf_obj = rf.RandomForest()
    rf_obj.model = model
    df = utils.read_to_df(file_path)
    df = pre.preprocess(df)
    X = fe.extract_features(df, FEATURE_LIST)
    X = X.drop(columns=['id'])
    y = (df['cb_level'] == 3).astype(int)
    y_prob_rf = rf_obj.predict(X)
    pred = np.where(y_prob_rf > 0.5, 1, 0)
    return per.get_performances(y, pred)
