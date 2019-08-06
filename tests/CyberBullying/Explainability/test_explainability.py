from source.CyberBullying import utils
from source.CyberBullying.Explainability import explanation
import source.CyberBullying.Preprocessing.preprocessing as pre
import source.CyberBullying.FeatureExtraction.featureExtraction as fe
import os.path
import pandas as pd
ROOT = os.path.abspath(os.path.join(__file__, '../../../../'))


def test_explain_dataset():
    model = utils.get_model(ROOT + '/source/CyberBullying/outputs/RandomForest.pkl')
    tagged_df = utils.read_to_df(ROOT + '/source/CyberBullying/dataNew.csv')
    tagged_df = pre.preprocess(tagged_df)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(tagged_df, feature_list)
    X = X.drop(columns=['id'])
    explanation.explain_model(model, X)
    assert os.path.isfile(ROOT + '/source/CyberBullying/outputs/dependence_plot.png')
    assert os.path.isfile(ROOT + '/source/CyberBullying/outputs/summary_plot_bar.png')
    assert os.path.isfile(ROOT + '/source/CyberBullying/outputs/summary_plot.png')


def test_explain_class():
    model = utils.get_model(ROOT + '/source/CyberBullying/outputs/RandomForest.pkl')
    post = 'אני אוהבת אותך'
    tagged_df = pd.DataFrame({'id': [1], 'text': [post]})
    tagged_df = pre.preprocess(tagged_df)
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    X = fe.extract_features(tagged_df, feature_list)
    X = X.drop(columns=['id'])
    explanation.explain_class(model, X)
    assert os.path.isfile(ROOT + '/source/CyberBullying/outputs/force_plot_post.png')
