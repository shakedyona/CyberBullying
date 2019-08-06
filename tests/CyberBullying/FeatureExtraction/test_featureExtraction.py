import pandas as pd
import source.CyberBullying.FeatureExtraction.featureExtraction as fe
from source.CyberBullying import utils
import pathlib
import pytest


def test_correct_features():
    raw_df = pd.DataFrame([[1, 'מילה עוד מילה חמודה', 0], [2, 'את מטומטמת', 3]], columns=['id', 'text', 'cb_level'])
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    test_df = fe.extract_features(raw_df, feature_list)
    expected_columns = ['id', 'post_length', 'tfidf', 'T1', 'T2', 'T3', 'screamer', 'off_dis', 'not_off_dis']
    if test_df.isnull().values.any():
        assert False
    for col in expected_columns:
        assert col in test_df.columns
    assert test_df.iloc[0]['post_length'] == 4
    assert test_df.iloc[1]['post_length'] == 2
    assert test_df.iloc[0]['tfidf'] > 0
    assert test_df.iloc[1]['tfidf'] > 0
    assert test_df.iloc[0]['off_dis'] > 0
    assert test_df.iloc[1]['off_dis'] > 0
    assert test_df.iloc[0]['not_off_dis'] > 0
    assert test_df.iloc[1]['not_off_dis'] > 0
    assert test_df.iloc[0]['T1'] > 0
    assert test_df.iloc[1]['T1'] > 0
    assert test_df.iloc[0]['T2'] > 0
    assert test_df.iloc[1]['T2'] > 0
    assert test_df.iloc[0]['T3'] > 0
    assert test_df.iloc[1]['T3'] > 0


def test_extract_post_length():
    test_df = pd.DataFrame([[1, 'טקסט באורך 4 מילים', 1]], columns=['id', 'text', 'cb_level'])
    df_result = fe.extract_post_length(test_df)
    assert df_result.shape[0] == 1
    assert df_result.iloc[0]['post_length'] == 4


def test_extract_screamer():
    test_df = pd.DataFrame([[1, 'טקסט עם סימן קריאה אחד!', 1],
                            [2, 'טקסט עם שני סימני קריאה אחד!!', 1],
                            [3, 'טקסט ללא סימני קריאה אחד', 1]],
                           columns=['id', 'text', 'cb_level'])
    df_result = fe.extract_screamer(test_df)
    assert df_result.shape[0] == 3
    assert df_result.iloc[0]['screamer'] == 0
    assert df_result.iloc[1]['screamer'] == 1
    assert df_result.iloc[2]['screamer'] == 0


def test_extract_topics():
    test_df = pd.DataFrame([[1, 'בדיקת 3 נושאים', 1]], columns=['id', 'text', 'cb_level'])
    expected_columns = ['id', 'T1', 'T2', 'T3']
    result_df = fe.extract_topics(test_df)
    if result_df.isnull().values.any():
        assert False
    for col in expected_columns:
        assert col in result_df.columns
    assert result_df.iloc[0]['T1'] > 0
    assert result_df.iloc[0]['T2'] > 0
    assert result_df.iloc[0]['T3'] > 0


def test_extract_tf_idf():
    test_df = pd.DataFrame([[1, 'בדיקת תדירויות', 1]], columns=['id', 'text', 'cb_level'])
    expected_columns = ['id', 'tfidf']
    result_df = fe.extract_tf_idf(test_df)
    if result_df.isnull().values.any():
        assert False
    for col in expected_columns:
        assert col in result_df.columns
    assert result_df.iloc[0]['tfidf'] > 0


def test_extract_features_with_bad_file():
    with pytest.raises(ValueError):
        test_df = pd.DataFrame(['not valid'], columns=["unknown"])
        feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
        fe.extract_features(test_df, feature_list)


def test_extract_features_with_wrong_feature_list():
    HERE = pathlib.Path(__file__).parents[3]

    test_df = utils.read_to_df(HERE / 'source/CyberBullying/dataNew.csv')
    with pytest.raises(ValueError):
        feature_list = ['wrong', 'feature', 'list']
        fe.extract_features(test_df, feature_list)
