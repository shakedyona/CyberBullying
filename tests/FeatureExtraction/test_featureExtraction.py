import pandas as pd
import source.FeatureExtraction.featureExtraction as fe


def test_correct_features():
    raw_df = pd.DataFrame([[1, 'מילה עוד מילה חמודה', 0], [2, 'את מטומטמת', 3]], columns=['id', 'text', 'cb_level'])
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    test_df = fe.extract_features(raw_df, feature_list)
    result_cols = ['id', 'post_length', 'tfidf', 'T1', 'T2', 'T3', 'screamer', 'off_dis', 'not_off_dis']
    if test_df.isnull().values.any():
        assert False
    for col in result_cols:
        if col not in test_df.columns:
            assert False
    if test_df.iloc[0]['post_length'] != 4 or test_df.iloc[1]['post_length'] != 2:
        assert False
    if test_df.iloc[0]['tfidf'] < 0 or test_df.iloc[1]['tfidf'] < 0:
        assert False
    if test_df.iloc[0]['off_dis'] < 0 or test_df.iloc[1]['off_dis'] < 0:
        assert False
    if test_df.iloc[0]['not_off_dis'] < 0 or test_df.iloc[1]['not_off_dis'] < 0:
        assert False
    if test_df.iloc[0]['T1'] < 0 or test_df.iloc[1]['T1'] < 0:
        assert False
    if test_df.iloc[0]['T2'] < 0 or test_df.iloc[1]['T2'] < 0:
        assert False
    if test_df.iloc[0]['T3'] < 0 or test_df.iloc[1]['T3'] < 0:
        assert False
    assert True


test_correct_features()