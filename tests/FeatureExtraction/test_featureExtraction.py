import pandas as pd
import source.FeatureExtraction.featureExtraction as fe


def test_correct_features():
    raw_df = pd.DataFrame([[1, 'מילה עוד מילה חמודה', 0], [2, 'מילה עוד מילה מטומטמת', 3]], columns=['id', 'text', 'cb_level'])
    feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
    test_df = fe.extract_features(raw_df, feature_list)
    result_cols = ['id', 'post_length', 'tfidf', 'T1', 'T2', 'T3', 'screamer', 'off_dis', 'not_off_dis']
    if test_df.isnull().values.any():
        assert False
    for col in result_cols:
        if col not in test_df.columns:
            assert False
    for index, row in test_df.iterrows():
        if row['post_length'] < 0 or (type(row['post_length']) is not int):
            assert False
        if row['tfidf'] < 0:
            assert False
        if row['T1'] < 0 or row['T2'] < 0 or row['T3'] < 0:
            assert False
        if row['off_dis'] < 0 or row['not_off_dis'] < 0 or row['not_off_dis'] < 0 or row['not_off_dis'] < 0:
            assert False
    assert True


test_correct_features()