import source.CyberBullying.FeatureExtraction.statistics as sta
import pandas as pd


def test_correct_length():
    raw_df = pd.DataFrame([[1, 'מילה עוד מילה חמודה', 0], [2, 'מילה מטומטמת', 3]], columns=['id', 'text', 'cb_level'])
    test_dictionary = sta.get_post_length(raw_df)
    result = {1: 4, 2: 2}
    assert test_dictionary == result


test_correct_length()