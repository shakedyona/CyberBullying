import source.utils as utils
import pandas as pd


def test_correct_offensive_data():
    raw_df = pd.DataFrame([[1, 'מילה עוד מילה חמודה', 0], [2, 'מילה עוד מילה מטומטמת', 3]], columns=['id', 'text', 'cb_level'])
    test_df = utils.get_abusive_df(raw_df)
    test_df.reset_index(inplace=True, drop=True)
    result = pd.DataFrame([[2, 'מילה עוד מילה מטומטמת', 3]], columns=['id', 'text', 'cb_level'])
    result.reset_index(inplace=True, drop=True)
    assert test_df.equals(result)


test_correct_offensive_data()