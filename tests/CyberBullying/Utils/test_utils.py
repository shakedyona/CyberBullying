from source.CyberBullying import utils
import pandas as pd
import pathlib
import pytest


def test_correct_offensive_data():
    raw_df = pd.DataFrame([[1, 'מילה עוד מילה חמודה', 1], [2, 'מילה עוד מילה מטומטמת', 3]],
                          columns=['id', 'text', 'cb_level'])
    test_df = utils.get_abusive_df(raw_df)
    test_df.reset_index(inplace=True, drop=True)
    result = pd.DataFrame([[2, 'מילה עוד מילה מטומטמת', 3]], columns=['id', 'text', 'cb_level'])
    result.reset_index(inplace=True, drop=True)
    assert test_df.equals(result)


def test_correct_not_offensive_data():
    raw_df = pd.DataFrame([[1, 'מילה עוד מילה חמודה', 1], [2, 'מילה עוד מילה מטומטמת', 3]],
                          columns=['id', 'text', 'cb_level'])
    test_df = utils.get_no_abusive_df(raw_df)
    test_df.reset_index(inplace=True, drop=True)
    result = pd.DataFrame([[1, 'מילה עוד מילה חמודה', 1]], columns=['id', 'text', 'cb_level'])
    result.reset_index(inplace=True, drop=True)
    assert test_df.equals(result)


def test_traverse():
    assert utils.traverse('מילה') == 'הלימ'


def test_read_data_correctly():
    HERE = pathlib.Path(__file__).parents[3]

    data = utils.read_to_df(HERE / 'source/CyberBullying/dataNew.csv')
    assert 'text' in data.columns
    assert 'cb_level' in data.columns
    assert data.shape[0] > 0


def test_read_data_from_unknown_path():
    with pytest.raises(FileNotFoundError):
        utils.read_to_df('unknown/path')


def test_read_data_with_bad_file():
    HERE = pathlib.Path(__file__).parents[1]
    with pytest.raises(ValueError):
        utils.read_to_df(HERE / 'badFile.csv')
