from source.CyberBullying.Preprocessing import preprocessing as pre
import pandas as pd
import pytest


def test_correct_preprocess():
    raw_df = pd.DataFrame([[1, 'מילה, עוד מילה. :) english @#$%^&* word סימן קריאה! שני סימני קריאה!! > אנגלית מחוברenglish', 0]],
                          columns=['id', 'text', 'cb_level'])
    clean_df = pre.preprocess(raw_df)
    text = clean_df['text'].tolist()[0]
    assert text == 'מילה עוד מילה סימן קריאה! שני סימני קריאה!! אנגלית מחובר '


def test_incorrect_column_dataset():
    with pytest.raises(ValueError):
        df = pd.DataFrame([[0, 0, 0, 0]], columns=['not', 'the', 'right', 'columns'])
        pre.preprocess(df)


def test_empty_dataset():
    with pytest.raises(ValueError):
        df = pd.DataFrame([], columns=['not', 'the', 'right', 'columns'])
        pre.preprocess(df)


def test_tokenize_post():
    post = 'ילד, כותב בתוכ - דלי'
    assert pre.tokenize_post(post) == ['ילד', ',', 'כותב', 'בתוכ', '-', 'דלי']
