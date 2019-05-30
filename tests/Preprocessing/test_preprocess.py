import source.Preprocessing.preprocessing as pre
import pandas as pd


def test_correct_preprocess():
    raw_df = pd.DataFrame([[1, 'מילה, עוד מילה. :) english word סימן קריאה! שני סימני קריאה!! > אנגלית מחוברenglish', 0]],
                          columns=['id', 'text', 'cb_level'])
    clean_df = pre.preprocess(raw_df)
    text = clean_df['text'].tolist()[0]
    assert text == 'מילה עוד מילה סימן קריאה! שני סימני קריאה!! אנגלית מחובר '


test_correct_preprocess()