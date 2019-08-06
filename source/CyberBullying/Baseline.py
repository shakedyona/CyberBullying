from .Preprocessing import preprocessing as pre
from . import utils
import pandas as pd


def run_baseline(tagged_df):
    """
    This function is responsible for running the baseline model.
    The function receives the tagged data and returns the classification of each record in the data.
    The baseline model returns classification 1 (offensive) if an abusive word exists from an abusive dictionary,
    otherwise classification 0 (non-offensive).
    """
    offensive = utils.get_offensive_words()
    df_compare = pd.DataFrame(columns=['Text', 'Algorithm classify', 'Original classify'])
    for index, row in tagged_df.iterrows():
        tokens = pre.tokenize_post(str(row.text))
        if any(elem in tokens for elem in offensive):
            df_compare = df_compare.append({'Text': row['text'], 'Algorithm classify': 1,
                                            'Original classify': row['cb_level']}, ignore_index=True)
        else:
            df_compare = df_compare.append({'Text': row['text'], 'Algorithm classify': 0,
                                            'Original classify': row['cb_level']}, ignore_index=True)
    return df_compare['Algorithm classify'].tolist()
