import Preprocessing.preprocessing as pre
import pandas as pd
import utils
import Performances.performances as per


def run_baseline(tagged_df):
    # offensive_words = 'offensive_words.csv'  # todo: this in main?
    # df_offensive = pd.read_csv(offensive_words, names=['words'])
    # offensive = df_offensive['words'].tolist()
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
