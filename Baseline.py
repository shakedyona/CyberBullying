import Preprocessing.preprocessing as pre
import pandas as pd
import utils

path = 'data.csv'
offensive_words = 'offensive_words.csv'
cols = ['id', 'time', 'source', 'sub_source', 'writer', 'link', 'text', 'cb_level', 'comment_shared_post']
df = pd.read_csv(path, names=cols)
tagged_df = utils.get_tagged_posts(df)
df_offensive = pd.read_csv(offensive_words, names=['words'])
offensive = df_offensive['words'].tolist()
df_compare = pd.DataFrame(columns=['Text', 'Algorithm classify', 'Original classify'])
pre.find_df(tagged_df, 0.8)
for index, row in tagged_df.iterrows():
    print(row.text)
    tokens = pre.tokenize_post(str(row.text))
    if any(elem in tokens for elem in offensive):
        df_compare.append({'Text': row['text'], 'Algorithm classify': '3', 'Original classify': row['cb_level']})
    else:
        df_compare.append({'Text': row['text'], 'Algorithm classify': '1', 'Original classify': row['cb_level']})
print(df_compare)


# tagged_df = pre.remove_stop_words(tagged_df, pre.get_stop_words())
# todo: tokenize, rule for each text- if one of the words from offensive ia there. after- present the score for the model
# todo: update to the new data.csv