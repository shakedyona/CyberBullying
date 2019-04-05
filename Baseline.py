import Preprocessing.preprocessing as pre
import pandas as pd
import utils
import Performances.performances as per


offensive_words = 'offensive_words.csv'  # todo: this in main?
tagged_df = utils.read_to_df()
df_offensive = pd.read_csv(offensive_words, names=['words'])
tagged_df = pre.preprocess(tagged_df)

offensive = df_offensive['words'].tolist()
df_compare = pd.DataFrame(columns=['Text', 'Algorithm classify', 'Original classify'])
for index, row in tagged_df.iterrows():
    tokens = pre.tokenize_post(str(row.text))
    if any(elem in tokens for elem in offensive):
        df_compare = df_compare.append({'Text': row['text'], 'Algorithm classify': '3', 'Original classify': row['cb_level']}, ignore_index = True)
    else:
        df_compare = df_compare.append({'Text': row['text'], 'Algorithm classify': '1', 'Original classify': row['cb_level']}, ignore_index = True)
f1, precision, recall = per.get_performances(df_compare['Original classify'].tolist(), df_compare['Algorithm classify'].tolist())
print(f1, precision, recall)  # todo: also in main
