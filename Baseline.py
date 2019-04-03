import Preprocessing.preprocessing as pre
import pandas as pd
import utils
import Performances.performances as per
import FeatureExtraction.statistics as sta
import FeatureExtraction.featureExtraction as fe
import Explainability.explanation as exp

path = 'data.csv'
offensive_words = 'offensive_words.csv'
cols = ['id', 'time', 'source', 'sub_source', 'writer', 'link', 'text', 'cb_level', 'comment_shared_post']
df = pd.read_csv(path, names=cols)
tagged_df = utils.get_tagged_posts(df)
df_offensive = pd.read_csv(offensive_words, names=['words'])
tagged_df = pre.preprocess(tagged_df)

X = fe.extract_feature(tagged_df, ['post_length', 'tfidf'])
y = (tagged_df['cb_level'] == '3').astype(int)
exp.explain_xgboost(X, y)

offensive = df_offensive['words'].tolist()
df_compare = pd.DataFrame(columns=['Text', 'Algorithm classify', 'Original classify'])
for index, row in tagged_df.iterrows():
    tokens = pre.tokenize_post(str(row.text))
    if any(elem in tokens for elem in offensive):
        df_compare = df_compare.append({'Text': row['text'], 'Algorithm classify': '3', 'Original classify': row['cb_level']}, ignore_index = True)
    else:
        df_compare = df_compare.append({'Text': row['text'], 'Algorithm classify': '1', 'Original classify': row['cb_level']}, ignore_index = True)
f1,prec,recall = per.get_performances(df_compare['Original classify'].tolist(),df_compare['Algorithm classify'].tolist())
print(f1,prec,recall)

# tagged_df = pre.remove_stop_words(tagged_df, pre.get_stop_words())
# todo: tokenize, rule for each text- if one of the words from offensive ia there. after- present the score for the model
# todo: update to the new data.csv