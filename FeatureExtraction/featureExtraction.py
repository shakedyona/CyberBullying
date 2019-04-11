from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import FeatureExtraction.statistics as st
import pandas as pd
import numpy as np
from FeatureExtraction.LDA import create_LDA_model
import utils


def get_functions_dictionary():
    return {
        'tfidf': extract_tf_idf,
        'post_length': extract_post_length,
        'topics': extract_topics,
        'screamer': contains_screamer
    }


def extract_tf_idf(df):
    posts = df['text'].tolist()
    tfidf = TfidfVectorizer(stop_words=utils.get_stop_words(), ngram_range=(1, 2))
    X = tfidf.fit_transform(posts)
    svdT = TruncatedSVD(n_components=1)
    svdTFit = svdT.fit_transform(X)
    df_svd = pd.DataFrame(columns=['id', 'tfidf'])
    df_svd['id'] = df['id'].tolist()
    df_svd['tfidf'] = np.array(svdTFit).flatten()
    return df_svd


def extract_post_length(df):
    df_length = pd.DataFrame(columns=['id', 'post_length'])
    df_length['id'] = df['id'].tolist()
    df_length['post_length'] = df['text'].apply(lambda x: len(x))
    return df_length


def extract_topics(df):
    dt_matrix = create_LDA_model(df, 3, '')
    features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3'])
    features['id'] = df['id'].tolist()
    return features


def contains_screamer(df):
    df_contains = pd.DataFrame(columns=['id', 'screamer'])
    df_contains['id'] = df['id'].tolist()
    for index, row in df.iterrows():
        if '!' in row['text']:
            print(row)

    #
    # df_contains['screamer'] = df['text'].apply(lambda x: len(x))
    # # df_contains['screamer'] = df['text'].apply(lambda x: 1 if x.contains('!!') else 0)
    # if df[row['text'].str.contains('!!')]:
    #     df_contains['screamer'] = 1
    # else:
    #     df_contains['screamer'] = 0
    return df_contains


def extract_features(df, features):
    functions_dict = get_functions_dictionary()
    features_df = pd.DataFrame(columns=['id'])
    features_df['id'] = df['id'].tolist()
    for feature in features:
        features_df = pd.merge(features_df, functions_dict[feature](df), on='id')
    return features_df
