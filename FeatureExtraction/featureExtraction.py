from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import FeatureExtraction.statistics as st
import pandas as pd
import Preprocessing.preprocessing as pre
from nltk.tokenize import word_tokenize
import numpy as np

import utils


def get_functions_dictionary():
    return {
        'tfidf': extract_tf_idf,
        'post_length': extract_post_length
    }


def extract_tf_idf(df):
    posts = df['text'].tolist()
    tfidf = TfidfVectorizer(stop_words=utils.get_stop_words(), max_df=0.85, ngram_range=(1, 2))
    X = tfidf.fit_transform(posts)
    svdT = TruncatedSVD(n_components=1)
    svdTFit = svdT.fit_transform(X)
    df_svd = pd.DataFrame(columns=['id', 'tfidf'])
    df_svd['id'] = df['id']
    df_svd['tfidf'] = np.array(svdTFit).flatten()
    return df_svd


def extract_post_length(df):
    df_length = pd.DataFrame(columns=['id', 'post_length'])
    df_length['id'] = df['id']
    df_length['post_length'] = df['text'].apply(lambda x: len(x))
    return df_length


def extract_feature(df, features):
    functions_dict = get_functions_dictionary()
    features_df = pd.DataFrame(columns=['id'])
    features_df['id'] = df['id']
    for feature in features:
        features_df = pd.merge(features_df, functions_dict[feature](df), on='id')
    return features_df
