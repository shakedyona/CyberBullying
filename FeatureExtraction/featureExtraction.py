from nltk import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import FeatureExtraction.statistics as st
import pandas as pd
import numpy as np
from FeatureExtraction.LDA import create_LDA_model
import utils

folder_name = None

def get_functions_dictionary():
    return {
        'tfidf': extract_tf_idf,
        'post_length': extract_post_length,
        'topics': extract_topics,
        'screamer': contains_screamer,
        'words': extract_meaningful_words_distance
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
    df_length['post_length'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    return df_length


def extract_topics(df):
    dt_matrix = create_LDA_model(df, 3, '',folder_name)
    features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3'])
    features['id'] = df['id'].tolist()
    return features


def contains_screamer(df):
    df_contains = pd.DataFrame(columns=['id', 'screamer'])
    df_contains['id'] = df['id'].tolist()
    for index, row in df.iterrows():
        df_contains['screamer'] = df['text'].apply(lambda x: 1 if '!!' in x else 0)
    return df_contains


def extract_meaningful_words_distance(df):
    tf_idf_difference = get_meaningful_words_tf_idf_difference(df).sort_values(by=0, axis=1, ascending=False)
    top_20 = tf_idf_difference.iloc[:, 0:20]
    df_abusive_words = pd.DataFrame(columns=['id'].__add__(list(top_20.columns.values)))
    df_abusive_words['id'] = df['id'].tolist()
    for word in list(top_20.columns.values):
        df_abusive_words[word] = df['text'].apply(lambda x: 1 if word in x else 0)
    return df_abusive_words


def extract_features(df, features,myfolder):
    global folder_name
    folder_name = myfolder
    functions_dict = get_functions_dictionary()
    features_df = pd.DataFrame(columns=['id'])
    # features_df = pd.DataFrame(columns=['id','text'])
    features_df['id'] = df['id'].tolist()
    #features_df['text'] = df['text'].tolist()  # todo
    for feature in features:
        features_df = pd.merge(features_df, functions_dict[feature](df), on='id')
    return features_df


def get_meaningful_words_tf_idf_difference(df):
    df_neg = utils.get_abusive_df(df)
    df_pos = utils.get_no_abusive_df(df)
    posts = [' '.join(df_neg['text'].tolist()), ' '.join(df_pos['text'].tolist())]

    tfidf = TfidfVectorizer(stop_words=utils.get_stop_words(), ngram_range=(1, 2))
    x = tfidf.fit_transform(posts)
    x = x[0,:] - x[1,:]
    df_tf_idf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
    return df_tf_idf
