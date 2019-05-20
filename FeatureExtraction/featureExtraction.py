from nltk import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from FeatureExtraction.LDA import create_LDA_model
import utils
import Embedding.word2vec as wv
from Embedding.word2vec import get_model
import gensim.models.keyedvectors as word2vec
from sklearn.feature_extraction.text import CountVectorizer

folder_name = None


def get_functions_dictionary():
    return {
        'tfidf': extract_tf_idf,
        'post_length': extract_post_length,
        'topics': extract_topics,
        'screamer': contains_screamer,
        'words': extract_meaningful_words_distance,
        'off_dis': extract_distance_from_offensive,
        'not_off_dis': extract_distance_from_not_offensive,
        'wmd_off': extract_wmd_offensive,
        'wmd_not_off': extract_wmd_not_offensive

    }


def extract_wmd_offensive(df):
    df_wmd_offensive = pd.DataFrame(columns=['id', 'wmd_off_tfidf'])
    df_wmd_offensive['id'] = df['id'].tolist()
    tf_idf_difference = get_meaningful_words_tf_idf_difference(df).sort_values(by=0, axis=1, ascending=False)
    offensive_words_tf_idf = tf_idf_difference.iloc[:, 0:20]
    offensive_words_tf_idf = list(offensive_words_tf_idf.columns.values)
    m_our = get_model(r"C:\Users\shake\PycharmProjects\CyberBullying_1\Embedding\our.corpus.word2vec.model")
    m_our.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
    df_wmd_offensive['wmd_off_tfidf'] = df['text'].apply(
        lambda x:
        float("{0:.4f}".format(m_our.wmdistance(offensive_words_tf_idf, word_tokenize(x))))
    )
    return df_wmd_offensive


def extract_wmd_not_offensive(df):
    df_wmd_not_offensive = pd.DataFrame(columns=['id', 'wmd_not_off_tfidf'])
    df_wmd_not_offensive['id'] = df['id'].tolist()
    tf_idf_difference = get_meaningful_words_tf_idf_difference(df).sort_values(by=0, axis=1, ascending=False)
    not_offensive = tf_idf_difference.iloc[:, -20:-1]
    not_offensive_words_tf_idf = list(not_offensive.columns.values)
    m_our = get_model(r"C:\Users\shake\PycharmProjects\CyberBullying_1\Embedding\our.corpus.word2vec.model")
    m_our.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
    df_wmd_not_offensive['wmd_not_off_tfidf'] = df['text'].apply(
        lambda x:
        float("{0:.4f}".format(m_our.wmdistance(not_offensive_words_tf_idf, word_tokenize(x))))
    )
    return df_wmd_not_offensive


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
    # for index, row in df.iterrows():
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


def extract_distance_from_offensive(df):
    tf_idf_difference = get_meaningful_words_tf_idf_difference(df).sort_values(by=0, axis=1, ascending=False)
    offensive = tf_idf_difference.iloc[:, 0:100]
    return get_distance_df(df, 'off_dis', offensive)


def extract_distance_from_not_offensive(df):
    tf_idf_difference = get_meaningful_words_tf_idf_difference(df).sort_values(by=0, axis=1, ascending=False)
    not_offensive = tf_idf_difference.iloc[:, -100:-1]
    return get_distance_df(df, 'not_off_dis', not_offensive)


def get_meaningful_words_tf_idf_difference(df):
    df_neg = utils.get_abusive_df(df)
    df_pos = utils.get_no_abusive_df(df)
    posts = [' '.join(df_neg['text'].tolist()), ' '.join(df_pos['text'].tolist())]

    tfidf = TfidfVectorizer(stop_words=utils.get_stop_words(), ngram_range=(1, 2))
    x = tfidf.fit_transform(posts)
    x = x[0,:] - x[1,:]
    df_tf_idf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
    return df_tf_idf


def get_distance_df(df, column_name, words_difference, distance_type='euclidean'):
    words = list(words_difference.columns.values)
    df_offensive_distance = pd.DataFrame(columns=['id', column_name])
    df_offensive_distance['id'] = df['id'].tolist()

    m_wiki = get_model(r"C:\Users\shake\PycharmProjects\CyberBullying_1\Embedding\wiki.he.word2vec.model")
    m_our = get_model(r"C:\Users\shake\PycharmProjects\CyberBullying_1\Embedding\our.corpus.word2vec.model")

    df_offensive_distance[column_name] = df['text'].apply(
        lambda x:
        utils.calculate_distance(wv.get_post_vector(m_our, m_wiki, x),
                                 wv.get_post_vector(m_our, m_wiki, ' '.join(words)), distance_type)
    )
    return df_offensive_distance


def get_vectorizing(df):
    posts = df['text'].values
    vect = CountVectorizer(max_df=0.6, min_df=0.01, stop_words=utils.get_stop_words())
    vect.fit(posts)
    simple_train_dtm = vect.transform(posts)
    df_vectorizing = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
    features_df = pd.DataFrame(columns=['id'])
    features_df['id'] = df['id'].tolist()
    features_df = pd.merge(features_df, df_vectorizing, on='id')

def extract_features(df, features,myfolder):
    global folder_name
    folder_name = myfolder
    functions_dict = get_functions_dictionary()
    features_df = pd.DataFrame(columns=['id'])
    # features_df = pd.DataFrame(columns=['id','text'])  # todo
    # features_df = pd.DataFrame(columns=['id', 'text', 'cb_level'])  # todo
    features_df['id'] = df['id'].tolist()
    # features_df['text'] = df['text'].tolist()  # todo
    # features_df['cb_level'] = df['cb_level'].tolist()  # todo
    for feature in features:
        features_df = pd.merge(features_df, functions_dict[feature](df), on='id')
    return features_df
