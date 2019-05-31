from nltk import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from source.FeatureExtraction.LDA import create_LDA_model
from source.Embedding import word2vec as wv
from source import utils
from source.Embedding.word2vec import get_model, get_post_vector

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
        'wmd_not_off': extract_wmd_not_offensive,
        'dis_avg_vec': extract_distance_from_avg_vector
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
    svdT = TruncatedSVD(n_components=1)
    X = tfidf.fit_transform(posts)
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

    m_wiki = get_model(r"Embedding/wiki.he.word2vec.model")
    m_our = get_model(r"Embedding/our.corpus.word2vec.model")

    df_offensive_distance[column_name] = df['text'].apply(
        lambda x:
        utils.calculate_distance(wv.get_post_vector(m_our, m_wiki, x),
                                 wv.get_post_vector(m_our, m_wiki, ' '.join(words)), distance_type)
    )
    return df_offensive_distance


def extract_distance_from_avg_vector(df):
    df_neg = utils.get_abusive_df(df)
    df_pos = utils.get_no_abusive_df(df)
    m_wiki = get_model(r"Embedding/wiki.he.word2vec.model")
    m_our = get_model(r"Embedding/our.corpus.word2vec.model")

    pos_matrix = np.zeros((df_pos.shape[0], 100))
    neg_matrix = np.zeros((df_neg.shape[0], 100))
    for i, post in enumerate(df_pos['text']):
        embedding_vector = get_post_vector(m_our, m_wiki, post)
        pos_matrix[i] = embedding_vector
    for i, post in enumerate(df_neg['text']):
        embedding_vector = get_post_vector(m_our, m_wiki, post)
        neg_matrix[i] = embedding_vector
    neg_avg_vec = np.mean(neg_matrix)
    pos_avg_vec = np.mean(pos_matrix)
    distance_type='euclidean'
    df_offensive_distance = pd.DataFrame(columns=['id', 'dist_avg_neg', 'dist_avg_pos'])
    df_offensive_distance['id'] = df['id'].tolist()
    df_offensive_distance['dist_avg_neg'] = df['text'].apply(
        lambda x:
        utils.calculate_distance(get_post_vector(m_our, m_wiki, x),
                                 neg_avg_vec, distance_type)
    )
    df_offensive_distance['dist_avg_pos'] = df['text'].apply(
        lambda x:
        utils.calculate_distance(get_post_vector(m_our, m_wiki, x),
                                 pos_avg_vec, distance_type)
    )
    return df_offensive_distance


def extract_features(df, features):
    functions_dict = get_functions_dictionary()
    features_df = pd.DataFrame(columns=['id'])
    features_df['id'] = df['id'].tolist()
    for feature in features:
        features_df = pd.merge(features_df, functions_dict[feature](df), on='id')
    return features_df


