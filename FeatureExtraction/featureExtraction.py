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


folder_name = None


def get_functions_dictionary():
    return {
        'tfidf': extract_tf_idf,
        'post_length': extract_post_length,
        'topics': extract_topics,
        'screamer': contains_screamer,
        'words': extract_meaningful_words_distance,
        'offensive_distance': extract_distance_from_offensive,
        'not_offensive_distance': extract_distance_from_not_offensive,
        'wmd_offensive': extract_wmd_offensive,
        'wmd_not_offensive': extract_wmd_not_offensive

    }


def extract_wmd_offensive(df):
    df_wmd_offensive = pd.DataFrame(columns=['id', 'wmd_our_offensive_tf_idf'])
    df_wmd_offensive['id'] = df['id'].tolist()
    tf_idf_difference = get_meaningful_words_tf_idf_difference(df).sort_values(by=0, axis=1, ascending=False)
    offensive_words_tf_idf = tf_idf_difference.iloc[:, 0:20]
    offensive_words_tf_idf = list(offensive_words_tf_idf.columns.values)
    m_our = get_model(r"C:\Users\shake\PycharmProjects\CyberBullying_1\Embedding\our.corpus.word2vec.model")
    m_our.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
    df_wmd_offensive['wmd_our_offensive_tf_idf'] = df['text'].apply(
        lambda x:
        float("{0:.4f}".format(m_our.wmdistance(offensive_words_tf_idf, word_tokenize(x))))
    )
    #
    # for index, row in df.iterrows():
    #     post = row['text']
    #     post = word_tokenize(post)
    #     # wiki_distance_dic = float("{0:.4f}".format(m_wiki.wmdistance(offensive_words_dictionary, post)))
    #     our_distance_dic = float("{0:.4f}".format(m_our.wmdistance(offensive_words_dictionary, post)))
    #     # wiki_distance_tf_idf = float("{0:.4f}".format(m_wiki.wmdistance(offensive_words_tf_idf, post)))
    #     our_distance_tf_idf = float("{0:.4f}".format(m_our.wmdistance(offensive_words_tf_idf, post)))
    #     # df_wmd_offensive.set_value(index, 'wmd_wiki_offensive_dictionary', wiki_distance_dic)
    #     df_wmd_offensive.at[index, 'wmd_our_offensive_dictionary'] = our_distance_dic
    #     # df_wmd_offensive.set_value(index, 'wmd_wiki_offensive_tf_idf', wiki_distance_tf_idf)
    #     df_wmd_offensive.at[index, 'wmd_our_offensive_tf_idf'] = our_distance_tf_idf
    return df_wmd_offensive


def extract_wmd_not_offensive(df):
    df_wmd_not_offensive = pd.DataFrame(columns=['id', 'wmd_our_not_offensive_tf_idf'])
    df_wmd_not_offensive['id'] = df['id'].tolist()
    tf_idf_difference = get_meaningful_words_tf_idf_difference(df).sort_values(by=0, axis=1, ascending=False)
    not_offensive = tf_idf_difference.iloc[:, -20:-1]
    not_offensive_words_tf_idf = list(not_offensive.columns.values)
    m_our = get_model(r"C:\Users\shake\PycharmProjects\CyberBullying_1\Embedding\our.corpus.word2vec.model")
    m_our.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
    df_wmd_not_offensive['wmd_our_not_offensive_tf_idf'] = df['text'].apply(
        lambda x:
        float("{0:.4f}".format(m_our.wmdistance(not_offensive_words_tf_idf, word_tokenize(x))))
    )
    # for index, row in df.iterrows():
    #     post = row['text'].split()
    #     # wiki_distance_tf_idf = float("{0:.4f}".format(m_wiki.wmdistance(not_offensive_words_tf_idf, post)))
    #     our_distance_tf_idf = float("{0:.4f}".format(m_our.wmdistance(not_offensive_words_tf_idf, post)))
    #     # df_wmd_not_offensive.set_value(index, 'wmd_wiki_not_offensive_tf_idf', wiki_distance_tf_idf)
    #     df_wmd_not_offensive.at[index, 'wmd_our_not_offensive_tf_idf'] = our_distance_tf_idf
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


def extract_distance_from_offensive(df):
    tf_idf_difference = get_meaningful_words_tf_idf_difference(df).sort_values(by=0, axis=1, ascending=False)
    offensive = tf_idf_difference.iloc[:, 0:100]
    return get_distance_df(df, 'offensive_distance', offensive)


def extract_distance_from_not_offensive(df):
    tf_idf_difference = get_meaningful_words_tf_idf_difference(df).sort_values(by=0, axis=1, ascending=False)
    not_offensive = tf_idf_difference.iloc[:, -100:-1]
    return get_distance_df(df, 'not_offensive_distance', not_offensive)


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


def extract_features(df, features,myfolder):
    global folder_name
    folder_name = myfolder
    functions_dict = get_functions_dictionary()
    # features_df = pd.DataFrame(columns=['id'])
    features_df = pd.DataFrame(columns=['id','text'])  # todo
    features_df['id'] = df['id'].tolist()
    features_df['text'] = df['text'].tolist()  # todo
    for feature in features:
        features_df = pd.merge(features_df, functions_dict[feature](df), on='id')
    return features_df
