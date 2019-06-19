from nltk.tokenize import word_tokenize
import re
import pandas as pd


def preprocess(df):
    """
    get a dataframe - keepersData and preprocess its data and return the result dataframe
    :param df: dataframe
    :return clean_df: dataframe
    """
    if 'text' not in df.columns and 'id' not in df.columns or df.shape[0] == 0:
        raise ValueError
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(
        lambda x:
        re.sub(r'[^א-ת!]+', ' ', x)
    )
    df.reset_index(drop=True, inplace=True)
    df['id'] = list(range(df.shape[0]))
    return df


def tokenize_df(df):
    """
    tokenize all the words in the dataframe
    :param df:
    :return list of tokens:
    """
    text = df.text.tolist()
    tokens = []
    for post in text:
        tokens = tokens + word_tokenize(post)
    return tokens


def tokenize_post(post):
    """
    tokenize only one post
    :param post:
    :return list of tokens:
    """
    return word_tokenize(post)
