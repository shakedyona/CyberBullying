from nltk.tokenize import word_tokenize
import re


def preprocess(df):
    """
    get a dataframe - keepersData and preprocess its data and return the result dataframe
    :param df: dataframe
    :return clean_df: dataframe
    """
    for index, row in df.iterrows():
        row['text'] = re.sub(r'^[א-ת]', ' ', ' '.join(word_tokenize(row['text']))).strip().rstrip()

    df.drop_duplicates(subset='id', inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
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
