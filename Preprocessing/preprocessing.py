from nltk.tokenize import word_tokenize
import re


def preprocess(df):
    """
    get a dataframe - clean and preprocess its data and return the result dataframe
    :param df: dataframe
    :return clean_df: dataframe
    """
    return clean_tokens(df)


def clean_tokens(df):
    """
    clean all non-Hebrew characters from a given data frame.
    :param df:
    :return:
    """
    for index, row in df.iterrows():
        row['text'] = re.sub(r'[^א-ת]', ' ', row['text']).strip().rstrip()

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


def get_post_length(dataframe):
    """
    gets length of all posts
    :param dataframe:
    :return:
    """
    post_frequency = {}
    for index, row in dataframe.iterrows():
        post_frequency[row.id] = len(word_tokenize(row.text))

    return post_frequency

