from nltk.tokenize import word_tokenize
import utils
from collections import Counter
import re


def remove_stop_words(data_frame, my_stop_words):
    """
    tokenize all the words in the data frame and removes stop words
    :param data_frame:
    :param my_stop_words:
    :return: dictionary of words and their number of performances
    """
    # print("remove_stop_words")
    text = data_frame.text.tolist()
    all_words = " ".join(text)
    all_words = word_tokenize(all_words)
    for word in my_stop_words:
        if word in all_words:
            all_words.remove(word)

    top_words = dict(Counter(all_words))
    return top_words

# todo:


def clean_tokens(df):
    """
    clean all non-Hebrew characters from a given data frame.
    :param df:
    :return:
    """
    text = df.text.tolist()
    number_posts = len(text)
    for index_post in range(1, number_posts):
        text[index_post] = re.sub(r'[^א-ת]', ' ', text[index_post]).strip().rstrip()
    return text


def get_stop_words():
    """
    gets list of stop words from a file
    :return:
    """
    stop_words = utils.file_to_list(r'stop_words.txt')
    return stop_words


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
