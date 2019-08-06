import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from sklearn.externals import joblib
from .TraditionalMLArchitecture import XGBoost as xgb
from .TraditionalMLArchitecture import RandomForest as rf
from .TraditionalMLArchitecture import NaiveBayes as nb
import pathlib
import matplotlib.pyplot as plt
import base64
SOURCE = pathlib.Path(__file__).parent


def get_abusive_df(df):
    return df.loc[df['cb_level'] == 3]


def get_no_abusive_df(df):
    return df.loc[df['cb_level'] == 1]


def get_tagged_posts(df):
    return df.loc[(df['cb_level'] == 1) | (df['cb_level'] == 3)]


def file_to_list(path, encoding='cp1255'):
    """
    Insert file data into list
    :param path:
    :param encoding:
    :return:
    """
    try:
        with open(path, mode='r', encoding=encoding) as infile:
            myList = [line.strip('\n') for line in infile]
    except UnicodeDecodeError as e:
        with open(path,mode='r', encoding='utf-8') as infile:
            myList = [line.strip('\n') for line in infile]
    return myList


def traverse(word):
    """
    Traverse words from left to right
    :param word:
    :return word:
    """
    if type(word) is list:
        return [''.join(wrd[-1:-(len(wrd)+1):-1]) if type(wrd) is str and len(wrd)>0 and wrd[0] in 'אבגדהוזחטיכלמנסעפצקרשת' else wrd for wrd in word]
    elif type(word) is str: return traverse([word])[0]
    elif type(word) is set: return set(traverse(list(word)))
    elif type(word) is dict: dict(zip(traverse(word.keys()), traverse(word.values())))
    elif type(word) == type(pd.Series()): return pd.Series(data=traverse(list(word)), index=word.index, name=word.name)
    elif type(word) == type(type(pd.DataFrame())): return word.applymap(lambda x: traverse(x))
    return word


def get_offensive_words(offensive_words_file='offensive_words.csv'):
    """
    get a list of offensive word from offensive words file
    :return:
    """
    df_offensive = pd.read_csv(offensive_words_file, names=['words'])
    offensive = df_offensive['words'].tolist()
    return offensive


def get_stop_words():
    """
    gets list of stop words from a file
    :return:
    """
    stop_words = file_to_list(SOURCE / 'stop_words.txt')
    return stop_words


def create_stop_words_list(dataframe, threshold):
    """
    create list of frequent words according to a given threshold
    :param dataframe: dataframe
    :param threshold: double
    :return stop_words: list
    """
    stop_words = get_stop_words()
    text = dataframe.text.tolist()
    term_df = {}
    number_posts = len(text)

    for index_post in range(1, number_posts):
        tokens = word_tokenize(text[index_post])
        for token in tokens:
            if token in term_df:
                list_posts = term_df[token]
                if index_post not in list_posts:
                    term_df[token].append(index_post)  # change from indexes to counter
            else:
                term_df[token] = []
                term_df[token].append(index_post)

    for token, posts in term_df.items():
        df = len(posts)
        df_normal = float(df / number_posts)
        if df_normal > threshold:
            stop_words.append(token)

    return stop_words


def calculate_distance(a, b, distance_type='euclidean'):
    """
    calculate the distance between a and b according to a given type.
    the default type is euclidean but could be cosine.
    :param a:
    :param b:
    :param distance_type:
    :return:
    """
    if distance_type == 'euclidean':
        return distance.euclidean(a, b)
    if distance_type == 'cosine':
        return distance.cosine(a, b)
    return None


def read_to_df(path='dataNew.csv'):
    """
    reads the csv data file to a data frame and gets only the tagged post
    :return df:
    """
    try:
        df = pd.read_csv(path)
        if df.shape[0] == 0:
            raise ValueError
        return get_tagged_posts(df)
    except KeyError:
        raise ValueError


def create_dataframe_from_keepers_files(folder_path =r'keepersData'):
    """
    keeperes sends 2 files - one for the negative sentences and one for positive sentences.
    this function get the directory of those 2 files and convert them to the correct data frame
    :param folder_path:
    :return: dataframe
    """
    path_neg = folder_path + r"/neg.neg"
    path_pos = folder_path + r"/pos.pos"
    cols = ['id', 'text', 'cb_level']
    df_neg = pd.DataFrame(columns=cols)
    df_pos = pd.DataFrame(columns=cols)
    with open(path_neg, 'r', encoding="utf8") as f:
        df_neg['text'] = f.readlines()
    with open(path_pos, 'r', encoding="utf8") as f:
        df_pos['text'] = f.readlines()
    df_neg['cb_level'] = 3
    df_pos['cb_level'] = 1

    df_neg = df_neg.reset_index(drop=True)
    df_pos = df_pos.reset_index(drop=True)
    df = pd.concat([df_neg, df_pos],axis=0,ignore_index=True)
    df['id'] = list(range(df.shape[0]))
    return df


def save_model(model, path='outputs/model.pkl'):
    """
    save models to a pkl file
    :param model:
    :param path:
    :return:
    """
    joblib.dump(model, path)


def get_model(path='outputs/model.pkl'):
    """
    return a model from a pkl file
    :param path:
    :return:
    """
    path_object = pathlib.Path(path)
    if path_object.exists():
        return joblib.load(path)
    return None


def create_list_of_models():
    """
    return list of tuples where in the 0 index is the model name and in the 1 index is the machine learning model itself
    :return:
    """
    return [('NB', nb.NaiveBayes().model), ('RF', rf.RandomForest().model), ('XGB', xgb.XGBoost().model)]


def save_picture(path):
    """
    save plot to a file
    :param path:
    :return:
    """
    plt.savefig(path)


def clear_plot():
    """
    clear the plt from plot
    :return:
    """
    plt.clf()


def get_image_string(path):
    """
    convert image to a base64 string
    :param path:
    :return:
    """
    with open(path, mode='rb') as file:
        img = file.read()
    return base64.encodebytes(img).decode("utf-8")
