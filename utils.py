import pandas as pd
from nltk.tokenize import word_tokenize


def get_abusive_df(df):
    return df.loc[df['cb_level'] == 3]


def get_no_abusive_df(df):
    return df.loc[df['cb_level'] == 1]


def get_tagged_posts(df):
    return df.loc[(df['cb_level'] == 1) | (df['cb_level'] == 3)]


def file_to_list(path, encoding='cp1255', header=True):
    '''
    Insert file data into list
    :param path:
    :param encoding:
    :param header:
    :return:
    '''
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


def get_stop_words():
    """
    gets list of stop words from a file
    :return:
    """
    stop_words = file_to_list(r'stop_words.txt')
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
                    term_df[token].append(index_post)  # todo: change from indexes to counter
            else:
                term_df[token] = []
                term_df[token].append(index_post)

    for token, posts in term_df.items():
        df = len(posts)
        df_normal = float(df / number_posts)
        if df_normal > threshold:
            stop_words.append(token)

    return stop_words


def read_to_df():
    """
    reads the csv data file to a data frame and gets only the tagged post
    :return df:
    """
    path = 'dataNew.csv'
    cols = ['id', 'time', 'source', 'sub_source', 'writer', 'link', 'text', 'cb_level']
    df = pd.read_csv(path, names=cols, header=0)
    return get_tagged_posts(df)


def create_csv_from_keepers_files(folder_path = 'keepersData'):
    path_neg = folder_path + '/sentences.neg'
    path_pos = folder_path + '/sentences.pos'
    cols = ['id', 'text', 'cb_level']
    df_neg = pd.DataFrame(columns=cols)
    df_pos = pd.DataFrame(columns=cols)
    with open(path_neg, 'r') as f:
        df_neg['text'] = f.readlines()
    with open(path_pos, 'r') as f:
        df_pos['text'] = f.readlines()
    df_neg['cb_level'] = '3'
    df_pos['cb_level'] = '1'

    df_neg = df_neg.reset_index(drop=True)
    df_pos = df_pos.reset_index(drop=True)
    df = pd.concat([df_neg, df_pos],axis=0,ignore_index=True)
    df['id'] = list(range(df.shape[0]))
    return df


