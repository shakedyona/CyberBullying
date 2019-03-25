import pandas as pd


def get_abusive_df(df):
    return df.loc[df['cb_level'] == '3']


def get_no_abusive_df(df):
    return df.loc[df['cb_level'] == '1']


def get_tagged_posts(df):
    return df.loc[(df['cb_level'] == '1') | (df['cb_level'] == '3')]


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


# def read_csv_to_df(path):
    # column_names = []
    # df = pd.read_csv(path)
    # return df
    # df.columns = column_names
