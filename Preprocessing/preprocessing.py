from nltk.tokenize import word_tokenize
import re


def preprocess(df):
    """
    get a dataframe - clean and preprocess its data and return the result dataframe
    :param df: dataframe
    :return clean_df: dataframe
    """
    for index, row in df.iterrows():
        row['text'] = re.sub(r'[^א-ת]', ' ', ' '.join(word_tokenize(row['text']))).strip().rstrip()

    return df


