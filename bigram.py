from collections import Counter
from functools import partial
# import preprocessing as pre
from nltk import ngrams, word_tokenize

def clean(df):
    stop_words = pre.get_stop_words()

def bigrams_from_text(df):

for i in range(1, 4):
    _ngrams = partial(ngrams, n=i)
    df['{}-grams'.format(i)] = df['text'].apply(lambda x: Counter(_ngrams(word_tokenize(x))))



# todo: add to utils read_csv to df function, build a bigram from text column- needs to remove stop words. finish article