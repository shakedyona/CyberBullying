from gensim.corpora import WikiCorpus
from nltk.tokenize import word_tokenize
import numpy as np
from Embedding.word2vec import get_model, train
import csv
import utils


def create_wiki_corpus(inp="hewiki-latest-pages-articles.xml.bz2", outp="wiki3.he.text"):
    print("Starting to create wiki corpus")
    output = open(outp, 'w')
    space = " "
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for i, text in enumerate(wiki.get_texts()):
        article = space.join([t for t in text])

        output.write("{}\n".format(article))
        if i % 1000 == 0:
            print("Saved " + str(i) + " articles")

    output.close()


def get_word_vector(model, word):
    return model.wv[word]


def get_post_vector(our_word2vec_model, wiki_word2vec_model, post):
    post = word_tokenize(post)
    # remove out-of-vocabulary words
    post = [word for word in post if word in our_word2vec_model.wv.vocab]
    if len(post) == 0:
        post = [word for word in post if word in wiki_word2vec_model.wv.vocab]
        if len(post) == 0:
            raise ValueError('words not in vocabulary')
        return np.mean(wiki_word2vec_model[post], axis=0)

    return np.mean(our_word2vec_model[post], axis=0)


def create_our_corpus(inp='corpus.csv', outp='corpus.txt'):
    with open(inp) as csvfile, open(outp, 'w') as txtFile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            txtFile.write(' '.join(row))
    print('done..')
