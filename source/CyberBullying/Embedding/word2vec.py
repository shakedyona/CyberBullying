import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time
from nltk.tokenize import word_tokenize
import numpy as np


def train(inp="wiki2.he.text", out_model="wiki.he.word2vec.model"):
    """
    get path to input file of Hebrew text and an output file path
    train word2vec model with the text and save the word2vec model in the output path
    :param inp:
    :param out_model:
    :return:
    """
    start = time.time()
    model = Word2Vec(LineSentence(inp), sg = 1, # 0=CBOW , 1= SkipGram
                     size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.init_sims(replace=True)
    print(time.time()-start)
    model.save(out_model)


def get_model(model="wiki.he.word2vec.model"):
    """
    get a path to a saved word2vec model, load it and return the model
    :param model:
    :return:
    """
    model = Word2Vec.load(model)
    return model


def get_word_vector(model, word):
    """
    returns a 100 damnation vector represent a given word from a given word2vec model
    :param model:
    :param word:
    :return:
    """
    return model.wv[word]


def get_post_vector(our_word2vec_model, wiki_word2vec_model, post):
    """
    returns a 100 damnation vector that represent a given post.
    the post vector is an average vector of all the word2vec vectors which represent the words in the post
    the function gets 2 word2vec models: the first one is the main model
    which we want to get the word representative vector
    if the word is not in the vocabulary of the first model then the function tries the second model
    if the word is not in the vocabulary of neither of the models - the function returns 100 damnation vector of zeros
    :param our_word2vec_model:
    :param wiki_word2vec_model:
    :param post:
    :return:
    """
    post = word_tokenize(post)
    # remove out-of-vocabulary words
    postEmbedding = []
    for word in post:
        if word in our_word2vec_model.wv.vocab:
            postEmbedding.append(our_word2vec_model.wv[word])
        elif word in wiki_word2vec_model.wv.vocab:
            postEmbedding.append(wiki_word2vec_model.wv[word])

    if len(postEmbedding) == 0:
        return np.zeros((1, 100))
    return np.mean(postEmbedding, axis=0)
