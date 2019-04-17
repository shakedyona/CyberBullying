import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time


def train(inp="wiki2.he.text", out_model="wiki.he.word2vec.model"):
    start = time.time()

    model = Word2Vec(LineSentence(inp), sg = 1, # 0=CBOW , 1= SkipGram
                     size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)

    print(time.time()-start)

    model.save(out_model)


def get_model(model="wiki.he.word2vec.model"):

    model = Word2Vec.load(model)

    return model


m = get_model('our.corpus.word2vec.model')
print(m.wv.most_similar('שלום'))
