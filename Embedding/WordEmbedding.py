from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = LineSentence(datapath('HebrewWordEmbedding/wiki.he.text'))

model = Word2Vec(min_count=5, size=300, workers=10, sg=0, negative=2)
model.build_vocab("אני הולך הביתה")
model.train("אני הולך הביתה", total_examples=model.corpus_count, epochs=model.epochs)
# model.save("word2vec_test.model")

