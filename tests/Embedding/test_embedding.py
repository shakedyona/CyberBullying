import source.utils as utils
import pandas as pd
from source.Embedding import word2vec as w2v


def test_correct_embedding():
    m_wiki = w2v.get_model(r"Embedding/wiki.he.word2vec.model")
    m_our = w2v.get_model(r"Embedding/our.corpus.word2vec.model")
    post = 'אני אוהבת אותו כל כך'
    post_vector = w2v.get_post_vector(m_our, m_wiki, post)
    assert post_vector.shape == (1, 100)


test_correct_embedding()