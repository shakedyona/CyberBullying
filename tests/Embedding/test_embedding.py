import os
from source.Embedding import word2vec as w2v
ROOT = os.path.abspath(os.path.join(__file__, '../../../'))


def test_correct_embedding():
    m_wiki = w2v.get_model(ROOT + "/source/Embedding/wiki.he.word2vec.model")
    m_our = w2v.get_model(ROOT + "/source/Embedding/our.corpus.word2vec.model")
    post = 'אני אוהבת אותו כל כך'
    post_vector = w2v.get_post_vector(m_our, m_wiki, post)
    print(post_vector.shape)
    assert post_vector.shape == (1, 100)


test_correct_embedding()