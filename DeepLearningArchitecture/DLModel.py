import numpy as np
from keras.layers import Embedding


class DLModel:
    def __init__(self, embedding_matrix=None):
        self.embedding_matrix = embedding_matrix

    def create_input_from_embedding(self, model):
        if self.embedding_matrix is not None:
            return
        embedding_matrix = np.zeros((len(model.wv.vocab), 100))
        for i in range(len(model.wv.vocab)):
            embedding_vector = model.wv[model.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        self.embedding_matrix = embedding_matrix
        return embedding_matrix

    def word2vec_embedding_layer(self):
        layer = Embedding(input_dim=self.embedding_matrix.shape[0],
                          output_dim=self.embedding_matrix.shape[1],
                          weights=[self.embedding_matrix])
        return layer
