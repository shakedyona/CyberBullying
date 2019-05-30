import numpy as np
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


class DLModel:
    def __init__(self, dataframe, embedding_matrix=None, model=None):
        self.dataframe = dataframe
        self.embedding_matrix = embedding_matrix
        self.model = model

    def create_embedding_matrix(self, model):
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

    def create_data_input(self):
        posts = self.dataframe['text'].tolist()
        t = Tokenizer()
        t.fit_on_texts(posts)

        encoded_docs = t.texts_to_sequences(posts)

        max_length = 200
        return pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    def print_evaluation(self, history, X_train, y_train, X_test, y_test, verbose=False):
        (loss, accuracy, mae,  mse) = self.model.evaluate(X_train, y_train, verbose=verbose)
        print("Training Accuracy: {:.4f}".format(accuracy))
        (loss, accuracy, mae, mse) = self.model.evaluate(X_test, y_test, verbose=verbose)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        plot_history(history)


def plot_history(history):
    plt.style.use('ggplot')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
