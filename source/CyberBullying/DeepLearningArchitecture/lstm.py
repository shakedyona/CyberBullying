from keras.models import Sequential
from keras import layers
from source.CyberBullying.DeepLearningArchitecture.DLModel import DLModel


class lstm(DLModel):
    def __init__(self, dataframe, embedding_matrix=None, model=None):
        super().__init__(dataframe, embedding_matrix, model)

    def create_model(self):
        model = Sequential()
        model.add(self.word2vec_embedding_layer())
        model.add(layers.LSTM(self.embedding_matrix.shape[1], dropout=0.2, recurrent_dropout=0.2))
        model.add(layers.Dropout(0.50))
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy', 'mae', 'mean_squared_error'])
        self.model = model
        return model

    def train(self, x_train, y_train, x_test, y_test, epochs=30, batch_size=10, verbose=True):
        history = self.model.fit(x_train, y_train,
                                 epochs=epochs,
                                 verbose=verbose,
                                 validation_data=(x_test, y_test),
                                 batch_size=batch_size)
        return history

    def predict(self, X):
        pass




