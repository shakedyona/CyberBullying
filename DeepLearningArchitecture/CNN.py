from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras import layers
from DeepLearningArchitecture.DLModel import DLModel


class CNN(DLModel):
    def __init__(self, dataframe, embedding_matrix=None, model=None):
        super().__init__(dataframe, embedding_matrix, model)

    def create_model(self, num_filters, kernel_size):
        model = Sequential()
        model.add(self.word2vec_embedding_layer())
        model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dropout(0.50))
        model.add(layers.Dense(20, activation='relu'))
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

    def run_grid_search(self, X_train, y_train, X_test, y_test, epochs):
        # Parameter grid for grid search
        param_grid = dict(num_filters=[32, 64, 128],
                          kernel_size=[3, 5, 7])
        model = KerasClassifier(build_fn=self.create_model,
                                epochs=epochs, batch_size=10,
                                verbose=False)
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                  cv=4, verbose=1, n_iter=5)
        grid_result = grid.fit(X_train, y_train)

        test_accuracy = grid.score(X_test, y_test)
        return test_accuracy, grid_result

    def print_grid_evaluation(self, test_accuracy, grid_result, source):
        s = ('Running {} data set\nBest Accuracy : '
             '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        output_string = s.format(
            source,
            grid_result.best_score_,
            grid_result.best_params_,
            test_accuracy)
        return output_string
