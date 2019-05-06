from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from DeepLearningArchitecture.DLModel import DLModel


class CNN(DLModel):
    def __init__(self, embedding_matrix=None, model=None):
        super().__init__(embedding_matrix)
        self.model = model

    def create_model(self, num_filters, kernel_size):
        model = Sequential()
        model.add(self.word2vec_embedding_layer())
        model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        return model

    def train(self, x_train, y_train, x_test, y_test, epochs=50, verbose=False, batch_size=10):
        history = self.model.fit(x_train, y_train,
                                 epochs=epochs,
                                 verbose=verbose,
                                 validation_data=(x_test, y_test),
                                 batch_size=batch_size)
        return history

    def predict(self, X):
        pass

    def run_grid_search(self, X_train, y_train, X_test, y_test, vocab_size, embedding_dim, maxlen, epochs):
        # Parameter grid for grid search
        param_grid = dict(num_filters=[32, 64, 128],
                          kernel_size=[3, 5, 7],
                          vocab_size=[vocab_size],
                          embedding_dim=[embedding_dim],
                          maxlen=[maxlen])
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

    def print_evaluation(self, history, X_train, y_train, X_test, y_test, verbose=False):
        loss, accuracy = self.model.evaluate(X_train, y_train, verbose=verbose)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=verbose)
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
