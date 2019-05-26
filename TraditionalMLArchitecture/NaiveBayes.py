from TraditionalMLArchitecture.MLModel import MLModel
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(MLModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train(self):
        nb = GaussianNB()
        model = nb.fit(self.x_train, self.y_train)
        return model


    def predict(self, model):
        y_pred = model.predict_proba(self.x_test)
        return y_pred[:, 1]
