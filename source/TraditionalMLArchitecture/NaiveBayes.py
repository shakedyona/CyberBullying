from source.TraditionalMLArchitecture.MLModel import MLModel
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(MLModel):
    def __init__(self):
        super().__init__()
        self.model = GaussianNB()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)


    def predict(self, x_test):
        y_pred = self.model.predict_proba(x_test)
        return y_pred[:, 1]
