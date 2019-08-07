from .MLModel import MLModel
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(MLModel):
    """
    This class handles the creation compilation and functionality of Naive Bayes model
    """
    def __init__(self):
        """
        Initiates a Gaussian Naive Bayes model
        """
        super().__init__()
        self.model = GaussianNB()

    def train(self, x_train, y_train):
        """
        trains Naive Bayes model with fit operation
        :param x_train:
        :param y_train:
        :return:
        """
        self.model.fit(x_train, y_train)


    def predict(self, x_test):
        """
        predicts probabilities for a given test set
        :param x_test:
        :return:
        """
        y_pred = self.model.predict_proba(x_test)
        return y_pred[:, 1]
