from TraditionalMLArchitecture.MLModel import MLModel
from sklearn.ensemble import RandomForestClassifier


class RandomForest(MLModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.bst = None

    def train(self, params=None):
        if params is None:
            params = {'max_depth': 10, 'min_samples_split': 10,
                      'n_estimators': 200}
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10)
        rf.fit(self.x_train, self.y_train)
        y_pred = rf.predict_proba(self.x_test)
        return y_pred[:, 1]

    def cross_validation(self, params=None):
        # if params is None:
        #     params = {'max_depth': 10, 'min_samples_split': 10,
        #               'n_estimators': 200}
        # d_train = xgb.DMatrix(self.x_train, label=self.y_train)
        # xgb_cv = xgb.cv(params, d_train, nfold=8, num_boost_round=300, metrics=['auc'], early_stopping_rounds=50)
        # s = xgb_cv.shape[0]
        # return xgb_cv.shape[0]  # the best number of rounds
        return True

