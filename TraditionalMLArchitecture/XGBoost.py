from TraditionalMLArchitecture.MLModel import MLModel
import xgboost as xgb


class XGBoost(MLModel):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.bst = None

    def train_predict(self, num_boost_round, params=None):
        if params is None:
            params = {'max_depth': 15, 'learning_rate': 0.01,
                      'objective': 'binary:logistic', 'scale_pos_weight': 1,
                      'n_estimators': 200, 'subsample': 0.3}
        d_train = xgb.DMatrix(self.x_train, label=self.y_train)
        d_test = xgb.DMatrix(self.x_test)
        self.bst = xgb.train(params, d_train, num_boost_round=num_boost_round)
        y_pred = self.bst.predict(d_test,output_margin=True)
        # self.classifier = xgb.XGBClassifier(objective='binary:logistic', max_depth=6, learning_rate=0.001,
        # n_estimators=550, subsample=0.7, scale_pos_weight=1)
        # self.bst = self.classifier.fit(train_X, train_y)
        # y_pred = self.classifier.predict_proba(test_X)[:, 1]
        return y_pred

    def cross_validation(self, params=None):
        if params is None:
            params = {'max_depth': 15, 'learning_rate': 0.01,
                      'objective': 'binary:logistic', 'scale_pos_weight': 1,
                      'n_estimators': 200, 'subsample': 0.3}
        d_train = xgb.DMatrix(self.x_train, label=self.y_train)
        xgb_cv = xgb.cv(params, d_train, nfold=10, num_boost_round=1000, metrics=['auc'], early_stopping_rounds=100)
        s = xgb_cv.shape[0]
        # print('AUC:', xgb_cv.values[s-1])
        return xgb_cv.shape[0]  # the best number of rounds

    def get_booster(self):
        return self.bst
