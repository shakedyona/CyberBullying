from sklearn.model_selection import GridSearchCV

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

    def train_predict(self, params=None):
        if params is None:
            params = {'max_depth': 10, 'min_samples_split': 10,
                      'n_estimators': 200}
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10)
        rf.fit(self.x_train, self.y_train)
        y_pred = rf.predict_proba(self.x_test)
        return y_pred[:, 1]

    def grid_search(self):
        print("grid_search - random forest")
        classifier = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10)
        grid_param = {
            'max_depth': [5, 10, 80, 90],
            # 'max_features': [2, 3, 4, 5, 6],
            # 'min_samples_leaf': [3, 4, 5, 6, 8],
            'min_samples_split': [5, 8, 10, 12],
            'n_estimators': [100, 200, 300, 500, 1000]
            # 'criterion': ['gini', 'entropy']
        }

        gd_sr = GridSearchCV(estimator=classifier,
                             param_grid=grid_param,
                             scoring='f1',
                             cv=5,
                             verbose=2,
                             n_jobs=-1)
        gd_sr.fit(self.x_train, self.y_train)
        best_parameters = gd_sr.best_params_
        print(best_parameters)
        best_result = gd_sr.best_score_
        print(best_result)

    def cross_validation(self, params=None):
        # if params is None:
        #     params = {'max_depth': 10, 'min_samples_split': 10,
        #               'n_estimators': 200}
        # d_train = xgb.DMatrix(self.x_train, label=self.y_train)
        # xgb_cv = xgb.cv(params, d_train, nfold=8, num_boost_round=300, metrics=['auc'], early_stopping_rounds=50)
        # s = xgb_cv.shape[0]
        # return xgb_cv.shape[0]  # the best number of rounds
        return True

