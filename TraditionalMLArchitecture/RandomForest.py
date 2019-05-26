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


    def train(self):
        rf = RandomForestClassifier(n_estimators=250, max_depth=13, min_samples_split=5)
        rf.fit(self.x_train, self.y_train)
        return rf


    def predict(self, rf):
        y_pred = rf.predict_proba(self.x_test)
        return y_pred[:, 1]


    def grid_search(self):
        classifier = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10)
        grid_param = {
            'max_depth': [5, 10, 80, 90],
            'min_samples_split': [5, 8, 10, 12],
            'n_estimators': [100, 200, 300, 500, 1000]
        }
        gd_sr = GridSearchCV(estimator=classifier, param_grid=grid_param, scoring='f1', cv=5, verbose=2, n_jobs=-1)
        gd_sr.fit(self.x_train, self.y_train)
        best_parameters = gd_sr.best_params_
        print(best_parameters)
        best_result = gd_sr.best_score_
        print(best_result)


