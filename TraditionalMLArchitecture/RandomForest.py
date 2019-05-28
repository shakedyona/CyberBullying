from sklearn.model_selection import GridSearchCV
from TraditionalMLArchitecture.MLModel import MLModel
from sklearn.ensemble import RandomForestClassifier


class RandomForest(MLModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=250, max_depth=13, min_samples_split=5)


    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)


    def predict(self, x_test):
        y_pred = self.model.predict_proba(x_test)
        return y_pred[:, 1]


    def grid_search(self, x_train, y_train):
        classifier = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10)
        grid_param = {
            'max_depth': [5, 10, 80, 90],
            'min_samples_split': [5, 8, 10, 12],
            'n_estimators': [100, 200, 300, 500, 1000]
        }
        gd_sr = GridSearchCV(estimator=classifier, param_grid=grid_param, scoring='f1', cv=5, verbose=2, n_jobs=-1)
        gd_sr.fit(x_train, y_train)
        best_parameters = gd_sr.best_params_
        print(best_parameters)
        best_result = gd_sr.best_score_
        print(best_result)


