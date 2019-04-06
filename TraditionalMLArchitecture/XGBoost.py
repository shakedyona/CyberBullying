from TraditionalMLArchitecture.MLModel import MLModel
import xgboost as xgb

def train(X_train, y_train):
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    return xgb_model


class XGBoost (MLModel):
    def train(self, train_X, train_y, test_X, params=None, num_boost_round=32):
        if params is None:
            params = {
                'max_depth': 10,
                'learning_rate': 0.001,
                'objective': 'binary:logistic',
                'scale_pos_weight': 1,
                'n_estimators': 1000,
                'subsample': 0.3
            }
        dtrain = xgb.DMatrix(train_X, label=train_y)
        dtest = xgb.DMatrix(test_X)
        self.bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        test_y = self.bst.predict(dtest)
        print('feature_importance: ', self.bst.get_score(importance_type='weight'))
        # self.classifier = xgb.XGBClassifier(objective='binary:logistic', max_depth=6, learning_rate=0.001, n_estimators=550, subsample=0.7,
        #                                    scale_pos_weight=1)
        # self.bst = self.classifier.fit(train_X, train_y)
        # test_y = self.classifier.predict_proba(test_X)[:, 1]
        return test_y
