from TraditionalMLArchitecture.MLModel import MLModel
from xgboost import XGBClassifier


def train(X_train, y_train):
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    return xgb_model


class XGBoost (MLModel):
    pass
