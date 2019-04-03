import xgboost
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def explain_xgboost(X,y):
    # check error
    X = X.drop(columns=['id'])
    shap.initjs()
    print(X.shape)
    print(X)
    print(y)
    print(X.dtypes)
    model = XGBClassifier()
    model.fit(X,y)
    print(model)
    #model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
    # explain the model's predictions using SHAP values
    # (same syntax works for LightGBM, CatBoost, and scikit-learn models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot("post_length", shap_values, X)
    # summarize the effects of all the features
    shap.summary_plot(shap_values, X)
    shap.summary_plot(shap_values, X, plot_type="bar")
    print('finish')


# load JS visualization code to notebook
# shap.initjs()
#
# # train XGBoost model
# # X = DataFrame , y = ndarray
#
# X,y = shap.datasets.boston()
# print('x')
# print(X)
# print('y')
# print(y)
# model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
#
# # explain the model's predictions using SHAP values
# # (same syntax works for LightGBM, CatBoost, and scikit-learn models)
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)
#
# # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
# shap.dependence_plot("RM", shap_values, X)
#
# # summarize the effects of all the features
# shap.summary_plot(shap_values, X)
#
# shap.summary_plot(shap_values, X, plot_type="bar")

