import shap


def explain_model(model, X):
    shap.initjs()
    # model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
    # explain the model's predictions using SHAP values
    # (same syntax works for LightGBM, CatBoost, and scikit-learn models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot("post_length", shap_values, X)
    # summarize the effects of all the features
    shap.summary_plot(shap_values, X)
    shap.summary_plot(shap_values, X, plot_type="bar")
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True)
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

