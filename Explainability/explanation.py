import xgboost
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt  # plotting


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
    print('finish shap')

    # xgb_model.fit(X, y, verbose=1)

    y_pred = model.predict(X)
    precision, recall, thresholds = precision_recall_curve(y, y_pred)
    area = auc(recall, precision)

    print('------------ Results for XGBClassifier ---------------')
    print('cm:', confusion_matrix(y, y_pred))
    print('precision: ', precision)
    print('recall: ', recall)
    # print('cr:', classification_report(y_test,y_pred))
    # print('recall_score:', recall_score(y_test,y_pred))
    print('roc_auc_score:', roc_auc_score(y, y_pred))
    print("Area Under P-R Curve: ", area)
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

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

