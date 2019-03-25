import xgboost
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

data_path = 'data.csv'
cols = ['id', 'time', 'source', 'sub_source', 'writer', 'link', 'text', 'cb_level', 'comment_shared_post']
df = pd.read_csv(data_path, delimiter=',', names=cols)
df = df[df.cb_level != '2']

# We can see how many records and features we have in our data set
# print(df.shape)
# We will use "cb_level" columns as our target variable
#y = df['cb_level']._ndarray_values
y = df['cb_level']
print(y)
X = df.drop(['cb_level'], axis=1)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# model = XGBClassifier()
# model.fit(X_train, y_train)

#model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
print('finish')




# # load JS visualization code to notebook
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

