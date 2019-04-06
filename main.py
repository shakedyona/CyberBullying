import utils
import Preprocessing.preprocessing as pre
import FeatureExtraction.featureExtraction as fe
import TraditionalMLArchitecture.XGBoost as xgb
from sklearn.model_selection import train_test_split
import Performances.performances as per
import visualization as vis
import numpy as np
# get tagged df
tagged_df = utils.read_to_df()
# pre process
tagged_df = pre.preprocess(tagged_df)
# extract features
X = fe.extract_features(tagged_df, ['post_length', 'tfidf', 'topics'])
y = (tagged_df['cb_level'] == '3').astype(int)
X = X.drop(columns=['id'])
# split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
xgbObj = xgb.XGBoost()
y_pred = xgbObj.train(train_X=X_train, train_y=y_train, test_X=X_test)
y_pred_bin = np.where(y_pred > 0.5, 1, 0)
performances = per.get_performances(y_test, y_pred_bin)
print(performances)

roc_auc, fpr, tpr = per.get_roc_auc(y_test, y_pred)
vis.plot_roc_curve(roc_auc, fpr, tpr)

