import utils
import Preprocessing.preprocessing as pre
import FeatureExtraction.featureExtraction as fe
import TraditionalMLArchitecture.XGBoost as xgb
from sklearn.model_selection import train_test_split
import Performances.performances as per
import visualization as vis
import numpy as np
import Baseline as bl

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

# 1.baseline
y_pred_bl = bl.run_baseline(tagged_df)
performances_bl = per.get_performances(y, y_pred_bl)

# 2.XGBoost
xgbObj = xgb.XGBoost(X_train, y_train, X_test, y_test)
num_boost_round = xgbObj.cross_validation()
y_pred = xgbObj.train(num_boost_round=num_boost_round)
y_pred_bin = np.where(y_pred > 0.5, 1, 0)
performances_xgb = per.get_performances(y_test, y_pred_bin)

# 3.


# visualization
roc_auc_xgb, fpr_xgb, tpr_xgb = per.get_roc_auc(y_test, y_pred)
roc_auc_bl, fpr_bl, tpr_bl = per.get_roc_auc(y, y_pred_bl)

# vis.plot_roc_curve(roc_auc_bl, fpr_bl, tpr_bl)
# vis.plot_roc_curve(roc_auc_xgb, fpr_xgb, tpr_xgb)

vis.plot_models_compare(performances_xgb, performances_bl)