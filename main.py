import datetime
import Logger
import utils
import Preprocessing.preprocessing as pre
import FeatureExtraction.featureExtraction as fe
import TraditionalMLArchitecture.XGBoost as xgb
import TraditionalMLArchitecture.RandomForest as rf
import TraditionalMLArchitecture.NaiveBayes as nb
from sklearn.model_selection import train_test_split
import Performances.performances as per
import visualization as vis
import numpy as np
import Baseline as bl
import os
from Explainability.explanation import explain_model

# logger
datetime_object = datetime.datetime.now()
print(datetime_object)
folder_name = r"Looger\Logger_"+str(datetime_object).replace(':','-')
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# get tagged df
tagged_df = utils.read_to_df()  # Vigo data
# tagged_df = utils.create_csv_from_keepers_files()  # Keepers data
# pre process
tagged_df = pre.preprocess(tagged_df)
# extract features
X = fe.extract_features(tagged_df, ['post_length', 'tfidf', 'topics', 'screamer', 'words'],folder_name)
y = (tagged_df['cb_level'] == '3').astype(int)
X = X.drop(columns=['id'])
# split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

performances_list = {}
auc_list = {}
# 1.baseline
y_pred_bl = bl.run_baseline(tagged_df)
performances_bl = per.get_performances(y, y_pred_bl)
performances_list['baseline'] = performances_bl

# 2.XGBoost
xgbObj = xgb.XGBoost(X_train, y_train, X_test, y_test)
num_boost_round = xgbObj.cross_validation()
y_pred = xgbObj.train_predict(num_boost_round=num_boost_round)
y_pred_bin = np.where(y_pred > 0.5, 1, 0)
performances_xgb = per.get_performances(y_test, y_pred_bin)
performances_list['XGBoost'] = performances_xgb

# 3.Random forest todo: add cross validation
rf_obj = rf.RandomForest(X_train, y_train, X_test, y_test)
y_pred_rf = rf_obj.train_predict()
y_pred_bin1 = np.where(y_pred_rf > 0.5, 1, 0)
performances_rf = per.get_performances(y_test, y_pred_bin1)
performances_list['Random forest']= performances_rf

# 4.Naive Bayes todo: add cross validation
nb_obj = nb.NaiveBayes(X_train, y_train, X_test, y_test)
y_pred_nb = nb_obj.train_predict()
y_pred_bin2 = np.where(y_pred_nb > 0.5, 1, 0)
performances_nb = per.get_performances(y_test, y_pred_bin2)
performances_list['Naive Bayes'] = performances_nb

# visualization
roc_auc_bl, fpr_bl, tpr_bl = per.get_roc_auc(y, y_pred_bl)
auc_list['baseline'] = roc_auc_bl
roc_auc_xgb, fpr_xgb, tpr_xgb = per.get_roc_auc(y_test, y_pred)
auc_list['XGBoost'] = roc_auc_xgb
roc_auc_rf, fpr_rf, tpr_rf = per.get_roc_auc(y_test, y_pred_rf)
auc_list['Random forest']= roc_auc_rf
roc_auc_nb, fpr_nb, tpr_nb = per.get_roc_auc(y_test, y_pred_nb)
auc_list['Naive Bayes'] = roc_auc_nb

vis.plot_roc_curve(roc_auc_bl, fpr_bl, tpr_bl,'baseline')
vis.plot_roc_curve(roc_auc_xgb, fpr_xgb, tpr_xgb, 'xgboost')
vis.plot_roc_curve(roc_auc_rf, fpr_rf, tpr_rf, 'random forest')
vis.plot_roc_curve(roc_auc_nb, fpr_nb, tpr_nb, 'naive bayes')

vis.plot_models_compare(performances_bl, performances_xgb, performances_rf, performances_nb)

# SHAP for XGBoost:
explain_model(xgbObj.get_booster(), X_test) # todo

Logger.write_performances(folder_name,auc_list,performances_list,datetime_object)


