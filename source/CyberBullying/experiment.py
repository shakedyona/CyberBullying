from source.CyberBullying import Logger, Baseline as bl, visualization as vis, utils
from source.CyberBullying.Preprocessing import preprocessing as pre
from source.CyberBullying.Performances import performances as per
from source.CyberBullying.FeatureExtraction import featureExtraction as fe
import source.CyberBullying.TraditionalMLArchitecture.XGBoost as xgb
import source.CyberBullying.TraditionalMLArchitecture.RandomForest as rf
import source.CyberBullying.TraditionalMLArchitecture.NaiveBayes as nb
from sklearn.model_selection import train_test_split
import numpy as np
from source.CyberBullying.Explainability import explanation as exp
from sklearn.metrics import accuracy_score

"""
Experiment - includes a complete experiment that runs the following models: baseline, XGBoost, Random forest, 
and Naive Bayes.
The experiment saves their results, to choose the best model with the best results.
"""

"""
Creating the logger
"""
logger = Logger.get_logger_instance()

"""
get tagged df
"""
tagged_df = utils.read_to_df()  # Vigo data
# tagged_df = utils.create_csv_from_keepers_files()  # Keepers data

"""
Run a pre-processing function on the tagged data
"""
tagged_df = pre.preprocess(tagged_df)

"""
Run a extract features function on the clean and tagged data
"""
feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
fe.folder_name = logger.folder_name
X = fe.extract_features(tagged_df, feature_list)
logger.write_features(feature_list)
y = (tagged_df['cb_level'] == 3).astype(int)
X = X.drop(columns=['id'])

"""
Split data to train and test
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

performances_list = {}
auc_list = {}

"""
Running the baseline model
"""
y_pred_bl = bl.run_baseline(tagged_df)
performances_bl = per.get_performances(y, y_pred_bl)
performances_list['baseline'] = performances_bl

"""
Running the XGBoost model
"""
xgb_obj = xgb.XGBoost()
xgb_classifier = xgb_obj.train(X_train, y_train)
y_prob_xgb = xgb_obj.predict(X_test)
y_pred_xgb = np.where(y_prob_xgb > 0.5, 1, 0)
performances_xgb = per.get_performances(y_test, y_pred_xgb)
performances_list['XGBoost'] = performances_xgb

"""
Running the Random forest model
"""
rf_obj = rf.RandomForest()
rf_classifier = rf_obj.train(X_train, y_train)
y_prob_rf = rf_obj.predict(X_test)
y_pred_rf = np.where(y_prob_rf > 0.5, 1, 0)
performances_rf = per.get_performances(y_test, y_pred_rf)
performances_list['Random forest'] = performances_rf

"""
Running the Naive Bayes model
"""
nb_obj = nb.NaiveBayes()
nb_classifier = nb_obj.train(X_train, y_train)
y_prob_nb = nb_obj.predict(X_test)
y_pred_nb = np.where(y_prob_nb > 0.5, 1, 0)
performances_nb = per.get_performances(y_test, y_pred_nb)
performances_list['Naive Bayes'] = performances_nb

"""
Part of visualization. Running functions that give the results of experiments.
"""
roc_auc_bl, fpr_bl, tpr_bl = per.get_roc_auc(y, y_pred_bl)
auc_list['baseline'] = roc_auc_bl
roc_auc_xgb, fpr_xgb, tpr_xgb = per.get_roc_auc(y_test, y_prob_xgb)
auc_list['XGBoost'] = roc_auc_xgb
roc_auc_rf, fpr_rf, tpr_rf = per.get_roc_auc(y_test, y_prob_rf)
auc_list['Random forest']= roc_auc_rf
roc_auc_nb, fpr_nb, tpr_nb = per.get_roc_auc(y_test, y_prob_nb)
auc_list['Naive Bayes'] = roc_auc_nb
vis.plot_roc_curve(roc_auc_bl, fpr_bl, tpr_bl,'baseline')
vis.plot_roc_curve(roc_auc_xgb, fpr_xgb, tpr_xgb, 'xgboost')
vis.plot_roc_curve(roc_auc_rf, fpr_rf, tpr_rf, 'random forest')
vis.plot_roc_curve(roc_auc_nb, fpr_nb, tpr_nb, 'naive bayes')
vis.plot_models_compare(performances_bl, performances_xgb, performances_rf, performances_nb)

"""
Running the SHAP for random forest model
"""
exp.explain_model(rf_obj.model, X_train)
logger.write_performances(auc_list, performances_list)

"""
Print the experimental results on the screen
"""
acc_bl = accuracy_score(y, y_pred_bl)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_nb = accuracy_score(y_test, y_pred_nb)
print('accuracy for baseline: ', acc_bl)
print('accuracy for xgboost: ', acc_xgb)
print('accuracy for random forest: ', acc_rf)
print('accuracy for naive bayes: ', acc_nb)