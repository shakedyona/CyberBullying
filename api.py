import utils
import Preprocessing.preprocessing as pre
import FeatureExtraction.featureExtraction as fe
import TraditionalMLArchitecture.XGBoost as xgb
from sklearn.model_selection import train_test_split
from Explainability.explanation import explain_class

import xgboost

def train_file(file):
    # get tagged df
    tagged_df = utils.read_to_df()  # Vigo data
    # tagged_df = utils.create_csv_from_keepers_files()  # Keepers data
    # pre process
    print("pre-processing..")
    tagged_df = pre.preprocess(tagged_df)
    # extract features
    print("extract features..")

    feature_list = ['post_length',
                    'tfidf',
                    'topics',
                    'screamer',
                    'words',
                    'off_dis',
                    'not_off_dis']
    X = fe.extract_features(tagged_df, feature_list, None)
    y = (tagged_df['cb_level'] == 3).astype(int)
    X = X.drop(columns=['id'])

    # split data to train and test
    print("split train and test..")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 2.XGBoost
    print("run XGBoost..")

    xgbObj = xgb.XGBoost(X_train, y_train, X_test, y_test)
    num_boost_round = xgbObj.cross_validation()
    y_pred = xgbObj.train_predict(num_boost_round=num_boost_round)
    utils.save_model(xgbObj.get_booster(), )


def predict(post, explainability=True):
    model = utils.get_model()
    d = xgboost.DMatrix([post])
    result = {'class': model.predict(d)}
    if explainability:
        result['explain'] = explain_class(post)
    return result


def get_performances():
    model = utils.get_model()

