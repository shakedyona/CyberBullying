import utils
import Preprocessing.preprocessing as pre
import FeatureExtraction.featureExtraction as fe
import TraditionalMLArchitecture.XGBoost as xgb
from sklearn.model_selection import train_test_split
import Performances.performances as per

# get tagged df
tagged_df = utils.read_to_df()
# pre process
tagged_df = pre.preprocess(tagged_df)
# extract features
X = fe.extract_feature(tagged_df, ['post_length', 'tfidf'])
y = (tagged_df['cb_level'] == '3').astype(int)
X = X.drop(columns=['id'])
# split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = xgb.train(X_train, y_train)
y_pred = model.predict(X_test)
performances = per.get_performances(y_test, y_pred)
print(performances)
