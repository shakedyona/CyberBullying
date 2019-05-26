#from DeepLearningArchitecture.CNN import CNN
import utils
import Embedding.word2vec as w2v
import Preprocessing.preprocessing as pre
from sklearn.model_selection import train_test_split
import pandas as pd

# get tagged df
from DeepLearningArchitecture.lstm import lstm

path = '../dataNew.csv'
cols = ['id', 'time', 'source', 'sub_source', 'writer', 'link', 'text', 'cb_level']
df = pd.read_csv(path, names=cols, header=0)
df = utils.get_tagged_posts(df)

# pre process
df = pre.preprocess(df)

# DNN preparation
model = w2v.get_model('../Embedding/our.corpus.word2vec.model')
cnnObj = lstm(df)
# cnnObj = CNN(df)
embedding_matrix = cnnObj.create_embedding_matrix(model)

y = (df['cb_level'] == 3).astype(int)
X = cnnObj.create_data_input()

# split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cnnObj.create_model()  # 128, 5
history = cnnObj.train(X_train, y_train, X_test, y_test)
cnnObj.print_evaluation(history, X_train, y_train, X_test, y_test)
