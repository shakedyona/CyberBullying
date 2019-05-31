from source.DeepLearningArchitecture.CNN import CNN

from source import Preprocessing as pre, Embedding as w2v, utils
from sklearn.model_selection import train_test_split

# path = '../dataNew.csv'
# cols = ['id', 'time', 'source', 'sub_source', 'writer', 'link', 'text', 'cb_level']
# df = pd.read_csv(path, names=cols, header=0)
# df = utils.get_tagged_posts(df)
df = utils.create_csv_from_keepers_files('../keepersData')

# pre process
df = pre.preprocess(df)

# # DNN preparation
# model = w2v.get_model('../Embedding/wiki.he.word2vec.model')
# lstmObj = lstm(df)
# # cnnObj = CNN(df)
# embedding_matrix = lstmObj.create_embedding_matrix(model)
#
# y = (df['cb_level'] == 3).astype(int)
# X = lstmObj.create_data_input()
#
# # split data to train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# lstmObj.create_model()  # 128, 5
# history = lstmObj.train(X_train, y_train, X_test, y_test)
# lstmObj.print_evaluation(history, X_train, y_train, X_test, y_test)


# DNN preparation
model = w2v.get_model('../Embedding/wiki.he.word2vec.model')
cnnObj = CNN(df)
# cnnObj = CNN(df)
cnnObj.create_embedding_matrix(model)

y = (df['cb_level'] == 3).astype(int)
X = cnnObj.create_data_input()

# split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cnnObj.create_model(128, 5)
history = cnnObj.train(X_train, y_train, X_test, y_test)
cnnObj.print_evaluation(history, X_train, y_train, X_test, y_test)