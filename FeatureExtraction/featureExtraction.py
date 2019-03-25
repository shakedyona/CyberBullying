import pandas as pd
import FeatureExtraction.statistics as statistics
import FeatureExtraction.LDA as LDA


def extract_feature(df, columns):
    pass


df = pd.read_csv('../data.csv')
tfidf = statistics.create_tf_idf(df, 20)

[print(key + ': ' + str(value)) for key, value in tfidf.items()]
