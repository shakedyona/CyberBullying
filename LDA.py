from sklearn.decomposition import NMF, LatentDirichletAllocation
import visualization
import preprocessing
from sklearn.feature_extraction.text import CountVectorizer


def create_LDA_model(df, no_topics, name_image):
    vectorizer = CountVectorizer(min_df=10, max_df=0.6, encoding="cp1255", stop_words=preprocessing.get_stop_words)
    matrix = vectorizer.fit_transform(df['text'])
    feature_names = vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online',
                                    learning_offset=50., random_state=0).fit(matrix)
    visualization.create_word_cloud(no_topics, lda, feature_names, name_image)
    return lda

