from sklearn.decomposition import NMF, LatentDirichletAllocation
# import visualization
import utils
import Preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import visualization


def create_LDA_model(df, no_topics, name_image,folder_name):
    # vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=utils.get_stop_words(), max_df=0.85)
    # posts = df['text'].values
    # matrix = vectorizer.fit_transform(posts)
    # feature_names = vectorizer.get_feature_names()
    # # lda = LatentDirichletAllocation(n_components=no_topics, max_iter=20, random_state=0).fit_transform(matrix)
    # lda = LatentDirichletAllocation(n_components=no_topics, max_iter=20, random_state=0).fit(matrix)
    # visualization.create_word_cloud(no_topics, lda, feature_names, name_image)
    # lda = lda.transform(matrix)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.6, min_df=0.01, stop_words=utils.get_stop_words())
    posts = df['text'].values
    tfidf = tfidf_vectorizer.fit_transform(posts)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    tf_vectorizer = CountVectorizer(max_df=0.6, min_df=0.01, stop_words=utils.get_stop_words())
    tf = tf_vectorizer.fit_transform(posts)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tf)

    visualization.create_word_cloud(no_topics, lda, tf_feature_names, name_image,folder_name)
    lda = lda.transform(tf)
    return lda


