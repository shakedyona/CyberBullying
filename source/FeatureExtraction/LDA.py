from sklearn.decomposition import LatentDirichletAllocation
# import visualization
from source import utils
from sklearn.feature_extraction.text import CountVectorizer


def create_LDA_model(df, no_topics, name_image,folder_name):
    posts = df['text'].values
    tf_vectorizer = CountVectorizer(max_df=0.6, min_df=0.01, stop_words=utils.get_stop_words())
    tf = tf_vectorizer.fit_transform(posts)
    # tf_feature_names = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tf)
    # visualization.create_word_cloud(no_topics, lda, tf_feature_names,folder_name)
    lda = lda.transform(tf)
    return lda


