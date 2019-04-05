from sklearn.decomposition import NMF, LatentDirichletAllocation
# import visualization
import utils
import Preprocessing
from sklearn.feature_extraction.text import CountVectorizer


def create_LDA_model(df, no_topics, name_image):
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=utils.get_stop_words())
    posts = df['text'].tolist()
    matrix = vectorizer.fit_transform(posts)
    feature_names = vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=20, random_state=0).fit_transform(matrix)
    # visualization.create_word_cloud(no_topics, lda, feature_names, name_image)
    return lda

