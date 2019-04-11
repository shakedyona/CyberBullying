from sklearn.decomposition import NMF, LatentDirichletAllocation
# import visualization
import utils
import Preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import visualization


def create_LDA_model(df, no_topics, name_image):
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=utils.get_stop_words(), max_df=0.85)
    posts = df['text'].values
    matrix = vectorizer.fit_transform(posts)
    feature_names = vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=20, random_state=0).fit_transform(matrix)
    # lda_visualization = LatentDirichletAllocation(n_components=no_topics, max_iter=20, random_state=0).fit(matrix)
    return lda


