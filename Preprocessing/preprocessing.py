import collections
from nltk.tokenize import word_tokenize
import utils
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def create_tf_idf(dataframe, num_of_words):
    """
    calculates tf-idf for each post
    :param dataframe:
    :param num_of_words:
    :return:
    """
    # get the text column
    posts = dataframe['text'].tolist()
    dict = {}
    # create a vocabulary of words,
    cv = CountVectorizer(max_df=0.85, stop_words=get_stop_words())
    word_count_vector = cv.fit_transform(posts)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    # you only needs to do this once, this is a mapping of index to
    feature_names = cv.get_feature_names()

    # get the document that we want to extract keywords from
    for post in posts:
        # generate tf-idf for the given document
        tf_idf_vector = tfidf_transformer.transform(cv.transform([post]))
        # sort the tf-idf vectors by descending order of scores
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        # extract only the top n; n here is 10
        keywords = extract_topn_from_vector(feature_names, sorted_items, num_of_words)
        dict[post] = [(k, keywords[k]) for k in keywords]
    return dict


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    # use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def remove_stop_words(data_frame, my_stop_words):
    """
    tokenize all the words in the data frame and removes stop words
    :param data_frame:
    :param my_stop_words:
    :return: dictionary of words and their number of performances
    """
    # print("remove_stop_words")
    text = data_frame.text.tolist()
    all_words = " ".join(text)
    all_words = word_tokenize(all_words)
    for word in my_stop_words:
        if word in all_words:
            all_words.remove(word)

    top_words = dict(Counter(all_words))
    return top_words

# todo:

def clean_tokens(df):
    """
    clean all non-Hebrew characters from a given data frame.
    :param df:
    :return:
    """
    text = df.text.tolist()
    number_posts = len(text)
    for index_post in range(1, number_posts):
        text[index_post] = re.sub(r'[^א-ת]', ' ', text[index_post]).strip().rstrip()
    return text


def get_stop_words():
    """
    gets list of stop words from a file
    :return:
    """
    stop_words = utils.file_to_list(r'stop_words.txt')
    return stop_words


######################### statistics ################################
def get_common_words(dataframe, number):
    """
    returns a dictionary of the most frequent words as keys and
    their frequency as values,
    in the given dataframe limited by the given number
    :param dataframe:
    :param number:
    :return:
    """
    stop_words = get_stop_words()
    tokens = tokenize_df(dataframe)
    word_frequency = {}

    for word in tokens:
        if word not in stop_words:
            if word not in word_frequency:
                word_frequency[word] = 1
            else:
                word_frequency[word] += 1

    word_counter = collections.Counter(word_frequency)
    most_common_dictionary = word_counter.most_common(number)

    return most_common_dictionary


def tokenize_df(df):
    """
    tokenize all the words in the dataframe
    :param df:
    :return list of tokens:
    """
    text = df.text.tolist()
    tokens = []
    for post in text:
        tokens = tokens + word_tokenize(post)

    return tokens


def tokenize_post(post):
    """
    tokenize only one post
    :param post:
    :return list of tokens:
    """
    return word_tokenize(post)


def find_df(dataframe, threshold):
    stop_words = get_stop_words()
    text = dataframe.text.tolist()
    term_df = {}
    number_posts = len(text)
    print(number_posts)

    for index_post in range(1, number_posts):
        tokens = word_tokenize(text[index_post])
        for token in tokens:
            if token in term_df:
                list_posts = term_df[token]
                if index_post not in list_posts:
                    term_df[token].append(index_post)  # todo: change from indexes to counter
            else:
                term_df[token] = []
                term_df[token].append(index_post)

    for token, posts in term_df.items():
        df = len(posts)
        df_normal = float(df / number_posts)
        if df_normal > threshold:
            stop_words.append(token)


def get_post_length(dataframe):
    """
    gets length of all posts
    :param dataframe:
    :return:
    """
    post_frequency = {}
    for index, row in dataframe.iterrows():
        post_frequency[row.id] = len(word_tokenize(row.text))

    return post_frequency


def num_of_abusive_per_column(df, column_name):
    """
    number of abusive posts for a given column
    :param df:
    :param column_name:
    :return:
    """
    # columns: column_name, total_count, total_from_corpus, number_of_abusive, normalized_abusive
    total_count_df = df.groupby(column_name)['cb_level'].apply(lambda x: x.count())
    total_from_corpus_df = df.groupby(column_name)['cb_level'].apply(lambda x: (x.count() / df.shape[0]))
    number_of_abusive_df = df.groupby(column_name)['cb_level'].apply(lambda x: x[x == '3'].count())
    normalized_abusive_df = df.groupby(column_name)['cb_level'].apply(lambda x: (x[x == '3'].count() / x.count()) * 100)


    result = pd.DataFrame({'total_count': total_count_df})\
        .merge(pd.DataFrame({'total_from_corpus': total_from_corpus_df}), on=[column_name], right_index=True)

    result = result\
        .merge(pd.DataFrame({'number_of_abusive': number_of_abusive_df}), on=[column_name], right_index=True)
    result = result\
        .merge(pd.DataFrame({'normalized_abusive': normalized_abusive_df}), on=[column_name], right_index=True)

    return result


def avg_per_class(df):
    df_abusive = utils.get_abusive_df(df)
    df_no_abusive = utils.get_no_abusive_df(df)
    length_abusive_dictionary = get_post_length(df_abusive)
    avg_length_abusive = float(sum(length_abusive_dictionary.values())/float(len(length_abusive_dictionary)))
    length_no_abusive_dictionary = get_post_length(df_no_abusive)
    avg_length_no_abusive = float(sum(length_no_abusive_dictionary.values())/float(len(length_no_abusive_dictionary)))
    dictionary_length = {utils.traverse('abusive'): avg_length_abusive,
                         utils.traverse('no abusive'): avg_length_no_abusive}
    return dictionary_length
