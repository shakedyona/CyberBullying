import matplotlib.pyplot as plt
import utils
import os
import pandas as pd
import xgboost as xgb
import wordcloud


def plot_tf_idf_post(dictionary_tf_idf, title, unique=False):
    dic_post = dict(dictionary_tf_idf[title])
    dic_post_travers = {}
    for term,val in dic_post.items():
        dic_post_travers[utils.traverse(term)] = val
    df2 = pd.DataFrame.from_dict(dic_post_travers,orient='index').sort_values(by=0, ascending=False)
    pl = df2.plot(kind='bar', figsize=(15,7), fontsize=8, legend=False, title=utils.traverse(title))
    for p in pl.patches:
        pl.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001), fontsize=14)
    plt.show()


def plot_length_posts(dictionary_length, title, unique=False):
    df2 = pd.DataFrame.from_dict(dictionary_length, orient='index').sort_values(by=0, ascending=False)
    pl = df2.plot(kind='bar', figsize=(15, 7), fontsize=8, legend=False, title=utils.traverse(title))
    for p in pl.patches:
        pl.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001), fontsize=14)
    plt.show()


def create_word_cloud(no_topics, lda, feature_names,name_image):
    global stop_words
    font_path = os.path.join(os.path.join(os.environ['WINDIR'], 'Fonts'), 'ahronbd.ttf')
    for i in range(0, no_topics):
        d = dict(zip(utils.traverse(feature_names), lda.components_[i]))
        wc = wordcloud.WordCloud(background_color='white', font_path=font_path, max_words=100,stopwords=stop_words)
        image = wc.generate_from_frequencies(d)
        image.to_file(name_image+str(i)+'.png')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.figure()
        # plt.show()


def print_tf_idf_dict(tf_idf_dict):
    for key, value in tf_idf_dict.items():
        print('post: ')
        print(key)
        for v in value:
            print('word: ' + str(v[0]) + ', tf-idf: ' + str(v[1]))


def plot_part_of_day(dictionary_time, title, unique=False):
    df2 = pd.DataFrame.from_dict(dictionary_time, orient='index').sort_values(by=0, ascending=False)
    pl = df2.plot(kind='bar', figsize=(15, 7), fontsize=8, legend=False, title=utils.traverse(title))
    for p in pl.patches:
        pl.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001), fontsize=14)
    plt.show()


def plot_roc_curve(roc_auc, fpr, tpr):
    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importance_xgb(booster):
    xgb.plot_importance(booster, importance_type='gain')
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()

