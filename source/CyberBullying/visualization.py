import matplotlib.pyplot as plt
import os
import pandas as pd
import xgboost as xgb
import wordcloud
import numpy as np
from . import Logger, utils
import pathlib
WHERE_OUTPUTS = pathlib.Path(__file__).parent


def plot_dictionary(dictionary_length, title):
    """
    plot dictionary with hebrew keys support
    :param dictionary_length:
    :param title:
    :return:
    """
    df2 = pd.DataFrame.from_dict(dictionary_length, orient='index').sort_values(by=0, ascending=False)
    pl = df2.plot(kind='bar', figsize=(15, 7), fontsize=8, legend=False, title=utils.traverse(title))
    for p in pl.patches:
        pl.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001), fontsize=14)
    plt.show()


def create_word_cloud(no_topics, lda, feature_names):
    """
    create picture with words in different size according to frequency
    :param no_topics:
    :param lda:
    :param feature_names:
    :return:
    """
    logger = Logger.get_logger_instance()
    font_path = os.path.join(os.path.join(os.environ['WINDIR'], 'Fonts'), 'ahronbd.ttf')
    for i in range(0, no_topics):
        d = dict(zip(utils.traverse(feature_names), lda.components_[i]))
        wc = wordcloud.WordCloud(background_color='white', font_path=font_path, max_words=50, stopwords=utils.get_stop_words())
        image = wc.generate_from_frequencies(d)
        image.to_file(logger.folder_name + r'\Topic' + str(i+1) + '.png')
        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()


def create_lda_visualization(no_topics, lda_model):
    """
    create a word cloud for each topic from an lda model.
    :param no_topics:
    :param lda_model:
    :return:
    """
    tf_vectorizer = utils.get_model(os.path.join(WHERE_OUTPUTS / 'outputs', 'tf.pkl'))
    tf_feature_names = tf_vectorizer.get_feature_names()

    create_word_cloud(no_topics, lda_model, tf_feature_names)


def plot_roc_curve(roc_auc, fpr, tpr, name):
    """
    plot ROC curve graph with a given auc, false positive rate, true positive rate and a graph title (name)
    :param roc_auc:
    :param fpr:
    :param tpr:
    :param name:
    :return:
    """
    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic -'+str(name))
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importance_xgb(booster):
    """
    plot XGBoost feature importance of a given XGBoost booster
    :param booster:
    :return:
    """
    xgb.plot_importance(booster, importance_type='gain')
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()


def plot_models_compare(per1, per2, per3, per4):
    """
    plot a bar graph for comparison of the 4 models: Baseline, XGBoost, Random forest and Naive bayes
    compare values of precision recall and f-measure
    :param per1:
    :param per2:
    :param per3:
    :param per4:
    :return:
    """
    fig, ax = plt.subplots()
    index = np.arange(3)
    ax.bar(index, [per1[key] for key in sorted(per1.keys())], color=(0.5, 0.4, 0.8, 0.4), width=0.2, label='Baseline')
    ax.bar(index+0.25, [per2[key] for key in sorted(per2.keys())], color=(0.8, 0.5, 0.4, 0.6), width=0.2, label='XGBoost')
    ax.bar(index+0.5, [per3[key] for key in sorted(per3.keys())], color=(0.2, 0.8, 0.4, 0.6), width=0.2, label='Random forest')
    ax.bar(index+0.75, [per4[key] for key in sorted(per4.keys())], color=(0.5, 0.8, 0.3, 0.5), width=0.2, label='Naive bayes')
    ax.set_xlabel('Performances')
    ax.set_ylabel('')
    ax.set_title('Model compare')
    ax.set_xticks(index+0.3)
    ax.set_xticklabels(['F-Measure', 'Precision', 'Recall'])
    ax.legend()
    fig.tight_layout()
    plt.show()

