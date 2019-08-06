from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from .. import utils


def get_performances(true, pred):
    """
    return a dictionary with 'f-score', 'precision', and 'recall' and theirs values
    calculated from lists of true values and predicted values
    :param true:
    :param pred:
    :return:
    """
    return {
        'f-score': f_measure(true, pred),
        'precision': precision(true, pred),
        'recall': recall(true, pred),
    }


def f_measure(true, pred):
    """
    return the result of f-score calculated on lists of true values and predicted values
    :param true:
    :param pred:
    :return:
    """
    return f1_score(true, pred, average='binary')


def precision(true, pred):
    """
    return the result of precision calculated on lists of true values and predicted values
    :param true:
    :param pred:
    :return:
    """
    return precision_score(true, pred, average='binary')


def recall(true, pred):
    """
    return the result of recall calculated on lists of true values and predicted values
    :param true:
    :param pred:
    :return:
    """
    return recall_score(true, pred, average='binary')


def get_roc_auc(y, y_pred):
    """
    calculate roc curve and auc on lists of true values and predicted values
    return auc result, list of false positive rate and list of true positive rate
    :param y:
    :param y_pred:
    :return:
    """
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def precision_recall(y, y_pred):
    """
    plot precision-recall graph
    :param y:
    :param y_pred:
    :return:
    """
    p, r, _ = precision_recall_curve(y, y_pred)
    plt.step(r, p, color='b', alpha=0.2, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.legend(loc="lower right")


def evaluate(X, y):
    """
    print accuracy, precision, recall and auc of a given samples and corresponded targets,
    evaluated with cross validation with cv=3
    :param X:
    :param y:
    :return:
    """
    models = utils.create_list_of_models()
    results = []
    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=3,
                                    scoring=('accuracy', 'precision', 'recall', 'roc_auc'),
                                    return_train_score=True)
        results.append((name, cv_results))
        msg = "%s: train_accuracy: %f. test_accuracy: %f)" % (name, np.mean(cv_results['train_accuracy']),
                                                              np.mean(cv_results['test_accuracy']))
        print(msg)
        msg = "%s: train_precision: %f. test_precision: %f)" % (name, np.mean(cv_results['train_precision']),
                                                                np.mean(cv_results['test_precision']))
        print(msg)
        msg = "%s: train_recall: %f. test_recall: %f)" % (name, np.mean(cv_results['train_recall']),
                                                          np.mean(cv_results['test_recall']))
        print(msg)
        msg = "%s: train_roc_auc: %f. test_roc_auc: %f)" % (name, np.mean(cv_results['train_roc_auc']),
                                                            np.mean(cv_results['test_roc_auc']))
        print(msg)
    return results
