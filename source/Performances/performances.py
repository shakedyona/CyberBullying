from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.model_selection import cross_validate
from source import utils


def get_performances(true, pred):
    return {
        'f-score': f_measure(true, pred),
        'precision': precision(true, pred),
        'recall': recall(true, pred),
    }


def f_measure(true, pred):
    return f1_score(true, pred, average='macro')


def precision(true, pred):
    return precision_score(true, pred, average='macro')


def recall(true,pred):
    return recall_score(true, pred, average='macro')


def get_roc_auc(y, y_pred):
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def evaluate(X, y):
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
