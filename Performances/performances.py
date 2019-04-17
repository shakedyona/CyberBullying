from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import random


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
