import json
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import matthews_corrcoef, make_scorer, confusion_matrix


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_store_name(experiment, results_folder):
    return os.path.join(results_folder, f'{experiment}.json')


def confusion_matrix_custom(preds, y, normalize=True):
    confusion_matrix = pd.crosstab(
        preds, y, margins=True, margins_name='total', normalize=normalize)
    confusion_matrix.columns = pd.Index(
        [0, 1, 'total'], dtype='object', name='real')
    confusion_matrix.index = pd.Index(
        [0, 1, 'total'], dtype='object', name='pred')
    return confusion_matrix.round(4)


def scaled_mcc(y_true, y_pred):
    matthews_corrcoef_scaled = (matthews_corrcoef(y_true, y_pred) + 1)/2
    return matthews_corrcoef_scaled


def gps_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Evitar división por cero usando una pequeña constante en el denominador
    eps = 1e-10

    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    ppv = tp / (tp + fp + eps)
    npv = tn / (tn + fn + eps)

    gps_num = 4 * ppv * recall * specificity * npv
    gps_denom = (ppv * recall * npv) + (ppv * recall * specificity) + (npv * specificity * ppv) + (npv * specificity * recall)

    gps = gps_num / (gps_denom + eps)
    return gps


def named_scorer(score_func, name, greater_is_better=True, **kwargs):
    scorer = make_scorer(score_func, greater_is_better=greater_is_better, **kwargs)
    scorer.__name__ = name
    return scorer

