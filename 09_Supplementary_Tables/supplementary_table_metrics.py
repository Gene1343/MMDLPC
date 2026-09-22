from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_curve


MODELS = ('LR', 'NaiveBayes', 'SVM', 'KNN', 'RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM', 'GradientBoosting', 'MLP')
TABLES = {'S4': ('Radiology', 'Rad'),
          'S8': ('Pathology', 'Path'),
          'S9': ('Pathology_Radiology', 'Path-Rad')}


def normalize_ids(values):
    if values.isna().any():
        raise ValueError('Missing patient IDs')
    return values.astype(str).str.strip().str.replace(r'\.nii(?:\.gz)?$', '', regex=True, case=False)


def auc_delong(y, score):

    positive, negative = score[y == 1], score[y == 0]
    m, n = len(positive), len(negative)
    if not m or not n:
        raise ValueError('AUC requires both outcome classes')
    ranks = rankdata(np.r_[positive, negative], method='average')
    auc = ranks[:m].sum() / (m * n) - (m + 1) / (2 * n)
    if min(m, n) < 2:
        return auc, np.nan, np.nan
    v01 = (ranks[:m] - rankdata(positive)) / n
    v10 = 1 - (ranks[m:] - rankdata(negative)) / m
    variance = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n
    half_width = norm.ppf(.975) * np.sqrt(max(0, variance))
    return auc, max(0., auc - half_width), min(1., auc + half_width)


def youden_cutoff(y, score):
    fpr, tpr, thresholds = roc_curve(y, score, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds <= score.max())
    objective = tpr[valid] - fpr[valid]
    return float(thresholds[valid][np.isclose(objective, objective.max(), atol=1e-12, rtol=0)].max())


def compute_metrics(y, score, cutoff):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.shape != score.shape or y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected aligned one-dimensional binary labels and scores')
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError('Scores must be finite probabilities between 0 and 1')
    predicted = score >= cutoff
    tp, tn = int(np.sum(predicted & (y == 1))), int(np.sum(~predicted & (y == 0)))
    fp, fn = int(np.sum(predicted & (y == 0))), int(np.sum(~predicted & (y == 1)))
    def ratio(a, b):
        return a / b if b else np.nan
    auc, lower, upper = auc_delong(y, score)
    return {'N': len(y), 'Positive': int(np.sum(y)), 'Cutoff': cutoff,
            'AUC': auc, 'AUC_CI_low': lower, 'AUC_CI_high': upper,
            'Accuracy': (tp + tn) / len(y), 'Sensitivity': ratio(tp, tp + fn),
            'Specificity': ratio(tn, tn + fp), 'PPV': ratio(tp, tp + fp),
            'NPV': ratio(tn, tn + fn), 'F1': ratio(2 * tp, 2 * tp + fp + fn),
            'Brier': np.mean((score - y) ** 2), 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def load_labels(root):
    path = root / '00_Shared_Data_and_Code/Data/CPGEA-TCGA 20230106 OK.csv'
    labels = pd.read_csv(path, usecols=['ID', 'HRR_ANY'])
    labels['ID'] = normalize_ids(labels.ID)
    if labels.ID.duplicated().any() or not labels.HRR_ANY.isin([0, 1]).all():
        raise ValueError('Labels need unique IDs and binary HRR_ANY')
    return labels


def load_score(root, directory, prefix, model, split, labels):
    path = root / '07_Model_Scores' / directory / f'{prefix}_{model}_{split}.csv'
    frame = pd.read_csv(path, usecols=['ID', 'HRR_ANY-1']).rename(columns={'HRR_ANY-1': 'score'})
    frame['ID'] = normalize_ids(frame.ID)
    if frame.ID.duplicated().any():
        raise ValueError(f'Duplicate IDs: {path}')
    frame = frame.merge(labels, on='ID', how='left', validate='one_to_one')
    if frame.HRR_ANY.isna().any():
        raise ValueError(f'Unmatched labels: {path}')
    if not frame.score.between(0, 1).all():
        raise ValueError(f'Invalid probabilities: {path}')
    return frame
