from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, ttest_ind

MODEL_NAMES = ['SVM', 'KNN', 'RandomForest', 'ExtraTrees', 'XGBoost',
               'LightGBM', 'NaiveBayes', 'GradientBoosting', 'LR', 'MLP']

def read_indexed(path):
    frame = pd.read_csv(path, dtype={'ID': str})
    frame['ID'] = frame['ID'].str.replace('.nii.gz', '', regex=False).str.strip()
    if frame.ID.isna().any() or frame.ID.eq('').any() or frame.ID.duplicated().any():
        raise ValueError(f'Patient IDs must be present and unique: {path}')
    return frame.set_index('ID')

def load_dataset(data_dir, model_set):

    data_dir = Path(data_dir)
    key = 'PR' if model_set == 'CPR' else ('P' if model_set == 'CP' else
          ('R' if model_set == 'CR' else model_set))
    split = read_indexed(data_dir / f'{key}_fixed_partition.csv')
    if set(split.Split) != {'Train', 'Test'}:
        raise ValueError('Split must contain exactly Train and Test.')
    master = read_indexed(data_dir / 'CPGEA-TCGA 20230106 OK.csv')
    if not split.index.isin(master.index).all():
        raise ValueError('Partition contains patients absent from the label table.')
    y = master.loc[split.index, 'HRR_ANY']
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError('HRR_ANY must be observed and encoded HRD=1, HRP=0.')
    if 'P' in model_set:
        p_split = read_indexed(data_dir / 'P_fixed_partition.csv')
        if not split.index.isin(p_split.index).all():
            raise ValueError('A fixed-partition patient is absent from the confirmed pathology analysis list.')
    if model_set in {'C', 'P', 'CP'}:
        cohorts = master.loc[split.index, 'group'].astype(str)
        if not (cohorts[split.Split == 'Test'] == 'CPGEA').all() or (cohorts[split.Split == 'Train'] == 'CPGEA').any():
            raise ValueError('External evaluation requires TCGA training and CPGEA test membership.')
    parts = {}
    if 'C' in model_set:
        clinical = read_indexed(data_dir / 'clinical_raw.csv').drop(columns=['HRR_ANY', 'group'])
        parts['C'] = clinical.add_prefix('clinical__')
    if 'R' in model_set:
        raw = pd.read_csv(data_dir / 'rad_features.csv', dtype={'ID': str})
        series = []
        for ending, suffix in [('-1.nii.gz', 'DWI'), ('-2.nii.gz', 'T2WI')]:
            frame = raw.loc[raw.ID.str.endswith(ending)].copy()
            frame['ID'] = frame.ID.str.removesuffix(ending)
            if frame.ID.duplicated().any():
                raise ValueError(f'Duplicate radiology patient/sequence: {suffix}')
            series.append(frame.set_index('ID').add_suffix(suffix))
        parts['R'] = series[0].join(series[1], how='inner').add_prefix('radiology__')
    if 'P' in model_set:
        prob = read_indexed(data_dir / 'superwise_path_prob_histogram.csv')
        pred = read_indexed(data_dir / 'superwise_path_pred_histogram.csv')
        parts['P'] = prob.join(pred, how='inner').add_prefix('pathology__')
    for name, frame in parts.items():
        if not split.index.isin(frame.index).all():
            missing = split.index[~split.index.isin(frame.index)].tolist()
            raise ValueError(f'{name}: missing features for {missing}; no silent patient deletion is allowed.')
    X = pd.concat([parts[name].loc[split.index] for name in model_set], axis=1)
    X = X.apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    return X, y.astype(int), split.Split

def parse_p_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in {'', 'na', 'nan', 'n/a'}:
        return np.nan
    text = text.lstrip('<>≤≥=').strip()
    result = float(text)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError('A p value must be between zero and one.')
    return result


def correlation_keep_indices(values, method, threshold):
    correlation = pd.DataFrame(values).corr(method=method).abs().to_numpy()
    keep = list(range(values.shape[1]))
    while len(keep) > 1:
        current = correlation[np.ix_(keep, keep)].copy()
        np.fill_diagonal(current, 0)
        pairs = np.argwhere(np.triu(current > threshold, 1))
        if not len(pairs):
            break
        a, b = pairs[0]
        removed = a if current[a].mean() > current[b].mean() else b
        keep.pop(int(removed))
    return keep


class ModalityPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, modality, alpha=0.001, correlation_threshold=0.9, pvalue_threshold=None):
        self.modality = modality
        self.alpha = alpha
        self.correlation_threshold = correlation_threshold
        self.pvalue_threshold = pvalue_threshold

    def _expand(self, X, fitting=False):
        X = X.copy()
        if self.modality == 'P':
            if (X.fillna(0) < 0).any().any() or X.isna().any().any():
                raise ValueError('Pathology histograms must contain observed nonnegative frequencies.')
            if fitting:
                self.tfidf_ = {}
            for kind in ['prob', 'pred']:
                columns = [c for c in X.columns if c.startswith(f'pathology__{kind}-')]
                if fitting:

                    self.tfidf_[kind] = TfidfTransformer().fit(X[columns])
                values = self.tfidf_[kind].transform(X[columns]).toarray()
                names = [c.replace('pathology__', 'pathology__bow_').replace('-', '').replace('.', '') for c in columns]
                X = pd.concat([X, pd.DataFrame(values, index=X.index, columns=names)], axis=1)
        return X

    def fit(self, X, y):
        self.fit_patient_ids_ = tuple(X.index.astype(str))
        expanded = self._expand(X, fitting=True)
        self.observed_columns_ = expanded.columns[expanded.notna().any()].tolist()
        if not self.observed_columns_:
            raise ValueError('No observed features in this training fold.')
        self.imputer_ = SimpleImputer(strategy='median').fit(expanded[self.observed_columns_])
        filled = self.imputer_.transform(expanded[self.observed_columns_])
        self.scaler_ = StandardScaler().fit(filled)
        scaled = self.scaler_.transform(filled)
        keep = np.flatnonzero(np.var(scaled, axis=0) > 0).tolist()
        if not keep:
            raise ValueError('All features are constant in this training fold.')
        if self.pvalue_threshold is not None:
            if not 0 < self.pvalue_threshold <= 1:
                raise ValueError('The p-value threshold must be in (0, 1].')
            outcome = np.asarray(y)
            if min(np.sum(outcome == 0), np.sum(outcome == 1)) < 2:
                raise ValueError('Feature testing requires two observations per class.')
            self.p_values_ = np.full(scaled.shape[1], np.nan)
            self.p_values_[keep] = ttest_ind(
                scaled[outcome == 0][:, keep], scaled[outcome == 1][:, keep],
                axis=0, equal_var=False).pvalue
            keep = [i for i in keep if np.isfinite(self.p_values_[i]) and self.p_values_[i] < self.pvalue_threshold]
            if not keep:
                raise ValueError('No features passed the training-fold p-value threshold.')
        if self.modality != 'C':
            method = 'spearman' if self.modality == 'P' else 'pearson'
            selected = correlation_keep_indices(scaled[:, keep], method, self.correlation_threshold)
            keep = [keep[i] for i in selected]
        self.correlation_indices_ = keep
        self.lasso_ = None
        if self.modality != 'C':
            if self.alpha <= 0:
                raise ValueError('The LASSO alpha must be positive.')
            self.lasso_ = Lasso(alpha=self.alpha, max_iter=20000, tol=1e-5).fit(scaled[:, keep], y)
            chosen = np.flatnonzero(np.abs(self.lasso_.coef_) > 1e-6)
            if not len(chosen):
                raise ValueError('LASSO selected no features for this candidate alpha.')
            keep = [keep[i] for i in chosen]
        self.selected_indices_ = keep
        self.selected_features_ = np.asarray(self.observed_columns_)[keep]
        return self

    def transform(self, X):
        expanded = self._expand(X)
        values = self.imputer_.transform(expanded[self.observed_columns_])
        return self.scaler_.transform(values)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features_

def make_pipeline(X, model_set, classifier):
    prefix = {'C': 'clinical__', 'R': 'radiology__', 'P': 'pathology__'}
    transforms = [(m, ModalityPreprocessor(m), [c for c in X.columns if c.startswith(prefix[m])])
                  for m in model_set]
    return Pipeline([('features', ColumnTransformer(transforms, remainder='drop')),
                     ('classifier', clone(classifier))])

def training_cv(y, seed=0, requested=5):
    n = min(requested, int(pd.Series(y).value_counts().min()))
    if n < 2:
        raise ValueError('At least two training observations per class are required for stratified CV.')
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)

def fit_on_training(X_train, y_train, model_set, classifier, grid, seed=0, nested_cv=True):

    cv_auc = []
    if nested_cv:
        for fold, (train_idx, val_idx) in enumerate(training_cv(y_train, seed).split(X_train, y_train), 1):
            fitted, _ = fit_training_partition(X_train.iloc[train_idx], y_train.iloc[train_idx],
                                               model_set, classifier, grid, seed)
            p = positive_probability(fitted, X_train.iloc[val_idx])
            cv_auc.append({'Fold': fold, 'AUC': roc_auc_score(y_train.iloc[val_idx], p)})
    fitted, parameters = fit_training_partition(X_train, y_train, model_set, classifier, grid, seed)
    return fitted, pd.DataFrame(cv_auc), parameters

def best_finite_parameters(search):
    scores = np.asarray(search.cv_results_['mean_test_score'])
    if not np.isfinite(scores).any():
        raise ValueError('No valid training-CV parameter candidate. Review training data and search grid.')
    best = int(np.nanargmax(scores))
    return search.cv_results_['params'][best]

def fit_training_partition(X, y, model_set, classifier, grid, seed):

    parameters = {}
    for modality in model_set:
        if modality == 'C':
            continue
        modality_grid = {key: values for key, values in grid.items() if key.startswith(f'features__{modality}__')}
        search = GridSearchCV(make_pipeline(X, modality, classifier), modality_grid,
                              scoring='roc_auc', cv=training_cv(y, seed), n_jobs=1,
                              error_score=np.nan, refit=False)
        search.fit(X, y)
        parameters.update(best_finite_parameters(search))
    fitted = make_pipeline(X, model_set, classifier).set_params(**parameters).fit(X, y)
    return fitted, parameters

def positive_probability(model, X):
    classes = list(model.classes_)
    if set(classes) != {0, 1}:
        raise ValueError('Binary HRD classes must be 0 and 1.')
    return model.predict_proba(X)[:, classes.index(1)]

def model_metrics(y, probability):

    fpr, tpr, thresholds = roc_curve(y, probability, pos_label=1)
    finite = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    threshold = thresholds[np.flatnonzero(finite)[np.argmax((tpr - fpr)[finite])]]
    tn, fp, fn, tp = confusion_matrix(y, probability >= threshold, labels=[0, 1]).ravel()
    positive, negative = probability[np.asarray(y) == 1], probability[np.asarray(y) == 0]
    comparisons = (positive[:, None] > negative[None, :]).astype(float)
    comparisons += 0.5 * (positive[:, None] == negative[None, :])
    auc = roc_auc_score(y, probability)
    variance = (comparisons.mean(axis=1).var(ddof=1) / len(positive) +
                comparisons.mean(axis=0).var(ddof=1) / len(negative)) if min(len(positive), len(negative)) > 1 else np.nan
    delta = norm.ppf(0.975) * np.sqrt(variance)
    return {'N': len(y), 'AUC': auc, 'AUC_CI_Lower': max(0, auc - delta) if np.isfinite(delta) else np.nan,
            'AUC_CI_Upper': min(1, auc + delta) if np.isfinite(delta) else np.nan, 'Threshold': threshold,
            'Sensitivity': tp / (tp + fn), 'Specificity': tn / (tn + fp),
            'PPV': tp / (tp + fp) if tp + fp else np.nan,
            'NPV': tn / (tn + fn) if tn + fn else np.nan,
            'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
            'Accuracy': (tp + tn) / len(y), 'Brier': brier_score_loss(y, probability)}

def model_estimators(model_set):

    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import sklearn
    sklearn_version = tuple(int(part) for part in sklearn.__version__.split('.')[:2])
    gradient_loss = 'deviance' if sklearn_version < (1, 1) else 'log_loss'
    models = {
        'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=0),
        'KNN': KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree'),
        'RandomForest': RandomForestClassifier(n_estimators=10, max_features='sqrt', random_state=0),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=10, max_features='sqrt', random_state=0),
        'XGBoost': XGBClassifier(n_estimators=10, objective='binary:logistic', eval_metric='error', random_state=0),
        'LightGBM': LGBMClassifier(n_estimators=10, num_leaves=31, learning_rate=0.1, random_state=0),
        'NaiveBayes': GaussianNB(var_smoothing=1e-9),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=10, loss=gradient_loss, random_state=0),
        'LR': LogisticRegression(C=1.0, max_iter=100, random_state=0),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, solver='sgd', random_state=0),
    }
    if model_set in {'PR', 'CPR'}:
        models['RandomForest'] = RandomForestClassifier(n_estimators=4, max_depth=3, random_state=0)
        models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=3, max_depth=2, random_state=0)
        models['XGBoost'] = XGBClassifier(n_estimators=1, objective='binary:logistic', max_depth=4,
                                        use_label_encoder=False, eval_metric='error')
        models['LightGBM'] = LGBMClassifier(n_estimators=1, max_depth=2)
    if model_set == 'CR':
        models['LR'] = LogisticRegression(penalty='l2', max_iter=21, C=0.8)
        models['SVM'] = SVC(probability=True, kernel='rbf', max_iter=46, C=0.2, random_state=0)
        models['LightGBM'] = LGBMClassifier(n_estimators=3, max_depth=2, random_state=0)
        models['RandomForest'] = RandomForestClassifier(n_estimators=2, max_depth=3, random_state=0)
        models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=3, max_depth=2, random_state=0)
        models['XGBoost'] = XGBClassifier(n_estimators=1, objective='binary:logistic', max_depth=2,
                                        min_child_weight=2, use_label_encoder=False, eval_metric='error')
    if model_set == 'R':
        models['SVM'] = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=0)
    if model_set == 'P':
        models['LightGBM'] = LGBMClassifier(n_estimators=10, num_leaves=31, learning_rate=0.1, random_state=0)
    if model_set == 'PR':
        models['NaiveBayes'] = GaussianNB(var_smoothing=1e-9)
    if model_set == 'CP':
        models['RandomForest'] = RandomForestClassifier(n_estimators=10, criterion='gini', bootstrap=True, random_state=0)
    for model in models.values():
        if 'random_state' in model.get_params() and model.get_params()['random_state'] is None:
            model.set_params(random_state=0)
    return models

def training_search_grids(model_set, models):

    alphas = np.logspace(-6, 0, 40).tolist()
    return {name: {f'features__{modality}__alpha': alphas
                   for modality in model_set if modality != 'C'} for name in models}
