from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB

COLORS = {0: '#348ABD', 1: '#E7A838'}
FAMILIES = ('Radiomics', 'PLH', 'BoW')
FAMILY_LABELS = {'Radiomics': 'Radiomics', 'PLH': 'Pathology: histogram', 'BoW': 'Pathology: TF-IDF'}


def clean_ids(frame):
    frame = frame.copy()
    if frame.ID.isna().any():
        raise ValueError('Missing patient IDs')
    frame['ID'] = frame.ID.astype(str).str.strip().str.replace(r'\.nii(?:\.gz)?$', '', regex=True)
    if frame.ID.duplicated().any():
        raise ValueError('Duplicate patient IDs')
    return frame


def estimate_parameters(root):
    topic = root / '04_Multimodal'
    predictions = pd.read_csv(topic / 'fixed85_predictions_final.csv')
    cohort = clean_ids(predictions.loc[predictions.model.eq('Patho-Radiology'), ['ID', 'split', 'HRR_ANY']])
    if not cohort.HRR_ANY.isin([0, 1]).all() or set(cohort.split) != {'Training', 'Validation'}:
        raise ValueError('Training/Validation splits and binary HRR_ANY are required')
    train = cohort.loc[cohort.split.eq('Training')].copy()
    radiomics = clean_ids(pd.read_csv(topic / 'fixed85_radiology_features.csv'))
    pathology = clean_ids(pd.read_csv(topic / 'fixed85_pathology_features.csv'))
    if set(radiomics.ID) != set(cohort.ID) or set(pathology.ID) != set(cohort.ID):
        raise ValueError('Features and locked cohort IDs differ')

    plh = [c for c in pathology if c.startswith(('prob-', 'pred-'))]
    bow = [c for c in pathology if c != 'ID' and c not in plh]
    if not plh or not bow or not set(radiomics.columns).difference({'ID'}):
        raise ValueError('Histogram, TF-IDF and radiomics features are required')
    groups = {c: 'Radiomics' for c in radiomics.columns if c != 'ID'}
    groups.update({c: 'PLH' for c in plh})
    groups.update({c: 'BoW' for c in bow})
    data = train.merge(radiomics, on='ID', validate='one_to_one').merge(pathology, on='ID', validate='one_to_one')
    features = list(groups)
    x = data[features].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Missing/nonfinite feature values; supply the verified prepared matrix')

    model = GaussianNB(var_smoothing=1e-9).fit(x, data.HRR_ANY)
    rows = []
    for i, cls in enumerate(model.classes_):
        for j, feature in enumerate(features):
            rows.append({'Feature': feature, 'Family': groups[feature], 'Class': int(cls),
                         'Mean': model.theta_[i, j], 'Variance': model.var_[i, j],
                         'N': int(model.class_count_[i]), 'Variance_smoothing': float(model.epsilon_)})
    return pd.DataFrame(rows)


def validate_parameters(frame):
    required = {'Feature', 'Family', 'Class', 'Mean', 'Variance', 'N'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    if frame.duplicated(['Feature', 'Class']).any() or frame.Feature.isna().any():
        raise ValueError('Duplicate or missing feature/class keys')
    if set(frame.Family) != set(FAMILIES) or not frame.Class.isin([0, 1]).all():
        raise ValueError('Family must be Radiomics/PLH/BoW; class must be HRP=0 or HRD=1')
    if not np.isfinite(frame[['Mean', 'Variance', 'N']].to_numpy(float)).all() or not frame.Variance.gt(0).all():
        raise ValueError('Finite means, positive variances and sample counts are required')
    if not (frame.N.ge(2) & frame.N.eq(np.floor(frame.N))).all():
        raise ValueError('Each class needs an integer N >= 2')
    for feature, rows in frame.groupby('Feature', sort=False):
        if set(rows.Class) != {0, 1} or rows.Family.nunique() != 1:
            raise ValueError(f'Incomplete or inconsistent class parameters for {feature}')
    return frame


def feature_label(name, width=25):
    return '\n'.join(textwrap.wrap(name.replace('_', ' '), width=width, break_long_words=True))


def plot_s7(parameters, output):
    with PdfPages(output) as pdf:
        for panel, family in zip('ABC', FAMILIES):
            data = parameters.loc[parameters.Family.eq(family)]
            features = data.Feature.drop_duplicates().tolist()
            cols, rows = 3, int(np.ceil(len(features) / 3))
            fig, axes = plt.subplots(rows, cols, figsize=(12, 2.45 * rows + .8), squeeze=False)
            for ax, feature in zip(axes.flat, features):
                pars = data.loc[data.Feature.eq(feature)].set_index('Class')
                lo = min(pars.Mean - 4 * np.sqrt(pars.Variance))
                hi = max(pars.Mean + 4 * np.sqrt(pars.Variance))
                grid = np.linspace(lo, hi, 401)
                for cls in [0, 1]:
                    mean, variance = pars.loc[cls, ['Mean', 'Variance']]
                    density = norm.pdf(grid, loc=mean, scale=np.sqrt(variance))
                    ax.plot(grid, density, color=COLORS[cls], lw=1.2)
                    ax.fill_between(grid, density, color=COLORS[cls], alpha=.22)
                ax.set_title(feature_label(feature, 36), fontsize=8)
                ax.set_xlabel('Feature value', fontsize=7)
                ax.set_ylabel('Density', fontsize=7)
                ax.tick_params(labelsize=7)
                ax.spines[['top', 'right']].set_visible(False)
            for ax in list(axes.flat)[len(features):]:
                ax.axis('off')
            fig.suptitle(f'{panel}  {FAMILY_LABELS[family]} - class-conditional Gaussian densities', fontsize=13, y=.994)
            fig.legend(handles=[Patch(color=COLORS[c], label='HRP' if c == 0 else 'HRD') for c in [0, 1]],
                       loc='upper center', bbox_to_anchor=(.5, .973), ncol=2, frameon=False)
            fig.tight_layout(rect=(0, 0, 1, .95))
            pdf.savefig(fig)
            plt.close(fig)


def fisher_scores(parameters):
    rows = []
    for feature, data in parameters.groupby('Feature', sort=False):
        pars = data.set_index('Class')
        weighted_mean = np.average(pars.Mean, weights=pars.N)
        between = np.sum(pars.N * (pars.Mean - weighted_mean) ** 2)
        within = np.sum(pars.N * pars.Variance)
        rows.append({'Feature': feature, 'Family': pars.Family.iloc[0], 'Fisher_score': between / within})
    return pd.DataFrame(rows).sort_values('Fisher_score', ascending=False, kind='stable')


def plot_s8(parameters, scores, output):
    radiomics = parameters.loc[parameters.Family.eq('Radiomics'), 'Feature'].drop_duplicates()
    radiomics_codes = {feature: f'R{i + 1}' for i, feature in enumerate(radiomics)}
    fig = plt.figure(figsize=(15, 14))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.25], hspace=.95, wspace=.25)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    for panel, family, ax in zip('ABC', ['PLH', 'BoW', 'Radiomics'], axes):
        data = parameters.loc[parameters.Family.eq(family)]
        features = data.Feature.drop_duplicates().tolist()
        for cls, offset in [(0, -.11), (1, .11)]:
            pars = data.loc[data.Class.eq(cls)].set_index('Feature').loc[features]
            ax.errorbar(np.arange(len(features)) + offset, pars.Mean, yerr=np.sqrt(pars.Variance),
                        fmt='o', ms=3.5, capsize=2, lw=.9, color=COLORS[cls],
                        label='HRP' if cls == 0 else 'HRD')
        labels = [feature_label(f'{radiomics_codes[f]}: {f}' if f in radiomics_codes else f, 32) for f in features]
        ax.set_xticks(np.arange(len(features)), labels,
                      rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Mean +/- SD')
        ax.set_title(f'{panel}  {FAMILY_LABELS[family]}', loc='left', weight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    ax = fig.add_subplot(grid[2, :])
    palette = {'Radiomics': '#348ABD', 'PLH': '#EDAD69', 'BoW': '#EDAD69'}
    ax.bar(np.arange(len(scores)), scores.Fisher_score, color=scores.Family.map(palette))
    ax.set_xticks(np.arange(len(scores)), [radiomics_codes.get(f, f) for f in scores.Feature],
                  rotation=65, ha='right', fontsize=6.5)
    ax.set_ylabel('Fisher score')
    ax.set_title('D  Between-class / within-class variance', loc='left', weight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(handles=[Patch(color=palette['Radiomics'], label='Radiomics'),
                       Patch(color=palette['PLH'], label='Pathology')], frameon=False)
    fig.subplots_adjust(left=.065, right=.985, bottom=.22, top=.96)
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
