# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
while not (PROJECT_ROOT / "modeling_pipeline.py").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Run this code from the MMDLPC_Code_PDF folder.")
    PROJECT_ROOT = PROJECT_ROOT.parent
CODE_DIR = PROJECT_ROOT / "01_Clinical_and_Cohort/02_Clinical_Model"
DATA_ROOT = PROJECT_ROOT.parent / "MMDLPC_Code_PDF_Data"
OUTPUT_DIR = CODE_DIR
sys.path.insert(0, str(CODE_DIR))
PANELS = ('D',)
COEFFICIENTS_FILE = CODE_DIR / 'FigureS3B_logistic_estimates.csv'
# %%
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(PROJECT_ROOT / '09_Supplementary_Tables'))
from supplementary_table_metrics import MODELS, auc_delong, load_labels, load_score


def plot_external_auc(root, output):
    labels = load_labels(root)
    estimates = []
    patient_ids = None
    for model in MODELS:
        data = load_score(root, 'Clinical', 'Clinical', model, 'test', labels)
        if patient_ids is not None and set(data.ID) != patient_ids:
            raise ValueError('Clinical models must use the same test patients')
        patient_ids = set(data.ID)
        estimates.append(auc_delong(data.HRR_ANY.to_numpy(), data.score.to_numpy()))
    values = np.asarray(estimates)
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(MODELS))
    ax.errorbar(x, values[:, 0], yerr=[values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]],
                fmt='o', color='#427F9E', capsize=4, ms=6)
    ax.axhline(.5, linestyle='--', color='#999999', lw=1)
    ax.set_xticks(x, MODELS, rotation=45, ha='right')
    ax.set(ylim=(0, 1.02), ylabel='External-test ROC AUC (95% DeLong CI)',
           title=f'D  Clinical models - external test set (n={len(patient_ids)})')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_logistic_estimates(path, output):
    if not path.is_file():
        raise FileNotFoundError(f'Provide verified logistic-regression estimates: {path}')
    frame = pd.read_csv(path)
    columns = ['Clinical_feature', 'OR', 'CI_low', 'CI_high']
    if not set(columns).issubset(frame):
        raise ValueError(f'Required columns: {columns}')
    if frame.Clinical_feature.isna().any() or frame.Clinical_feature.duplicated().any():
        raise ValueError('Clinical features must be uniquely labeled, including reference categories/units')
    values = frame[['OR', 'CI_low', 'CI_high']].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any() or not ((frame.CI_low <= frame.OR) & (frame.OR <= frame.CI_high)).all():
        raise ValueError('Expected positive finite odds ratios with ordered confidence intervals')
    fig, ax = plt.subplots(figsize=(9, max(4, .45 * len(frame) + 1.5)))
    y = np.arange(len(frame))
    ax.errorbar(frame.OR, y, xerr=[frame.OR - frame.CI_low, frame.CI_high - frame.OR], fmt='o', capsize=3, color='#8A648F')
    ax.axvline(1, color='#888888', linestyle='--', lw=1)
    ax.set_yticks(y, frame.Clinical_feature)
    ax.invert_yaxis()
    ax.set(xscale='log', xlabel='Odds ratio (95% CI)', title='B  Clinical variables and HRD status')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

# %%
if 'B' in PANELS and not COEFFICIENTS_FILE.is_file():
    raise FileNotFoundError(COEFFICIENTS_FILE)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10})

# %%
if 'B' in PANELS:
    plot_logistic_estimates(COEFFICIENTS_FILE, OUTPUT_DIR / 'Figure_S3B_Clinical_Logistic_Estimates.pdf')
if 'D' in PANELS:
    plot_external_auc(DATA_ROOT, OUTPUT_DIR / 'Figure_S3D_Clinical_External_AUC_Recalculated.pdf')
