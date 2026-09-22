from __future__ import annotations

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
while not (PROJECT_ROOT / "modeling_pipeline.py").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Run this code from the MMDLPC_Code_PDF folder.")
    PROJECT_ROOT = PROJECT_ROOT.parent
CODE_DIR = PROJECT_ROOT / "09_Supplementary_Tables"
DATA_ROOT = PROJECT_ROOT.parent / "MMDLPC_Code_PDF_Data"
OUTPUT_DIR = CODE_DIR
sys.path.insert(0, str(CODE_DIR))

# %%
import importlib.util
import itertools

import pandas as pd


OUTPUT = OUTPUT_DIR
PREDICTIONS = DATA_ROOT / "04_Multimodal/fixed85_predictions_final.csv"
CSV_OUTPUT = OUTPUT / "Table_S11_fixed85_DeLong_R1_recalculated.csv"
DELONG_CODE = PROJECT_ROOT / "04_Multimodal/paired_delong.py"
if not DELONG_CODE.exists():
    DELONG_CODE = PROJECT_ROOT / "04_Multimodal/03_Cohort_Comparison/paired_delong.py"


MODELS = ("Pathology", "Radiology", "Patho-Radiology")
ANALYSIS_SETS = (
    ("A. All cohort (descriptive; n={n})", None),
    ("B. Training set (apparent; n={n})", "Training"),
    ("C. Held-out test set (source split: Validation; n={n})", "Validation"),
)


def load_delong_module():
    spec = importlib.util.spec_from_file_location("paired_delong_support", DELONG_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DeLong implementation: {DELONG_CODE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# %%
delong = load_delong_module()
predictions = pd.read_csv(PREDICTIONS, dtype={'ID': str})
required = {'ID', 'HRR_ANY', 'split', 'model', 'score'}
if not required.issubset(predictions):
    raise ValueError(f'Missing prediction columns: {sorted(required - set(predictions))}')
if predictions[list(required)].isna().any().any() or predictions.duplicated(['ID', 'model']).any():
    raise ValueError('Predictions contain missing values or duplicate patient-model rows')
if set(predictions.model) != set(MODELS) or set(predictions.split) != {'Training', 'Validation'}:
    raise ValueError('Expected the fixed three models and Training/Validation splits')
if not predictions.HRR_ANY.isin([0, 1]).all() or not predictions.score.between(0, 1).all():
    raise ValueError('Expected binary HRR_ANY labels and finite probabilities')
if predictions.groupby('ID').model.nunique().ne(3).any():
    raise ValueError('The three models must contain the same patient IDs')
if predictions.groupby('ID').HRR_ANY.nunique().ne(1).any() or predictions.groupby('ID').split.nunique().ne(1).any():
    raise ValueError('Patient labels and split assignments must agree across models')
rows: list[dict[str, object]] = []

# %%
for analysis_label, split in ANALYSIS_SETS:
    subset = (
        predictions
        if split is None
        else predictions.loc[predictions["split"].eq(split)]
    )
    wide = (
        subset.pivot(index=["ID", "HRR_ANY"], columns="model", values="score")
        .reset_index()
        .sort_values("ID")
    )
    y_true = wide["HRR_ANY"].to_numpy(dtype=int)
    for first, second in itertools.combinations(MODELS, 2):
        auc_1, auc_2, difference, z_value, p_value = delong.paired_delong(
            y_true,
            wide[first].to_numpy(dtype=float),
            wide[second].to_numpy(dtype=float),
        )
        rows.append(
            {
                "Analysis set": analysis_label.format(n=len(wide)),
                "Model 1": first,
                "Model 2": second,
                "AUC 1": auc_1,
                "AUC 2": auc_2,
                "Delta AUC (1-2)": difference,
                "DeLong z": z_value,
                "Unadjusted P": p_value,
            }
        )

table = pd.DataFrame(rows)
# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
print(table.to_string(index=False))
print(f"Saved: {CSV_OUTPUT}")
