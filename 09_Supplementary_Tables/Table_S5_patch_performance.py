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
import numpy as np
import pandas as pd
from supplementary_table_metrics import compute_metrics

# %%
rows = []
for split, name in [('Training', 'BST_TRAIN_RESULTS.txt'), ('Test', 'BST_VAL_RESULTS.txt')]:
    path = DATA_ROOT / '03_Pathomics' / name
    if not path.exists():
        path = DATA_ROOT / '03_Pathomics/01_Patch_Evaluation' / name
    raw = pd.read_csv(path, sep='\t', header=None, usecols=[1, 2, 3],
                      names=['confidence', 'predicted', 'truth'])
    if not raw.predicted.isin([0, 1]).all() or not raw.truth.isin([0, 1]).all():
        raise ValueError(f'Non-binary patch labels: {path}')
    if not raw.confidence.between(.5, 1).all():
        raise ValueError('Expected confidence of the predicted class in [0.5, 1]')
    tumor_probability = np.where(raw.predicted.eq(1), raw.confidence, 1 - raw.confidence)

    rows.append({'Model': 'ResNet50', 'Cohort': split,
                 **compute_metrics(raw.truth.to_numpy(), tumor_probability, .5)})
frame = pd.DataFrame(rows)
frame['Precision'], frame['Recall'] = frame['PPV'], frame['Sensitivity']
table = frame

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = OUTPUT_DIR / "Table_S5_Patch_Performance.csv"
table.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(table.to_string(index=False))
print(f"Saved: {csv_path}")
