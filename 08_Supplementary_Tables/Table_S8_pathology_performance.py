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
import pandas as pd
from supplementary_table_metrics import MODELS, TABLES, load_labels, load_score, youden_cutoff, compute_metrics

# %%
TABLE_ID = "S8"
directory, prefix = TABLES[TABLE_ID]
labels = load_labels(DATA_ROOT)
# %%
rows, reference_ids = [], {}
for model in MODELS:
    train = load_score(DATA_ROOT, directory, prefix, model, 'train', labels)
    test = load_score(DATA_ROOT, directory, prefix, model, 'test', labels)
    if set(train.ID) & set(test.ID):
        raise ValueError(f'{TABLE_ID}: training/test patients overlap')
    for split, data in [('Training', train), ('Test', test)]:
        ids = set(data.ID)
        if split in reference_ids and ids != reference_ids[split]:
            raise ValueError(f'{TABLE_ID}: models do not use the same {split} patients')
        reference_ids[split] = ids
        y, score = data.HRR_ANY.to_numpy(), data.score.to_numpy()
        cutoff = youden_cutoff(y, score)
        rows.append({'Model': model, 'Cohort': split,
                     **compute_metrics(y, score, cutoff)})
table = pd.DataFrame(rows)

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = OUTPUT_DIR / "Table_S8_Pathology_Performance.csv"
table.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(table.to_string(index=False))
print(f"Saved: {csv_path}")
