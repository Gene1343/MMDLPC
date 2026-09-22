from __future__ import annotations

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
while not (PROJECT_ROOT / "modeling_pipeline.py").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Run this code from the MMDLPC_Code_PDF folder.")
    PROJECT_ROOT = PROJECT_ROOT.parent
CODE_DIR = PROJECT_ROOT / "04_Multimodal/01_Modeling_and_Evaluation"
DATA_ROOT = PROJECT_ROOT.parent / "MMDLPC_Code_PDF_Data"
OUTPUT_DIR = CODE_DIR
sys.path.insert(0, str(CODE_DIR))

# %%
import pandas as pd
import selected_models_plotting as plotting
plotting.PROJECT_ROOT = DATA_ROOT
plotting.SCRIPT_DIR = CODE_DIR
plotting.OUTPUT_DIR = OUTPUT_DIR
plotting.INDIVIDUAL_DIR = OUTPUT_DIR
plotting.LABELS_PATH = DATA_ROOT / '00_Shared_Data_and_Code/Data/CPGEA-TCGA 20230106 OK.csv'
INDIVIDUAL_DIR = OUTPUT_DIR
from selected_models_plotting import (
    SELECTED_MODELS, configure_matplotlib, load_labels, load_selected_predictions,
    save_main_outputs, save_individual_pngs,
)

# %%
configure_matplotlib()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

# %%
labels = load_labels()
prediction_map = {
    specification.code: load_selected_predictions(specification, labels)
    for specification in SELECTED_MODELS
}
# %%
metrics = save_main_outputs(prediction_map)
save_individual_pngs(prediction_map)
pd.DataFrame(metrics).to_csv(
    OUTPUT_DIR / "selected_models_all_samples_metric_audit.csv", index=False
)

for row in metrics:
    print(
        f"{row['modality']}: {row['selected_model']} | "
        f"ROC AUC={row['ROC_AUC_apparent']:.3f} | "
        f"PR-AUC={row['PR_AUC_trapezoid_apparent']:.3f}"
    )
print(f"Saved outputs to: {OUTPUT_DIR}")
