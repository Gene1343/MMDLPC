# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
while not (PROJECT_ROOT / "modeling_pipeline.py").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Run this code from the MMDLPC_Code_PDF folder.")
    PROJECT_ROOT = PROJECT_ROOT.parent
CODE_DIR = PROJECT_ROOT / "08_Supplementary_Figures"
DATA_ROOT = PROJECT_ROOT.parent / "MMDLPC_Code_PDF_Data"
OUTPUT_DIR = CODE_DIR
sys.path.insert(0, str(CODE_DIR))
PARAMETERS_FILE = None
# %%
import matplotlib.pyplot as plt
import pandas as pd
from feature_distribution_utils import estimate_parameters, validate_parameters, fisher_scores, plot_s7, plot_s8

# %%
parameters = pd.read_csv(PARAMETERS_FILE) if PARAMETERS_FILE else estimate_parameters(DATA_ROOT)
parameters = validate_parameters(parameters)

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 9})
pdf_path = OUTPUT_DIR / 'Figure_S7_Gaussian_Densities_Recalculated.pdf'
plot_s7(parameters, pdf_path)
print(f'Saved: {pdf_path}')
