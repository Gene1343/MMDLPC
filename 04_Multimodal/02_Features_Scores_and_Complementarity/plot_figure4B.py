from __future__ import annotations

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
while not (PROJECT_ROOT / "modeling_pipeline.py").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Run this code from the MMDLPC_Code_PDF folder.")
    PROJECT_ROOT = PROJECT_ROOT.parent
CODE_DIR = PROJECT_ROOT / "04_Multimodal/02_Features_Scores_and_Complementarity"
DATA_ROOT = PROJECT_ROOT.parent / "MMDLPC_Code_PDF_Data"
OUTPUT_DIR = CODE_DIR
sys.path.insert(0, str(CODE_DIR))

# %%
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.metrics import roc_auc_score

PACKAGE_ROOT = DATA_ROOT
SCORE_DIR = PACKAGE_ROOT / "07_Model_Scores"
LABEL_FILE = PACKAGE_ROOT / "00_Shared_Data_and_Code/Data/CPGEA-TCGA 20230106 OK.csv"

MODEL_ORDER = [
    "NaiveBayes",
    "LR",
    "LightGBM",
    "KNN",
    "SVM",
    "RandomForest",
    "MLP",
    "GradientBoosting",
    "ExtraTrees",
    "XGBoost",
]

SOURCES = {
    "AUC-C+P": ("Clinical_Pathology", "Clinic-Path"),
    "AUC-C+P+R": ("Clinical_Pathology_Radiology", "Combined"),
    "AUC-P+R": ("Pathology_Radiology", "Path-Rad"),
    "AUC-C+R": ("Clinical_Radiology", "Clinic-Rad"),
}

STYLE = {
    "AUC-C+P": {"line": "#118383", "marker": "#E0AEC6"},
    "AUC-C+P+R": {"line": "#D77E7F", "marker": "#2AAC3E"},
    "AUC-P+R": {"line": "#EFC554", "marker": "#6691CB"},
    "AUC-C+R": {"line": "#7A5195", "marker": "#A978C0"},
}

def load_auc_series() -> dict[str, list[float]]:
    labels = pd.read_csv(LABEL_FILE, usecols=["ID", "HRR_ANY"]).set_index("ID")["HRR_ANY"]
    assert labels.index.is_unique
    series, rows = {}, []
    for label, (folder, prefix) in SOURCES.items():
        series[label] = []
        for model in MODEL_ORDER:
            source = SCORE_DIR / folder / f"{prefix}_{model}_test.csv"
            frame = pd.read_csv(source)
            frame["ID"] = frame["ID"].str.replace(r"\.nii(?:\.gz)?$", "", regex=True)
            assert frame["ID"].is_unique
            score = frame["HRR_ANY-1"]
            y = labels.reindex(frame["ID"])
            assert y.notna().all() and score.between(0, 1).all()
            auc = float(roc_auc_score(y, score))
            series[label].append(auc)
            rows.append({"model": model, "series": label, "n_test": len(y),
                         "heldout_auc": auc, "source": str(source.relative_to(PACKAGE_ROOT))})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "Figure4B_AUC_values_R1_recalculated.csv", index=False)
    return series

def build_panel(panel_label: str, output_stem: str, series: dict[str, list[float]]) -> None:

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(4.72, 4.48), facecolor="white")
    ax = fig.add_axes([0.195, 0.219, 0.799, 0.712])
    x = list(range(len(MODEL_ORDER)))

    plotted_handles: list[Line2D] = []
    for label, aucs in series.items():
        style = STYLE[label]
        handle = ax.plot(
            x,
            aucs,
            color=style["line"],
            linewidth=1.30,
            marker="o",
            markersize=6.25,
            markerfacecolor=style["marker"],
            markeredgecolor="#040000",
            markeredgewidth=1.15,
            label=label,
            zorder=3,
        )[0]
        plotted_handles.append(handle)

    ax.set_xlim(-0.60, 9.60)
    ax.set_ylim(-1.0 / 6.0, 7.0 / 6.0)
    ax.set_ylabel("AUC", fontsize=16.5, labelpad=2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        MODEL_ORDER,
        rotation=48,
        ha="right",
        rotation_mode="anchor",
        fontsize=10.5,
    )
    yticks = [0.00, 0.25, 0.50, 0.75, 1.00]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{value:.2f}" for value in yticks], fontsize=10.5)
    ax.tick_params(axis="both", direction="out", length=4.0, width=1.2, pad=2)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#040000")
        spine.set_linewidth(1.25)

    model_text = Line2D([], [], linestyle="none", marker=None, label="Model")
    legend = ax.legend(
        handles=[model_text, *plotted_handles],
        labels=["Model", *series.keys()],
        loc="center",
        bbox_to_anchor=(0.50, 0.185),
        ncol=5,
        frameon=False,
        fontsize=7.05,
        handlelength=1.15,
        handletextpad=0.20,
        columnspacing=0.52,
        borderaxespad=0.0,
        markerscale=0.88,
    )
    for text in legend.get_texts():
        text.set_color("#040000")

    fig.text(0.040, 0.962, panel_label, fontsize=22, ha="left", va="top")

    fig.savefig(OUTPUT_DIR / f"{output_stem}_600dpi.png", dpi=600, facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{output_stem}.pdf", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{output_stem}.svg", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{output_stem}_preview.png", dpi=100, facecolor="white")
    plt.close(fig)

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
build_panel("B", "Figure4B_AUC_R1_recalculated", load_auc_series())
