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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

import selected_models_plotting as plotting
plotting.PROJECT_ROOT = DATA_ROOT
plotting.SCRIPT_DIR = CODE_DIR
plotting.OUTPUT_DIR = OUTPUT_DIR
plotting.INDIVIDUAL_DIR = OUTPUT_DIR
plotting.LABELS_PATH = DATA_ROOT / '00_Shared_Data_and_Code/Data/CPGEA-TCGA 20230106 OK.csv'
INDIVIDUAL_DIR = OUTPUT_DIR

from selected_models_plotting import (
    RECALL_GRID,
    ROC_GRID,
    SCRIPT_DIR,
    SEED,
    SELECTED_MODELS,
    THRESHOLDS,
    SelectedModel,
    bootstrap_summary,
    configure_matplotlib,
    decision_curve,
    load_labels,
    load_selected_predictions,
    pr_auc,
    style_axis,
)


EVALUATION_STYLES = (
    ("All", "All samples", "#6F4C9B"),
    ("Training", "Training", "#1B9E77"),
    ("Test", "Test", "#D95F5F"),
)

def evaluation_data(
    predictions: pd.DataFrame,
) -> tuple[tuple[str, str, str, pd.DataFrame], ...]:
    return (
        ("All", "All samples", "#6F4C9B", predictions),
        (
            "Training",
            "Training",
            "#1B9E77",
            predictions.loc[predictions["split"].eq("train")].copy(),
        ),
        (
            "Test",
            "Test",
            "#D95F5F",
            predictions.loc[predictions["split"].eq("test")].copy(),
        ),
    )

def add_metric_block(
    axis: plt.Axes,
    entries: list[tuple[str, float, str]],
    *,
    metric_label: str,
    horizontal_alignment: str,
) -> None:
    x = 0.04 if horizontal_alignment == "left" else 0.96
    for row, (short_label, value, colour) in enumerate(entries):
        axis.text(
            x,
            0.04 + (len(entries) - 1 - row) * 0.055,
            f"{short_label} {metric_label} = {value:.3f}",
            transform=axis.transAxes,
            ha=horizontal_alignment,
            va="bottom",
            fontsize=6.7,
            fontweight="bold",
            color=colour,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 0.8,
            },
        )

def plot_overlay_row(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    specification: SelectedModel,
    predictions: pd.DataFrame,
    model_index: int,
) -> list[dict[str, object]]:
    roc_axis, pr_axis, dca_axis = axes
    roc_axis.plot(
        [0, 1],
        [0, 1],
        linestyle=(0, (4, 3)),
        color="#8A8A8A",
        linewidth=1.0,
    )
    dca_axis.axhline(
        0,
        color="#222222",
        linewidth=1.05,
        linestyle=(0, (1.5, 2.2)),
    )

    roc_metrics: list[tuple[str, float, str]] = []
    pr_metrics: list[tuple[str, float, str]] = []
    metric_rows: list[dict[str, object]] = []

    for evaluation_index, (
        short_label,
        evaluation_label,
        colour,
        data,
    ) in enumerate(evaluation_data(predictions)):
        y_true = data["HRR_ANY"].to_numpy(dtype=int)
        score = data["score"].to_numpy(dtype=float)
        prevalence = float(y_true.mean())
        roc_auc = float(roc_auc_score(y_true, score))
        trapezoid_pr_auc = pr_auc(y_true, score)
        empirical_dca = decision_curve(y_true, score, THRESHOLDS)
        summary = bootstrap_summary(
            y_true,
            score,
            np.random.default_rng(SEED + model_index * 10 + evaluation_index),
        )

        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, score)
        _, roc_lower, roc_upper = summary["roc"]
        _, pr_lower, pr_upper = summary["pr"]
        _, dca_lower, dca_upper = summary["dca"]

        roc_axis.fill_between(
            ROC_GRID,
            roc_lower,
            roc_upper,
            color=colour,
            alpha=0.055,
            linewidth=0,
        )
        roc_axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=colour,
            linewidth=1.75,
            drawstyle="steps-post",
        )

        pr_axis.fill_between(
            RECALL_GRID,
            pr_lower,
            pr_upper,
            color=colour,
            alpha=0.055,
            linewidth=0,
        )
        precision, recall, _ = precision_recall_curve(y_true, score)
        pr_axis.plot(
            recall,
            precision,
            color=colour,
            linewidth=1.75,
        )

        pr_axis.axhline(
            prevalence,
            color=colour,
            linewidth=0.7,
            linestyle=(0, (4, 3)),
            alpha=0.45,
        )

        dca_axis.fill_between(
            THRESHOLDS,
            dca_lower,
            dca_upper,
            color=colour,
            alpha=0.045,
            linewidth=0,
        )
        dca_axis.plot(
            THRESHOLDS,
            empirical_dca,
            color=colour,
            linewidth=1.75,
        )
        treat_all = prevalence - (1.0 - prevalence) * THRESHOLDS / (
            1.0 - THRESHOLDS
        )
        dca_axis.plot(
            THRESHOLDS,
            treat_all,
            color=colour,
            linewidth=0.7,
            linestyle=(0, (4, 3)),
            alpha=0.45,
        )

        roc_metrics.append((short_label, roc_auc, colour))
        pr_metrics.append((short_label, trapezoid_pr_auc, colour))
        metric_rows.append(
            {
                "modality": specification.code,
                "signature": specification.signature,
                "display_name": specification.display_name,
                "selected_model": specification.model,
                "evaluation_set": evaluation_label,
                "ROC_AUC": roc_auc,
                "PR_AUC_trapezoid": trapezoid_pr_auc,
                "Brier_score": float(np.mean((score - y_true) ** 2)),
            }
        )

    add_metric_block(
        roc_axis,
        roc_metrics,
        metric_label="AUC",
        horizontal_alignment="left",
    )
    add_metric_block(
        pr_axis,
        pr_metrics,
        metric_label="PR-AUC",
        horizontal_alignment="right",
    )

    roc_axis.set(
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        xlabel="1 - Specificity",
        ylabel="Sensitivity",
    )
    pr_axis.set(
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        xlabel="Recall",
        ylabel="Precision",
    )
    dca_axis.set(
        xlim=(0.05, 0.60),
        ylim=(-0.09, 0.24),
        xlabel="Threshold probability",
        ylabel="Net benefit",
    )
    dca_axis.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    legend_handles = [
        Line2D([0], [0], color=colour, linewidth=1.75, label=short_label)
        for short_label, _, colour in EVALUATION_STYLES
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#777777",
                linewidth=0.9,
                linestyle=(0, (4, 3)),
                label="Treat all (set-specific)",
            ),
            Line2D(
                [0],
                [0],
                color="#222222",
                linewidth=1.05,
                linestyle=(0, (1.5, 2.2)),
                label="Treat none",
            ),
        ]
    )
    dca_axis.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=6.4,
        handlelength=1.7,
        labelspacing=0.25,
    )

    for axis in (roc_axis, pr_axis, dca_axis):
        style_axis(axis)

    return metric_rows

def create_overlay_figure(
    specifications: tuple[SelectedModel, ...],
    prediction_map: dict[str, pd.DataFrame],
    *,
    title: str,
) -> tuple[plt.Figure, list[dict[str, object]]]:
    row_count = len(specifications)
    figure_height = 3.25 * row_count + 0.75
    figure, axes = plt.subplots(
        row_count,
        3,
        figsize=(11.2, figure_height),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.125,
        right=0.985,
        top=0.948,
        bottom=0.035,
        wspace=0.34,
        hspace=0.36,
    )
    for column, column_title in enumerate(
        ("ROC", "Precision-recall", "Decision curve")
    ):
        axes[0, column].set_title(
            column_title, fontsize=12, fontweight="bold", pad=9
        )

    metrics: list[dict[str, object]] = []
    panel_labels = [chr(ord("A") + index) for index in range(row_count * 3)]
    for row, specification in enumerate(specifications):
        metrics.extend(
            plot_overlay_row(
                tuple(axes[row]),
                specification,
                prediction_map[specification.code],
                row,
            )
        )
        row_position = axes[row, 0].get_position()
        figure.text(
            0.025,
            (row_position.y0 + row_position.y1) / 2,
            specification.row_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
        )

    for panel_index, axis in enumerate(axes.flat):
        axis.text(
            -0.17,
            1.08,
            panel_labels[panel_index],
            transform=axis.transAxes,
            fontsize=11.5,
            fontweight="bold",
            va="top",
            ha="left",
        )

    figure.suptitle(title, fontsize=13, fontweight="bold", y=0.992)
    return figure, metrics

def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")

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
overview, metrics = create_overlay_figure(
    SELECTED_MODELS,
    prediction_map,
    title="Selected models: all samples, training, and test overlays",
)
# %%
save_figure(
    overview,
    OUTPUT_DIR / "Figure_SelectedModels_All_Training_Test_Overlay_7x3",
)
plt.close(overview)

for specification in SELECTED_MODELS:
    figure, _ = create_overlay_figure(
        (specification,),
        prediction_map,
        title=f"{specification.display_name}: selected {specification.model_label} model",
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.82,
        bottom=0.18,
        wspace=0.34,
    )
    stem = INDIVIDUAL_DIR / (
        f"{specification.code}_SelectedModel_All_Training_Test_Overlay"
    )
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)

pd.DataFrame(metrics).to_csv(
    OUTPUT_DIR / "selected_models_split_overlay_metric_audit.csv",
    index=False,
)
for row in metrics:
    print(
        f"{row['modality']} | {row['evaluation_set']}: "
        f"ROC AUC={row['ROC_AUC']:.3f} | "
        f"PR-AUC={row['PR_AUC_trapezoid']:.3f}"
    )
print(f"Saved outputs to: {OUTPUT_DIR}")
