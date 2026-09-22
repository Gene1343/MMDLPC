from __future__ import annotations

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
while not (PROJECT_ROOT / "modeling_pipeline.py").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Run this code from the MMDLPC_Code_PDF folder.")
    PROJECT_ROOT = PROJECT_ROOT.parent
CODE_DIR = PROJECT_ROOT / "04_Multimodal/03_Cohort_Comparison"
DATA_ROOT = PROJECT_ROOT.parent / "MMDLPC_Code_PDF_Data"
OUTPUT_DIR = CODE_DIR
sys.path.insert(0, str(CODE_DIR))

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

TOPIC_DIR = OUTPUT_DIR
PREDICTIONS = DATA_ROOT / "04_Multimodal/fixed85_predictions_final.csv"
FIGURE_3X3 = TOPIC_DIR / "Figure_S4_fixed85_ROC_PR_DCA_R1_recalculated.png"
FIGURE_3X3_PDF = TOPIC_DIR / "Figure_S4_fixed85_ROC_PR_DCA_R1_recalculated.pdf"
BOOTSTRAPS = 1_000
SEED = 20260730

MODELS = {
    "Pathology": "#6F4C9B",
    "Radiology": "#1B9E77",
    "Patho-Radiology": "#D95F5F",
}

def pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, score)
    return float(auc(recall, precision))

def decision_curve(
    y_true: np.ndarray, score: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    total = len(y_true)
    result = np.empty_like(thresholds, dtype=float)
    for index, threshold in enumerate(thresholds):
        predicted = score >= threshold
        true_positive = np.sum(predicted & (y_true == 1))
        false_positive = np.sum(predicted & (y_true == 0))
        result[index] = (
            true_positive / total
            - false_positive / total * threshold / (1.0 - threshold)
        )
    return result

def stratified_bootstrap(
    y_true: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    return np.concatenate(
        [
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ]
    )

def interpolate_pr(
    y_true: np.ndarray, score: np.ndarray, recall_grid: np.ndarray
) -> np.ndarray:
    precision, recall, _ = precision_recall_curve(y_true, score)
    recall, precision = recall[::-1], precision[::-1]
    right = np.searchsorted(recall, recall_grid, side="right")
    left = np.clip(right - 1, 0, len(recall) - 1)
    right = np.clip(right, 0, len(recall) - 1)
    width = recall[right] - recall[left]
    weight = np.divide(
        recall_grid - recall[left], width,
        out=np.zeros_like(recall_grid, dtype=float), where=width > 0,
    )
    return precision[left] + weight * (precision[right] - precision[left])

def bootstrap_summary(
    y_true: np.ndarray,
    score: np.ndarray,
    roc_grid: np.ndarray,
    recall_grid: np.ndarray,
    thresholds: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    roc_values = np.empty((BOOTSTRAPS, len(roc_grid)))
    pr_values = np.empty((BOOTSTRAPS, len(recall_grid)))
    dca_values = np.empty((BOOTSTRAPS, len(thresholds)))
    for bootstrap_index in range(BOOTSTRAPS):
        index = stratified_bootstrap(y_true, rng)
        sampled_y = y_true[index]
        sampled_score = score[index]
        fpr, tpr, _ = roc_curve(sampled_y, sampled_score)
        interpolated_roc = np.interp(roc_grid, fpr, tpr)

        interpolated_roc[0] = 0.0
        interpolated_roc[-1] = 1.0
        roc_values[bootstrap_index] = interpolated_roc
        pr_values[bootstrap_index] = interpolate_pr(
            sampled_y, sampled_score, recall_grid
        )
        dca_values[bootstrap_index] = decision_curve(
            sampled_y, sampled_score, thresholds
        )

    def summarize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            values.mean(axis=0),
            np.quantile(values, 0.025, axis=0),
            np.quantile(values, 0.975, axis=0),
        )

    return {
        "roc": summarize(roc_values),
        "pr": summarize(pr_values),
        "dca": summarize(dca_values),
    }

def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.2, width=0.8, labelsize=8.5)
    axis.grid(False)

def render_figure(
    data: pd.DataFrame,
    splits: tuple[str, ...],
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )
    row_count = len(splits)
    figure_height = 7.2 if row_count == 2 else 10.0
    figure, axes = plt.subplots(row_count, 3, figsize=(11.2, figure_height))
    top = 0.89 if row_count == 2 else 0.92
    bottom = 0.10 if row_count == 2 else 0.075
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=top,
        bottom=bottom,
        wspace=0.34,
        hspace=0.40,
    )
    for column, title in enumerate(("ROC", "Precision–recall", "Decision curve")):
        axes[0, column].set_title(title, fontsize=12, fontweight="bold", pad=10)

    panel_labels = list("ABCDEFGHI")[: row_count * 3]
    roc_grid = np.linspace(0.0, 1.0, 61)
    recall_grid = np.linspace(0.0, 1.0, 51)
    thresholds = np.linspace(0.05, 0.60, 28)
    rng = np.random.default_rng(SEED)

    for row, split in enumerate(splits):
        roc_axis, pr_axis, dca_axis = axes[row]
        split_data = data.loc[data["split"].eq(split)]
        reference = split_data.loc[split_data["model"].eq("Pathology")]
        y_true = reference["HRR_ANY"].to_numpy(dtype=int)
        prevalence = float(y_true.mean())

        roc_axis.plot(
            [0, 1], [0, 1], linestyle=(0, (4, 3)), color="#8A8A8A", linewidth=1.1
        )
        pr_axis.axhline(
            prevalence, linestyle=(0, (4, 3)), color="#8A8A8A", linewidth=1.1
        )

        for model, color in MODELS.items():
            subset = split_data.loc[split_data["model"].eq(model)]
            if not np.array_equal(
                subset["HRR_ANY"].to_numpy(dtype=int), y_true
            ):
                raise RuntimeError(f"Outcome order mismatch for {split}/{model}.")
            score = subset["score"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y_true, score)
            empirical_dca = decision_curve(y_true, score, thresholds)
            summary = bootstrap_summary(
                y_true, score, roc_grid, recall_grid, thresholds, rng
            )

            _, roc_lower, roc_upper = summary["roc"]
            _, pr_lower, pr_upper = summary["pr"]
            _, dca_lower, dca_upper = summary["dca"]

            overlap_style = split == "Validation" and model == "Pathology"
            line_style = "solid"
            marker = None
            marker_size = 0.0
            line_width = 1.5 if overlap_style else 2.0
            line_zorder = 5 if overlap_style else 3

            roc_axis.fill_between(
                roc_grid, roc_lower, roc_upper, color=color, alpha=0.07, linewidth=0
            )
            roc_axis.plot(
                fpr,
                tpr,
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                marker=marker,
                markersize=marker_size,
                markevery=None,
                markerfacecolor="white" if overlap_style else color,
                markeredgewidth=0.8 if overlap_style else 0.0,
                zorder=line_zorder,
                drawstyle="steps-post",
                label=f"{model}  AUC {roc_auc_score(y_true, score):.3f}",
            )

            pr_axis.fill_between(
                recall_grid, pr_lower, pr_upper, color=color, alpha=0.07, linewidth=0
            )
            precision, recall, _ = precision_recall_curve(y_true, score)
            pr_axis.plot(
                recall,
                precision,
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                marker=marker,
                markersize=marker_size,
                markevery=5 if overlap_style else None,
                markerfacecolor="white" if overlap_style else color,
                markeredgewidth=0.8 if overlap_style else 0.0,
                zorder=line_zorder,
                label=f"{model}  PR-AUC {pr_auc(y_true, score):.3f}",
            )

            dca_axis.fill_between(
                thresholds,
                dca_lower,
                dca_upper,
                color=color,
                alpha=0.06,
                linewidth=0,
            )
            dca_axis.plot(
                thresholds,
                empirical_dca,
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                marker=marker,
                markersize=marker_size,
                markevery=4 if overlap_style else None,
                markerfacecolor="white" if overlap_style else color,
                markeredgewidth=0.8 if overlap_style else 0.0,
                zorder=line_zorder,
                label=model,
            )

        treat_all = prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds)
        dca_axis.plot(
            thresholds,
            treat_all,
            color="#777777",
            linewidth=1.25,
            linestyle=(0, (4, 3)),
            label="Treat all",
        )
        dca_axis.axhline(
            0,
            color="#222222",
            linewidth=1.1,
            linestyle=(0, (1.5, 2.2)),
            label="Treat none",
        )

        roc_axis.set(
            xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02),
            xlabel="1 − Specificity",
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
        for axis in (roc_axis, pr_axis, dca_axis):
            style_axis(axis)
        roc_axis.legend(loc="upper right", fontsize=7.4, handlelength=1.7)
        pr_axis.legend(loc="upper right", fontsize=7.4, handlelength=1.7)
        dca_axis.legend(loc="upper right", fontsize=7.2, handlelength=1.7)

        row_position = roc_axis.get_position()
        figure.text(
            0.018,
            (row_position.y0 + row_position.y1) / 2,
            split,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11.5,
            fontweight="bold",
        )

    for panel_index, axis in enumerate(axes.flat):
        axis.text(
            -0.17,
            1.08,
            panel_labels[panel_index],
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )
    figure.suptitle(
        "Fixed Patho-Radiology cohort development and validation",
        fontsize=13,
        fontweight="bold",
        y=0.985 if row_count == 2 else 0.992,
    )
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    if output_path == FIGURE_3X3:
        figure.savefig(FIGURE_3X3_PDF, bbox_inches="tight")
        print(f"Saved: {FIGURE_3X3_PDF}")
    plt.close(figure)
    print(f"Saved: {output_path}")

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(PREDICTIONS)
expected = {"Pathology", "Radiology", "Patho-Radiology"}
if set(data["model"]) != expected or data.duplicated(["model", "ID"]).any():
    raise ValueError("Unexpected model identity or duplicate patient predictions")
if not data["score"].between(0, 1).all() or not data["HRR_ANY"].isin([0, 1]).all():
    raise ValueError("Invalid scores or labels")
for split in ("Training", "Validation"):
    subset = data.loc[data["split"].eq(split)]
    if subset.empty or subset.groupby("ID")["model"].nunique().ne(len(expected)).any():
        raise ValueError(f"Models must contain the same patient IDs in {split}")
    if (subset.groupby("ID")["HRR_ANY"].nunique() != 1).any():
        raise ValueError("Conflicting HRD labels across models")

# %%
all_cohort = data.copy()
all_cohort["split"] = "All cohort"
three_row_data = pd.concat([all_cohort, data], ignore_index=True)
# %%
render_figure(
    three_row_data,
    ("All cohort", "Training", "Validation"),
    FIGURE_3X3,
)
