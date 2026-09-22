from __future__ import annotations

from pathlib import Path
PROJECT_ROOT = None

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

SCRIPT_DIR = None
PROJECT_ROOT = None
OUTPUT_DIR = None
INDIVIDUAL_DIR = None
LABELS_PATH = None

BOOTSTRAPS = 1_000
SEED = 20260801
ROC_GRID = np.linspace(0.0, 1.0, 61)
RECALL_GRID = np.linspace(0.0, 1.0, 51)
THRESHOLDS = np.linspace(0.05, 0.60, 28)

@dataclass(frozen=True)
class SelectedModel:
    code: str
    signature: str
    display_name: str
    row_label: str
    model: str
    model_label: str
    colour: str
    score_dir: str

SELECTED_MODELS = (
    SelectedModel(
        "C",
        "Clinical",
        "Clinical",
        "C  Clinical (KNN)",
        "KNN",
        "KNN",
        "#E8A64E",
        "07_Model_Scores/Clinical",
    ),
    SelectedModel(
        "P",
        "Path",
        "Pathology",
        "P  Pathology (LightGBM)",
        "LightGBM",
        "LightGBM",
        "#C983B6",
        "07_Model_Scores/Pathology",
    ),
    SelectedModel(
        "R",
        "Rad",
        "Radiology",
        "R  Radiology (SVM)",
        "SVM",
        "SVM",
        "#76B7D8",
        "07_Model_Scores/Radiology",
    ),
    SelectedModel(
        "CP",
        "Clinic-Path",
        "Patho-Clinical",
        "CP  Patho-Clinical (RF)",
        "RandomForest",
        "Random Forest",
        "#59A14F",
        "07_Model_Scores/Clinical_Pathology",
    ),
    SelectedModel(
        "CR",
        "Clinic-Rad",
        "Radio-Clinical",
        "CR  Radio-Clinical (LR)",
        "LR",
        "Logistic regression",
        "#4C9F70",
        "07_Model_Scores/Clinical_Radiology",
    ),
    SelectedModel(
        "PR",
        "Path-Rad",
        "Patho-Radiology",
        "PR  Patho-Radiology (NB)",
        "NaiveBayes",
        "Naive Bayes",
        "#7A5AA6",
        "07_Model_Scores/Pathology_Radiology",
    ),
    SelectedModel(
        "CPR",
        "Combined",
        "Patho-Radio-Clinical",
        "CPR  Patho-Radio-Clinical (MLP)",
        "MLP",
        "MLP",
        "#D46A6A",
        "07_Model_Scores/Clinical_Pathology_Radiology",
    ),
)

def normalize_id(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.nii(?:\.gz)?$", "", regex=True, case=False)
    )

def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_PATH, usecols=["ID", "HRR_ANY"])
    labels["ID"] = normalize_id(labels["ID"])
    labels["HRR_ANY"] = pd.to_numeric(labels["HRR_ANY"], errors="raise").astype(int)
    if labels["ID"].duplicated().any():
        raise RuntimeError("Duplicate patient IDs in labels_locked.csv.")
    return labels

def load_selected_predictions(
    specification: SelectedModel, labels: pd.DataFrame
) -> pd.DataFrame:
    split_frames: list[pd.DataFrame] = []
    for split in ("train", "test"):
        path = (
            PROJECT_ROOT
            / specification.score_dir
            / f"{specification.signature}_{specification.model}_{split}.csv"
        )
        raw = pd.read_csv(path)
        positive_columns = [column for column in raw.columns if column.endswith("-1")]
        if "ID" not in raw.columns or len(positive_columns) != 1:
            raise RuntimeError(f"Unexpected prediction columns in {path.name}.")
        frame = raw[["ID", positive_columns[0]]].rename(
            columns={positive_columns[0]: "score"}
        )
        frame["ID"] = normalize_id(frame["ID"])
        frame["score"] = pd.to_numeric(frame["score"], errors="raise")
        frame["split"] = split
        frame["source_file"] = path.name
        split_frames.append(frame)

    predictions = pd.concat(split_frames, ignore_index=True, sort=False)
    predictions = predictions.merge(labels, how="left", on="ID", validate="one_to_one")
    if predictions["HRR_ANY"].isna().any():
        missing = predictions.loc[predictions["HRR_ANY"].isna(), "ID"].tolist()
        raise RuntimeError(
            f"Unmatched labels for {specification.code}: {missing[:5]}"
        )
    if predictions["ID"].duplicated().any():
        raise RuntimeError(
            f"Train and test files overlap for {specification.code}."
        )
    if not predictions["score"].between(0.0, 1.0).all():
        raise RuntimeError(f"Scores outside [0, 1] for {specification.code}.")
    predictions["HRR_ANY"] = predictions["HRR_ANY"].astype(int)
    return predictions

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
    rng: np.random.Generator,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    roc_values = np.empty((BOOTSTRAPS, len(ROC_GRID)))
    pr_values = np.empty((BOOTSTRAPS, len(RECALL_GRID)))
    dca_values = np.empty((BOOTSTRAPS, len(THRESHOLDS)))

    for bootstrap_index in range(BOOTSTRAPS):
        index = stratified_bootstrap(y_true, rng)
        sampled_y = y_true[index]
        sampled_score = score[index]

        false_positive_rate, true_positive_rate, _ = roc_curve(
            sampled_y, sampled_score
        )
        interpolated_roc = np.interp(
            ROC_GRID, false_positive_rate, true_positive_rate
        )
        interpolated_roc[0] = 0.0
        interpolated_roc[-1] = 1.0
        roc_values[bootstrap_index] = interpolated_roc
        pr_values[bootstrap_index] = interpolate_pr(
            sampled_y, sampled_score, RECALL_GRID
        )
        dca_values[bootstrap_index] = decision_curve(
            sampled_y, sampled_score, THRESHOLDS
        )

    def summarize(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    axis.tick_params(direction="out", length=3.2, width=0.8, labelsize=8.3)
    axis.grid(False)

    axis.set_box_aspect(0.82)

def plot_model_row(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    specification: SelectedModel,
    predictions: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, object]:
    roc_axis, pr_axis, dca_axis = axes
    y_true = predictions["HRR_ANY"].to_numpy(dtype=int)
    score = predictions["score"].to_numpy(dtype=float)
    prevalence = float(y_true.mean())
    roc_auc = float(roc_auc_score(y_true, score))
    trapezoid_pr_auc = pr_auc(y_true, score)
    empirical_dca = decision_curve(y_true, score, THRESHOLDS)
    summary = bootstrap_summary(y_true, score, rng)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, score)
    _, roc_lower, roc_upper = summary["roc"]
    _, pr_lower, pr_upper = summary["pr"]
    _, dca_lower, dca_upper = summary["dca"]

    roc_axis.plot(
        [0, 1],
        [0, 1],
        linestyle=(0, (4, 3)),
        color="#8A8A8A",
        linewidth=1.0,
    )
    roc_axis.fill_between(
        ROC_GRID,
        roc_lower,
        roc_upper,
        color=specification.colour,
        alpha=0.10,
        linewidth=0,
    )
    roc_axis.plot(
        false_positive_rate,
        true_positive_rate,
        color=specification.colour,
        linewidth=2.0,
        drawstyle="steps-post",
    )
    roc_axis.text(
        0.04,
        0.04,
        f"AUC = {roc_auc:.3f}",
        transform=roc_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=specification.colour,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )

    pr_axis.axhline(
        prevalence,
        linestyle=(0, (4, 3)),
        color="#8A8A8A",
        linewidth=1.0,
    )
    pr_axis.fill_between(
        RECALL_GRID,
        pr_lower,
        pr_upper,
        color=specification.colour,
        alpha=0.10,
        linewidth=0,
    )
    precision, recall, _ = precision_recall_curve(y_true, score)
    pr_axis.plot(
        recall,
        precision,
        color=specification.colour,
        linewidth=2.0,
    )
    pr_axis.text(
        0.96,
        0.04,
        f"PR-AUC = {trapezoid_pr_auc:.3f}",
        transform=pr_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=specification.colour,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )

    dca_axis.fill_between(
        THRESHOLDS,
        dca_lower,
        dca_upper,
        color=specification.colour,
        alpha=0.09,
        linewidth=0,
    )
    dca_axis.plot(
        THRESHOLDS,
        empirical_dca,
        color=specification.colour,
        linewidth=2.0,
        label=specification.model_label,
    )
    treat_all = prevalence - (1.0 - prevalence) * THRESHOLDS / (
        1.0 - THRESHOLDS
    )
    dca_axis.plot(
        THRESHOLDS,
        treat_all,
        color="#777777",
        linewidth=1.15,
        linestyle=(0, (4, 3)),
        label="Treat all",
    )
    dca_axis.axhline(
        0,
        color="#222222",
        linewidth=1.05,
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
    dca_axis.legend(loc="upper right", fontsize=7.0, handlelength=1.7)

    for axis in (roc_axis, pr_axis, dca_axis):
        style_axis(axis)

    return {
        "modality": specification.code,
        "signature": specification.signature,
        "display_name": specification.display_name,
        "selected_model": specification.model,
        "ROC_AUC_apparent": roc_auc,
        "PR_AUC_trapezoid_apparent": trapezoid_pr_auc,
        "Brier_score_apparent": float(np.mean((score - y_true) ** 2)),
        "interpretation": (
            "Saved training and held-out test predictions combined; "
            "descriptive apparent performance, not independent validation."
        ),
    }

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def create_figure(
    specifications: tuple[SelectedModel, ...],
    prediction_map: dict[str, pd.DataFrame],
    *,
    title: str,
) -> tuple[plt.Figure, list[dict[str, object]]]:
    row_count = len(specifications)
    figure_height = 3.25 * row_count + 0.75
    figure, raw_axes = plt.subplots(
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
        ("ROC", "Precision–recall", "Decision curve")
    ):
        raw_axes[0, column].set_title(
            column_title, fontsize=12, fontweight="bold", pad=9
        )

    panel_labels = [chr(ord("A") + index) for index in range(row_count * 3)]
    metrics: list[dict[str, object]] = []
    for row, specification in enumerate(specifications):
        metrics.append(
            plot_model_row(
                tuple(raw_axes[row]),
                specification,
                prediction_map[specification.code],
                np.random.default_rng(SEED + row),
            )
        )
        row_position = raw_axes[row, 0].get_position()
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

    for panel_index, axis in enumerate(raw_axes.flat):
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

def save_main_outputs(
    prediction_map: dict[str, pd.DataFrame]
) -> list[dict[str, object]]:
    title = "Selected modality models using all saved training and held-out samples"
    figure, metrics = create_figure(
        SELECTED_MODELS,
        prediction_map,
        title=title,
    )
    stem = OUTPUT_DIR / "Figure_SelectedModels_AllSamples_ROC_PR_DCA_7x3"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)

    multipage_path = OUTPUT_DIR / "Figure_SelectedModels_AllSamples_ROC_PR_DCA_2page.pdf"
    with PdfPages(multipage_path) as pdf:
        for page_index, page_specs in enumerate(
            (SELECTED_MODELS[:4], SELECTED_MODELS[4:]), start=1
        ):
            page_figure, _ = create_figure(
                page_specs,
                prediction_map,
                title=(
                    "Selected modality models using all saved training and held-out "
                    f"samples (page {page_index} of 2)"
                ),
            )
            pdf.savefig(page_figure, bbox_inches="tight")
            plt.close(page_figure)

    return metrics

def save_individual_pngs(prediction_map: dict[str, pd.DataFrame]) -> None:
    for specification in SELECTED_MODELS:
        figure, _ = create_figure(
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
        path = INDIVIDUAL_DIR / f"{specification.code}_SelectedModel_AllSamples_ROC_PR_DCA.png"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
