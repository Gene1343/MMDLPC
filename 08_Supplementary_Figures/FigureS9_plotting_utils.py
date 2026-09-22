from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent / "07_Model_Scores"
LABELS_PATH = SCRIPT_DIR.parent / "00_Shared_Data_and_Code/Data/CPGEA-TCGA 20230106 OK.csv"

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
        "Clinical",
    ),
    SelectedModel(
        "P",
        "Path",
        "Pathology",
        "P  Pathology (LightGBM)",
        "LightGBM",
        "LightGBM",
        "#C983B6",
        "Pathology",
    ),
    SelectedModel(
        "R",
        "Rad",
        "Radiology",
        "R  Radiology (SVM)",
        "SVM",
        "SVM",
        "#76B7D8",
        "Radiology",
    ),
    SelectedModel(
        "CP",
        "Clinic-Path",
        "Patho-Clinical",
        "CP  Patho-Clinical (RF)",
        "RandomForest",
        "Random Forest",
        "#59A14F",
        "Clinical_Pathology",
    ),
    SelectedModel(
        "CR",
        "Clinic-Rad",
        "Radio-Clinical",
        "CR  Radio-Clinical (LR)",
        "LR",
        "Logistic regression",
        "#4C9F70",
        "Clinical_Radiology",
    ),
    SelectedModel(
        "PR",
        "Path-Rad",
        "Patho-Radiology",
        "PR  Patho-Radiology (NB)",
        "NaiveBayes",
        "Naive Bayes",
        "#7A5AA6",
        "Pathology_Radiology",
    ),
    SelectedModel(
        "CPR",
        "Combined",
        "Patho-Radio-Clinical",
        "CPR  Patho-Radio-Clinical (MLP)",
        "MLP",
        "MLP",
        "#D46A6A",
        "Clinical_Pathology_Radiology",
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
        raise RuntimeError("Duplicate patient IDs in the label table.")
    if not labels["HRR_ANY"].isin([0, 1]).all():
        raise RuntimeError("Labels must be HRD=1 or HRP=0.")
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

