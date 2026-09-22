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
import math
import numpy as np
import pandas as pd

TOPIC_DIR = DATA_ROOT / "04_Multimodal"
MODELS = (("P", "Pathology"), ("R", "Radiology"), ("P-R", "Patho-Radiology"))
SCAN_PERCENTILES = tuple(range(10, 100, 10))


def load_predictions(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"ID", "HRR_ANY", "split", "model", "score"}
    missing = required.difference(data.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    if set(data["model"].unique()) != {name for _, name in MODELS}:
        raise RuntimeError("Prediction file does not contain the three locked models.")
    if set(data["split"].unique()) != {"Training", "Validation"}:
        raise RuntimeError("Expected Training and Validation source splits.")
    if data.duplicated(["ID", "model"]).any():
        raise RuntimeError("Duplicate patient-model rows were found.")
    if data['ID'].isna().any() or data.groupby('ID')['HRR_ANY'].nunique().ne(1).any():
        raise ValueError('Each patient must have one consistent observed HRR_ANY label')
    if data.groupby('ID')['split'].nunique().ne(1).any() or data.groupby('ID')['model'].nunique().ne(3).any():
        raise ValueError('All three models must use the same patients and split assignments')
    return data


def youden_threshold(y_true: np.ndarray, score: np.ndarray) -> float:


    thresholds = np.sort(np.unique(score))
    candidates: list[tuple[float, float]] = []
    for threshold in thresholds:
        predicted = score >= threshold
        sensitivity = np.mean(predicted[y_true == 1])
        specificity = np.mean(~predicted[y_true == 0])
        candidates.append((float(sensitivity + specificity - 1.0), float(threshold)))
    best = max(value for value, _ in candidates)
    return max(
        threshold
        for value, threshold in candidates
        if np.isclose(value, best, atol=1e-12, rtol=0.0)
    )


def clopper_pearson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan

    def binomial_tail(probability: float, start: int) -> float:
        return sum(
            math.comb(total, value)
            * probability**value
            * (1.0 - probability) ** (total - value)
            for value in range(start, total + 1)
        )

    def binomial_cdf(probability: float, end: int) -> float:
        return sum(
            math.comb(total, value)
            * probability**value
            * (1.0 - probability) ** (total - value)
            for value in range(0, end + 1)
        )

    lower = 0.0
    if successes > 0:
        left, right = 0.0, 1.0
        for _ in range(80):
            midpoint = (left + right) / 2.0
            if binomial_tail(midpoint, successes) < 0.025:
                left = midpoint
            else:
                right = midpoint
        lower = (left + right) / 2.0

    upper = 1.0
    if successes < total:
        left, right = 0.0, 1.0
        for _ in range(80):
            midpoint = (left + right) / 2.0
            if binomial_cdf(midpoint, successes) > 0.025:
                left = midpoint
            else:
                right = midpoint
        upper = (left + right) / 2.0
    return lower, upper


def metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = score >= threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    tn = int(np.sum(~predicted & (y_true == 0)))
    fp = int(np.sum(predicted & (y_true == 0)))
    fn = int(np.sum(~predicted & (y_true == 1)))
    sensitivity = tp / (tp + fn)
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    sens_ci = clopper_pearson(tp, tp + fn)
    npv_ci = clopper_pearson(tn, tn + fn)
    return {
        "sensitivity": sensitivity,
        "sensitivity_95ci_lower_exact": sens_ci[0],
        "sensitivity_95ci_upper_exact": sens_ci[1],
        "npv": npv,
        "npv_95ci_lower_exact": npv_ci[0],
        "npv_95ci_upper_exact": npv_ci[1],
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }



# %%
data = load_predictions(TOPIC_DIR / "fixed85_predictions_final.csv")
if not data.score.between(0, 1).all() or not data.HRR_ANY.isin([0, 1]).all():
    raise ValueError("Expected probabilities and binary HRR_ANY labels")
# %%
scan_rows: list[dict[str, object]] = []
youden_rows: list[dict[str, object]] = []
for code, model in MODELS:
    model_data = data.loc[data["model"].eq(model)]
    training = model_data.loc[model_data["split"].eq("Training")]
    test = model_data.loc[model_data["split"].eq("Validation")]
    y_train = training["HRR_ANY"].to_numpy(dtype=int)
    s_train = training["score"].to_numpy(dtype=float)
    y_test = test["HRR_ANY"].to_numpy(dtype=int)
    s_test = test["score"].to_numpy(dtype=float)
    for percentile in SCAN_PERCENTILES:
        threshold = float(
            np.percentile(s_train, percentile, method="linear")
        )
        training_metrics = metrics(y_train, s_train, threshold)
        test_metrics = metrics(y_test, s_test, threshold)
        scan_rows.append(
            {
                "modality": code,
                "model": model,
                "training_score_percentile": percentile,
                "training_probability_cutoff": threshold,
                **{
                    f"training_{key}": value
                    for key, value in training_metrics.items()
                },
                **{
                    f"held_out_test_{key}": value
                    for key, value in test_metrics.items()
                },
            }
        )

    training_cutoff = youden_threshold(y_train, s_train)
    training_metrics = metrics(y_train, s_train, training_cutoff)
    test_metrics = metrics(y_test, s_test, training_cutoff)
    youden_rows.append(
        {
            "modality": code,
            "model": model,
            "training_probability_cutoff": training_cutoff,
            "training_youden_percentile_rank": 100.0
            * float(np.mean(s_train <= training_cutoff)),
            **{
                f"training_{key}": value
                for key, value in training_metrics.items()
            },
            **{
                f"held_out_test_{key}": value
                for key, value in test_metrics.items()
            },
        }
    )
scan, youden = pd.DataFrame(scan_rows), pd.DataFrame(youden_rows)
# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
scan.to_csv(OUTPUT_DIR / "Table_S10_training_percentile_cutoffs_R1_recalculated.csv", index=False, encoding="utf-8-sig")
youden.to_csv(OUTPUT_DIR / "Table_S10_training_Youden_cutoffs_R1_recalculated.csv", index=False, encoding="utf-8-sig")
print(youden[["modality", "training_probability_cutoff", "training_youden_percentile_rank", "held_out_test_sensitivity", "held_out_test_npv"]].to_string(index=False))
