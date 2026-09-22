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
import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc, average_precision_score, brier_score_loss, precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TOPIC_DIR = OUTPUT_DIR
SEED = 20260805
FINAL_PARAMETERS = {
    "Pathology": {"C": 0.001, "gamma": 0.01675, "class_weight": "balanced"},
    "Radiology": {"C": 0.0003, "gamma": 0.003, "class_weight": None},
}
FEATURE_FILES = {
    "Pathology": "fixed85_pathology_features.csv",
    "Radiology": "fixed85_radiology_features.csv",
}

def metric_row(split, model_name, frame):
    y = frame.HRR_ANY.to_numpy(dtype=int)
    score = frame.score.to_numpy(dtype=float)
    precision, recall, _ = precision_recall_curve(y, score)
    return {
        "split": split, "model": model_name, "n": len(y),
        "positive": int(y.sum()), "negative": int((y == 0).sum()),
        "roc_auc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
        "pr_auc_trapezoidal": auc(recall, precision),
        "brier_score": brier_score_loss(y, score),
    }

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# %%
original = pd.read_csv(DATA_ROOT / "04_Multimodal/fixed85_predictions_final.csv")
if original.duplicated(["ID", "model"]).any():
    raise ValueError("Duplicate archived patient-model rows")
reference = original.loc[original.model.eq("Patho-Radiology")].copy()
train = reference.loc[reference.split.eq("Training")]
test = reference.loc[reference.split.eq("Validation")]
if reference.ID.duplicated().any():
    raise ValueError("Duplicate fixed-cohort IDs")
# %%
predictions = [reference.copy()]
for model_name, parameters in FINAL_PARAMETERS.items():
    features = pd.read_csv(
        DATA_ROOT / "04_Multimodal" / FEATURE_FILES[model_name], float_precision="round_trip"
    ).set_index("ID")
    if features.index.duplicated().any() or set(features.index) != set(reference.ID):
        raise ValueError(f"Feature IDs do not match the locked cohort: {model_name}")
    features = features.astype(float)
    if np.isinf(features.to_numpy()).any():
        raise ValueError(f"Infinite feature values: {model_name}")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=False,
                      random_state=SEED, **parameters)),
    ])
    calibration_arguments = {
        "method": "sigmoid",
        "cv": StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
    }

    if "estimator" in CalibratedClassifierCV().get_params():
        calibration_arguments["estimator"] = pipeline
    else:
        calibration_arguments["base_estimator"] = pipeline
    model = CalibratedClassifierCV(**calibration_arguments)
    model.fit(features.loc[train.ID], train.HRR_ANY.to_numpy(dtype=int))
    generated = reference.copy()
    generated["model"] = model_name
    generated["score"] = model.predict_proba(features.loc[generated.ID])[:, 1]
    predictions.append(generated)
# %%
predictions = pd.concat(predictions, ignore_index=True)
metrics = pd.DataFrame([
    metric_row(split, model_name, frame)
    for (split, model_name), frame in predictions.groupby(["split", "model"])
])
comparison = predictions.merge(
    original[["ID", "model", "split", "score"]].rename(columns={"score": "archived_score"}),
    on=["ID", "model", "split"], validate="one_to_one",
)
comparison["difference"] = comparison.score - comparison.archived_score
differences = comparison.groupby("model").difference.apply(lambda x: float(x.abs().max()))
# %%
predictions.to_csv(TOPIC_DIR / "fixed85_refit_selected_predictions_R1_recalculated.csv",
                   index=False, encoding="utf-8-sig")
metrics.to_csv(TOPIC_DIR / "fixed85_refit_selected_metrics_R1_recalculated.csv",
              index=False, encoding="utf-8-sig")
import sklearn
provenance = {
    "analysis": "Saved fixed-85 selected-feature analysis, not raw-feature pipeline CV",
    "parameter_source": "Final specified Pathology gamma 0.01675; archived Radiology gamma 0.003",
    "parameters": FINAL_PARAMETERS,
    "seed": SEED, "sklearn_version": sklearn.__version__,
    "calibration": "sigmoid, stratified 3-fold training only",
    "unchanged_model": "Patho-Radiology: archived NaiveBayes probabilities",
    "maximum_absolute_score_difference_from_archive": differences.to_dict(),
}
(TOPIC_DIR / "fixed85_selected_parameter_provenance_R1_recalculated.json").write_text(
    json.dumps(provenance, indent=2), encoding="utf-8"
)
print(metrics.to_string(index=False))
print("Maximum absolute score differences versus archive:")
print(differences.to_string())
