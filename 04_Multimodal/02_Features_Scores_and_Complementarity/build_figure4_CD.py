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
RAW_ROOT = DATA_ROOT / "07_Model_Scores"
# %%
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


@dataclass(frozen=True)
class SignatureSpec:
    code: str
    signature: str
    model: str
    score_subdir: str

SIGNATURES = (
    SignatureSpec("C", "Clinical", "KNN", "Clinical"),
    SignatureSpec("P", "Path", "LightGBM", "Pathology"),
    SignatureSpec("R", "Rad", "SVM", "Radiology"),
    SignatureSpec("CP", "Clinic-Path", "RandomForest", "Clinical_Pathology"),
    SignatureSpec("CR", "Clinic-Rad", "LR", "Clinical_Radiology"),
    SignatureSpec("PR", "Path-Rad", "NaiveBayes", "Pathology_Radiology"),
    SignatureSpec("CPR", "Combined", "MLP", "Clinical_Pathology_Radiology"),
)

DENSITY_ORDER = ["CPR", "PR", "CP", "CR", "R", "P", "C"]
CORRELATION_ORDER = ["CPR", "PR", "CP", "CR", "P", "R", "C"]

RIDGE_CMAP = LinearSegmentedColormap.from_list(
    "ridge_purple_orange",
    ["#6A00A8", "#9C179E", "#CC4778", "#ED7953", "#FDB32D"],
)

def normalize_id(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.nii(?:\.gz)?$", "", regex=True, case=False)
    )

def load_raw_predictions(raw_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for specification in SIGNATURES:
        signature_frames: list[pd.DataFrame] = []
        for split in ("train", "test"):
            source = (
                raw_root
                / specification.score_subdir
                / f"{specification.signature}_{specification.model}_{split}.csv"
            )
            if not source.exists():
                raise FileNotFoundError(f"Missing locked score file: {source}")
            raw = pd.read_csv(source)
            positive_columns = [column for column in raw.columns if column.endswith("-1")]
            if "ID" not in raw.columns or len(positive_columns) != 1:
                raise RuntimeError(f"Unexpected prediction columns in {source}")
            frame = raw[["ID", positive_columns[0]]].rename(
                columns={positive_columns[0]: "score"}
            )
            frame["ID"] = normalize_id(frame["ID"])
            frame["score"] = pd.to_numeric(frame["score"], errors="raise")
            frame["split"] = split
            frame["source_file"] = source.name
            signature_frames.append(frame)

        signature_data = pd.concat(signature_frames, ignore_index=True)
        if signature_data["ID"].duplicated().any():
            raise RuntimeError(f"Duplicate pooled IDs for {specification.code}")
        if not signature_data["score"].between(0.0, 1.0).all():
            raise RuntimeError(f"Scores outside [0, 1] for {specification.code}")

        signature_data["code"] = specification.code
        signature_data["signature"] = specification.signature
        signature_data["model"] = specification.model
        rows.append(signature_data)

    return pd.concat(rows, ignore_index=True, sort=False)

def load_scores(raw_root: Path | None) -> pd.DataFrame:
    return load_raw_predictions(raw_root or RAW_ROOT)

def build_wide_complete_case(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = scores.pivot(index="ID", columns="code", values="score").reset_index()
    complete = wide.dropna(subset=CORRELATION_ORDER).copy()
    return wide, complete

def build_matrices(complete: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    computed = complete[CORRELATION_ORDER].corr(method="spearman")
    return computed, computed.copy()

def kde_profile(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.unique(values).size < 2:
        density = np.zeros_like(x_grid)
        density[np.argmin(np.abs(x_grid - values[0]))] = 1.0
        return density
    density = gaussian_kde(values)(x_grid)
    maximum = float(np.max(density))
    return density / maximum if maximum > 0 else density

def draw_density_panel(
    ax: plt.Axes,
    scores: pd.DataFrame,
) -> None:
    base_positions = dict(
        zip(DENSITY_ORDER, range(len(DENSITY_ORDER) - 1, -1, -1))
    )

    for code in DENSITY_ORDER:
        base = float(base_positions[code])
        x_grid = np.linspace(0.0, 1.0, 501)
        values = scores.loc[scores["code"] == code, "score"].to_numpy(dtype=float)
        height = 1.42 * kde_profile(values, x_grid)
        ridge = base + height

        vertices = np.column_stack(
            [
                np.concatenate([[0.0], x_grid, [1.0]]),
                np.concatenate([[base], ridge, [base]]),
            ]
        )
        patch = PathPatch(
            MplPath(vertices),
            facecolor="none",
            edgecolor="black",
            linewidth=1.25,
            joinstyle="round",
            zorder=3,
        )
        ax.add_patch(patch)
        gradient = np.linspace(0.0, 1.0, 512)[None, :]
        image = ax.imshow(
            gradient,
            extent=(0.0, 1.0, base, base + max(1.45, float(np.max(height)) + 0.02)),
            origin="lower",
            aspect="auto",
            cmap=RIDGE_CMAP,
            interpolation="bicubic",
            zorder=2,
        )
        image.set_clip_path(patch)
        ax.axhline(base, color="#B8B8B8", linewidth=0.75, zorder=1)

    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(-0.70, len(DENSITY_ORDER) + 1.60)
    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=10.5)
    ax.set_yticks([base_positions[code] for code in DENSITY_ORDER])
    ax.set_yticklabels(DENSITY_ORDER, fontsize=11)
    ax.tick_params(axis="x", direction="out", width=1.1, length=4, pad=3)
    ax.tick_params(axis="y", length=0, pad=7)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.25)
    ax.grid(False)

def draw_correlation_panel(
    ax: plt.Axes,
    colorbar_axis: plt.Axes,
    display_matrix: pd.DataFrame,
) -> None:
    matrix = display_matrix.loc[CORRELATION_ORDER, CORRELATION_ORDER].to_numpy()
    mask = np.tril(np.ones_like(matrix, dtype=bool), k=-1)
    masked = np.ma.array(matrix, mask=mask)
    image = ax.imshow(masked, cmap="RdBu", vmin=-1.0, vmax=1.0, interpolation="none")

    size = len(CORRELATION_ORDER)
    ax.set_xticks(range(size))
    ax.set_xticklabels(CORRELATION_ORDER, fontsize=10.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(size))
    ax.set_yticklabels(CORRELATION_ORDER, fontsize=10.5)
    ax.tick_params(axis="both", length=0, pad=4)

    for row in range(size):
        for column in range(row, size):
            value = matrix[row, column]
            text = "1" if row == column else f"{value:.2f}"
            color = "white" if abs(value) >= 0.65 else "black"
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = plt.colorbar(image, cax=colorbar_axis)
    ticks = np.linspace(-1.0, 1.0, 11)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:.1f}" for value in ticks])
    colorbar.ax.tick_params(labelsize=7.5, length=2, width=0.8, pad=2)
    colorbar.outline.set_linewidth(1.0)

def save_all_formats(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    fig.savefig(output_dir / f"{stem}.svg", facecolor="white")
    fig.savefig(output_dir / f"{stem}_600dpi.png", dpi=600, facecolor="white")
    fig.savefig(output_dir / f"{stem}_preview.png", dpi=100, facecolor="white")

def build_combined(
    scores: pd.DataFrame,
    display_matrix: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(9.85, 3.79), facecolor="white")
    density_axis = fig.add_axes([0.081, 0.115, 0.385, 0.810])
    correlation_axis = fig.add_axes([0.565, 0.075, 0.315, 0.850])
    colorbar_axis = fig.add_axes([0.891, 0.075, 0.011, 0.850])

    draw_density_panel(density_axis, scores)
    draw_correlation_panel(correlation_axis, colorbar_axis, display_matrix)
    fig.text(0.013, 0.962, "C", fontsize=22, ha="left", va="top")
    fig.text(0.497, 0.962, "D", fontsize=22, ha="left", va="top")
    save_all_formats(fig, output_dir, "Figure4CD_with_CR_R1_recalculated")
    plt.close(fig)

def build_separate_panels(
    scores: pd.DataFrame,
    display_matrix: pd.DataFrame,
    output_dir: Path,
) -> None:
    density_figure = plt.figure(figsize=(4.72, 3.79), facecolor="white")
    density_axis = density_figure.add_axes([0.17, 0.115, 0.80, 0.810])
    draw_density_panel(density_axis, scores)
    density_figure.text(0.025, 0.962, "C", fontsize=22, ha="left", va="top")
    save_all_formats(density_figure, output_dir, "Figure4C_scores_with_CR_R1_recalculated")
    plt.close(density_figure)

    correlation_figure = plt.figure(figsize=(4.85, 3.79), facecolor="white")
    correlation_axis = correlation_figure.add_axes([0.17, 0.075, 0.68, 0.850])
    colorbar_axis = correlation_figure.add_axes([0.875, 0.075, 0.022, 0.850])
    draw_correlation_panel(correlation_axis, colorbar_axis, display_matrix)
    correlation_figure.text(0.025, 0.962, "D", fontsize=22, ha="left", va="top")
    save_all_formats(correlation_figure, output_dir, "Figure4D_correlation_with_CR_R1_recalculated")
    plt.close(correlation_figure)

def write_audit_outputs(
    scores: pd.DataFrame,
    complete: pd.DataFrame,
    computed: pd.DataFrame,
    display: pd.DataFrame,
    output_dir: Path,
) -> None:
    complete[["ID", *CORRELATION_ORDER]].to_csv(
        output_dir / "Figure4CD_complete_case_scores_n85_R1_recalculated.csv", index=False
    )
    computed.to_csv(output_dir / "Figure4D_spearman_n85_R1_recalculated.csv")

    manifest_rows = []
    for specification in SIGNATURES:
        subset = scores.loc[scores["code"] == specification.code]
        manifest_rows.append(
            {
                "code": specification.code,
                "signature": specification.signature,
                "model": specification.model,
                "pooled_n": len(subset),
                "train_n": int((subset["split"] == "train").sum()),
                "test_n": int((subset["split"] == "test").sum()),
                "score_min": float(subset["score"].min()),
                "score_max": float(subset["score"].max()),
                "correlation_complete_case_n": len(complete),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "Figure4CD_model_and_sample_counts_R1_recalculated.csv", index=False
    )

# %%
output_dir = OUTPUT_DIR.resolve()
output_dir.mkdir(parents=True, exist_ok=True)

raw_root = RAW_ROOT.resolve() if RAW_ROOT else None
# %%
scores = load_scores(raw_root)
_, complete = build_wide_complete_case(scores)
# %%
computed, display = build_matrices(complete)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

# %%
write_audit_outputs(scores, complete, computed, display, output_dir)
build_combined(scores, display, output_dir)
build_separate_panels(scores, display, output_dir)
