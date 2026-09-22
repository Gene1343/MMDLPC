from __future__ import annotations

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

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PDF_PATH = OUTPUT_DIR / "FigureS2_Annotation_and_Scanner_Distributions.pdf"

PURPLE = "#67236F"
ORANGE = "#FDAA3E"
RED = "#B62F4B"
BLACK = "#111111"

PATHOLOGISTS = ["TN", "ZY", "GYS"]
PATH_TCGA = np.array([211, 116, 58])
PATH_CPGEA = np.array([76, 37, 31])

SCANNER_TCGA = PATH_TCGA.copy()
SCANNER_HAMAMATSU = np.array([75, 36, 31])
SCANNER_UNICMED = np.array([1, 1, 0])

RADIOLOGISTS = ["ZQW", "HQ"]
MRI_TRAIN = np.array([74, 18])
MRI_HELD_OUT = np.array([19, 4])

MRI_TCGA = np.array([5, 2])
MRI_CPGEA_SIEMENS = np.array([85, 19])
MRI_CPGEA_GE = np.array([3, 1])

def verify_counts() -> None:

    assert np.array_equal(
        SCANNER_HAMAMATSU + SCANNER_UNICMED,
        PATH_CPGEA,
    )


def style_axis(axis: plt.Axes, ylim: tuple[int, int], tick_step: int) -> None:
    axis.set_ylim(*ylim)
    axis.set_yticks(np.arange(ylim[0], ylim[1] + 1, tick_step))
    axis.set_ylabel("Count", fontsize=12)
    axis.tick_params(axis="both", labelsize=10, width=0.9, length=4)
    axis.tick_params(axis="y", which="minor", length=2.5)
    axis.minorticks_on()
    axis.tick_params(axis="x", which="minor", bottom=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.9)
    axis.spines["bottom"].set_linewidth(0.9)

def add_labels(axis: plt.Axes, bars, pad: float = 1.2) -> None:
    for bar in bars:
        value = int(round(bar.get_height()))
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + pad,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
            color=BLACK,
        )

def panel_a(axis: plt.Axes) -> None:
    x = np.arange(len(PATHOLOGISTS))
    width = 0.39
    bars_1 = axis.bar(
        x - width / 2,
        PATH_TCGA,
        width,
        color=PURPLE,
        edgecolor=BLACK,
        linewidth=0.8,
    )
    bars_2 = axis.bar(
        x + width / 2,
        PATH_CPGEA,
        width,
        color=ORANGE,
        edgecolor=BLACK,
        linewidth=0.8,
    )
    add_labels(axis, bars_1, 1.8)
    add_labels(axis, bars_2, 1.8)
    axis.set_xticks(x, PATHOLOGISTS)
    axis.set_xlabel("Pathologists", fontsize=12)
    axis.set_title("Final pathology analytic set (post-QC)", fontsize=11, pad=8)
    style_axis(axis, (0, 250), 50)
    axis.legend(
        handles=[
            Patch(facecolor=PURPLE, edgecolor=BLACK, label=f"TCGA-PRAD (n={PATH_TCGA.sum()})"),
            Patch(facecolor=ORANGE, edgecolor=BLACK, label=f"CPGEA (n={PATH_CPGEA.sum()})"),
        ],
        frameon=False,
        fontsize=8.3,
        loc="upper right",
        handlelength=2.0,
        handletextpad=0.5,
    )

def panel_b(axis: plt.Axes) -> None:
    x = np.arange(len(PATHOLOGISTS))
    width = 0.25
    bars_1 = axis.bar(
        x - width,
        SCANNER_TCGA,
        width,
        color=PURPLE,
        edgecolor=BLACK,
        linewidth=0.8,
    )
    bars_2 = axis.bar(
        x,
        SCANNER_HAMAMATSU,
        width,
        color=ORANGE,
        edgecolor=BLACK,
        linewidth=0.8,
    )
    bars_3 = axis.bar(
        x + width,
        SCANNER_UNICMED,
        width,
        color=RED,
        edgecolor=BLACK,
        linewidth=0.8,
    )
    add_labels(axis, bars_1, 1.2)
    add_labels(axis, bars_2, 1.2)
    add_labels(axis, bars_3, 1.2)
    axis.set_xticks(x, PATHOLOGISTS)
    axis.set_xlabel("Pathologists", fontsize=12)
    axis.set_title("H&E slide-scanner distribution (post-QC)", fontsize=11, pad=8)
    style_axis(axis, (0, 250), 50)
    axis.legend(
        handles=[
            Patch(facecolor=PURPLE, edgecolor=BLACK, label=f"TCGA-PRAD (n={PATH_TCGA.sum()})"),
            Patch(
                facecolor=ORANGE,
                edgecolor=BLACK,
                label=f"CPGEA: Hamamatsu NanoZoomer S60 (n={SCANNER_HAMAMATSU.sum()})",
            ),
            Patch(facecolor=RED, edgecolor=BLACK, label=f"CPGEA: Unic-med (n={SCANNER_UNICMED.sum()})"),
        ],
        frameon=False,
        fontsize=7.5,
        loc="upper right",
        handlelength=2.0,
        handletextpad=0.5,
    )

def panel_c(axis: plt.Axes) -> None:
    x = np.arange(len(RADIOLOGISTS))
    width = 0.27
    bars_1 = axis.bar(
        x - width / 2,
        MRI_TRAIN,
        width,
        color=PURPLE,
        edgecolor=BLACK,
        linewidth=0.8,
    )
    bars_2 = axis.bar(
        x + width / 2,
        MRI_HELD_OUT,
        width,
        color=ORANGE,
        edgecolor=BLACK,
        linewidth=0.8,
    )
    add_labels(axis, bars_1, 0.7)
    add_labels(axis, bars_2, 0.7)
    axis.set_xticks(x, RADIOLOGISTS)
    axis.set_xlabel("Radiologists", fontsize=12)
    axis.set_title("Final MRI analytic set (post-QC)", fontsize=11, pad=8)
    style_axis(axis, (0, 80), 20)
    axis.legend(
        handles=[
            Patch(facecolor=PURPLE, edgecolor=BLACK, label=f"Training set (n={MRI_TRAIN.sum()})"),
            Patch(facecolor=ORANGE, edgecolor=BLACK, label=f"Held-out test set (n={MRI_HELD_OUT.sum()})"),
        ],
        frameon=False,
        fontsize=8.3,
        loc="upper right",
        handlelength=2.0,
        handletextpad=0.5,
    )

def panel_d(axis: plt.Axes) -> None:
    x = np.arange(len(RADIOLOGISTS))
    width = 0.22
    bars_1 = axis.bar(
        x - width,
        MRI_TCGA,
        width,
        color=PURPLE,
        edgecolor=BLACK,
        linewidth=0.8,
    )
    bars_2 = axis.bar(
        x,
        MRI_CPGEA_SIEMENS,
        width,
        color=ORANGE,
        edgecolor=BLACK,
        linewidth=0.8,
    )
    bars_3 = axis.bar(
        x + width,
        MRI_CPGEA_GE,
        width,
        color=RED,
        edgecolor=BLACK,
        linewidth=0.8,
    )
    add_labels(axis, bars_1, 0.7)
    add_labels(axis, bars_2, 0.7)
    add_labels(axis, bars_3, 0.7)
    axis.set_xticks(x, RADIOLOGISTS)
    axis.set_xlabel("Radiologists", fontsize=12)
    axis.set_title("MRI vendor distribution (post-QC)", fontsize=11, pad=8)
    style_axis(axis, (0, 100), 20)
    axis.legend(
        handles=[
            Patch(facecolor=PURPLE, edgecolor=BLACK, label=f"TCGA-PRAD (n={MRI_TCGA.sum()})"),
            Patch(
                facecolor=ORANGE,
                edgecolor=BLACK,
                label=f"CPGEA: Siemens Magnetom Skyra (n={MRI_CPGEA_SIEMENS.sum()})",
            ),
            Patch(facecolor=RED, edgecolor=BLACK, label=f"CPGEA: GE Discovery 750w (n={MRI_CPGEA_GE.sum()})"),
        ],
        frameon=False,
        fontsize=7.5,
        loc="upper right",
        handlelength=2.0,
        handletextpad=0.5,
    )


# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
verify_counts()
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.labelcolor": BLACK,
        "axes.edgecolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.3))
panel_a(axes[0, 0])
panel_b(axes[0, 1])
panel_c(axes[1, 0])
panel_d(axes[1, 1])

for label, axis in zip("ABCD", axes.flat):
    axis.text(
        -0.18,
        1.15,
        label,
        transform=axis.transAxes,
        fontsize=23,
        fontweight="normal",
        va="top",
        ha="left",
        clip_on=False,
    )

figure.subplots_adjust(
    left=0.085,
    right=0.985,
    top=0.92,
    bottom=0.08,
    wspace=0.30,
    hspace=0.32,
)

figure.savefig(PDF_PATH, format="pdf", facecolor="white")
plt.close(figure)

print(f"Created: {PDF_PATH}")
