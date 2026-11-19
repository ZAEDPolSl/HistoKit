import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from matplotlib.lines import Line2D

ARTIFACT_COLUMNS = {
    "Tissue Fold": "Tissue_Fold_pct",
    "Dark Spot / Foreign Object": "Dark_Spot_Foreign_Object_pct",
    "Pen": "Pen_pct",
    "Edge / Air Bubble": "Edge_Air_Bubble_pct",
    "Out of Focus": "Out_of_Focus_pct",
    "Total artifacts": "artifact_pct",
}

ARTIFACT_COLORS = {
    "Tissue Fold": "#FF6347",
    "Dark Spot / Foreign Object": "#00FF00",
    "Pen": "#FF0000",
    "Edge / Air Bubble": "#FF00FF",
    "Out of Focus": "#4B0082",
    "Total artifacts": "#03dbfc",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default="C:/Users/julia/Desktop/HistoKit_10_10_2025/slide_stats.csv")
    parser.add_argument(
        "--out", type=Path, default=Path("artifact_plots")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset="basename")

    labels = list(ARTIFACT_COLUMNS)
    data = [
        pd.to_numeric(df[col], errors="coerce").dropna().values
        for col in ARTIFACT_COLUMNS.values()
    ]

    # Większe czcionki do posteru
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.labelsize": 19,
        "axes.titlesize": 21,
        "xtick.labelsize": 15,
        "ytick.labelsize": 16,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Mniejsze odstępy między boxplotami
    positions = np.arange(len(labels)) * 0.75 + 1

    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.52,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"color": "#555555", "linewidth": 1.5},
        capprops={"color": "#555555", "linewidth": 1.5},
        flierprops={
            "marker": "o",
            "markersize": 5,
            "markerfacecolor": "#555555",
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
    )

    for box, label in zip(bp["boxes"], labels):
        color = ARTIFACT_COLORS[label]
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.65)
        box.set_linewidth(1.8)

    ax.set_xticks(positions)
    ax.set_xticklabels([
        "Tissue\nFold",
        "Dark Spot /\nForeign Object",
        "Pen",
        "Edge /\nAir Bubble",
        "Out of\nFocus",
        "Total\nArtifacts",
    ])

    ax.set_ylabel("Artifact area (% of tissue)", fontsize=19)
    ax.set_title(
        "Distribution of histological artifacts by type",
        fontsize=21,
        fontweight="bold",
        pad=18,
    )

    ax.set_xlim(positions[0] - 0.45, positions[-1] + 0.45)
    ax.set_yscale("log")

    ax.tick_params(axis="both", length=0, pad=8)

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"color": "#555555", "linewidth": 1.2},
        capprops={"color": "#555555", "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "#555555",
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
    )

    for box, label in zip(bp["boxes"], labels):
        color = ARTIFACT_COLORS[label]
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.65)
        box.set_linewidth(1.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([
        "Tissue\nFold",
        "Dark Spot /\nForeign Object",
        "Pen",
        "Edge /\nAir Bubble",
        "Out of\nFocus",
        "Total\nArtifacts",
    ])

    ax.set_ylabel("Artifact area (% of non background pixels)")
    ax.set_title(
        "Distribution of histological artifacts by type",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )

    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    for ext in ["pdf", "svg", "png"]:
        fig.savefig(
            args.out / f"artifacts_boxplot.{ext}",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()


if __name__ == "__main__":
    main()

