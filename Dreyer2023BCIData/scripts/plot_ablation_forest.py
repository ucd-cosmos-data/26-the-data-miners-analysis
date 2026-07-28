"""Generate a simple ablation forest plot from the consolidated summary."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import FIGURES, RESULTS  # noqa: E402
from src.visualization.style import apply_style, save_figure  # noqa: E402


def main() -> None:
    apply_style()
    summary = pd.read_csv(RESULTS / "tables" / "ablation_summary_full.csv")
    exclude = {"time_over_template", "nuisance_over_time", "template_plus_time_and_nuisance_controls"}
    summary = summary[~summary["ablation"].isin(exclude)].dropna(subset=["differential_partial_r2_mean"])
    summary = summary.sort_values("differential_partial_r2_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    y = np.arange(len(summary))
    colors = ["#B00020" if "circularity" in str(v) else "#1B6CA8" for v in summary["validity_label"]]
    ax.hlines(y, 0, summary["differential_partial_r2_mean"], color=colors, linewidth=2.2)
    ax.scatter(summary["differential_partial_r2_mean"], y, c=colors, s=42, zorder=3, edgecolors="white", linewidths=0.5)
    ax.axvline(0.005861, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(summary["ablation"])
    ax.set_xlabel("Mean confirmatory differential partial R²")
    ax.set_title("Ablation forest (sensitivity/circularity; not a new confirmatory endpoint)")
    handles = [
        Line2D([0], [0], color="#1B6CA8", marker="o", linestyle="None", label="Sensitivity / fixed series"),
        Line2D([0], [0], color="#B00020", marker="o", linestyle="None", label="Circularity (not primary)"),
        Line2D([0], [0], color="black", linestyle="--", label="Primary confirm mean"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)
    save_figure(fig, FIGURES / "ablation_forest_plot.png")
    print({"wrote": str(FIGURES / "ablation_forest_plot.png"), "n": len(summary)})


if __name__ == "__main__":
    main()
