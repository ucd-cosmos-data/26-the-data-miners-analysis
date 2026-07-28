"""Shared matplotlib helpers for readable research figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.framealpha": 0.95,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
        }
    )


def legend_outside(ax, loc: str = "upper left", ncol: int = 1, **kwargs):
    """Place a legend outside the axes so it does not cover the data."""
    return ax.legend(
        loc=loc,
        bbox_to_anchor=(1.02, 1.0) if "right" in loc or loc == "upper left" else (0.5, -0.18),
        borderaxespad=0.0,
        ncol=ncol,
        frameon=True,
        **kwargs,
    )


def legend_below(ax_or_fig, handles=None, labels=None, ncol: int = 2, **kwargs):
    """Place a legend below a figure or axes."""
    target = ax_or_fig
    if handles is None or labels is None:
        return target.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=ncol,
            frameon=True,
            **kwargs,
        )
    return target.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=ncol,
        frameon=True,
        **kwargs,
    )


def save_figure(fig, path: Path, *, rect: tuple[float, float, float, float] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rect is not None:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
