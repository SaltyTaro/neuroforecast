"""Figures for the validation harness, the split-optimism study and the recalibration replay.

Writes docs/figures/split_study.png and docs/figures/lab_recalibration.png from the shipped per-case tables.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import neuroforecast_harness as h  # noqa: E402
import neuroforecast_split_study as study  # noqa: E402
import posthoc_lab_recalibration as recalibration  # noqa: E402

FIG = ROOT / "docs/figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
V2, PERSIST, V1 = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#8d8c86"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
LABELS = {"random_condition": "random\nconditions", "chemical": "chemicals\nheld out", "batch": "batches\nheld out",
          "batch_chemical": "batches +\nchemicals\nheld out"}


def split_study() -> None:
    table = pd.read_csv(ROOT / "evaluation/harness/split_study/oof.csv.gz", dtype={"identity": str})
    by = {d: table.loc[table.design == d] for d in h.DESIGNS}
    models = [("extra_trees_v2", "Extra Trees (v2 recipe)", V2), ("lightgbm", "LightGBM", V1),
              ("ridge", "ridge", NEUTRAL), ("calibrated_day7", "calibrated day-7 curve", PERSIST)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(h.DESIGNS))
    for name, label, color in models:
        mae = [study.batch_losses(by[d], name).mean() for d in h.DESIGNS]
        a.plot(x, mae, marker="o", color=color, lw=1.6, ms=5, label=label)
    a.set_xticks(x, [LABELS[d] for d in h.DESIGNS], fontsize=7.5)
    a.set_ylabel("date-macro MAE (log2)")
    a.set_title("Error looks lower when batches are shared")
    a.grid(axis="y", color=GRID, lw=0.6)
    top = max(study.batch_losses(by[d], "calibrated_day7").mean() for d in h.DESIGNS)
    a.set_ylim(None, top + 0.13)
    a.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    reference = by["batch_chemical"]
    pooled = [study.transfer_coverage(by[d], reference, "extra_trees_v2", 0.8)["pooled"] for d in h.DESIGNS]
    worst = [study.transfer_coverage(by[d], reference, "extra_trees_v2", 0.8)["worst_fifth"] for d in h.DESIGNS]
    b.bar(x - 0.18, pooled, 0.34, color=V2, label="all new batches")
    b.bar(x + 0.18, worst, 0.34, color="#9cc3ee", label="worst fifth of batches")
    b.axhline(0.8, color=INK2, lw=0.8, ls="--")
    b.text(-0.5, 0.807, "nominal 0.80", ha="left", va="bottom", fontsize=7, color=INK2)
    for i, v in enumerate(pooled):
        b.text(i - 0.18, v + 0.012, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    b.set_ylim(0.4, 0.97)
    b.set_xticks(x, [LABELS[d] for d in h.DESIGNS], fontsize=7.5)
    b.set_ylabel("coverage on new batches")
    b.set_title("80% intervals validated that way")
    b.legend(frameon=False, fontsize=7, loc="upper right", ncol=2, bbox_to_anchor=(1.02, 1.03))
    fig.tight_layout()
    fig.savefig(FIG / "split_study.png", dpi=200)
    plt.close(fig)


def lab_recalibration() -> None:
    cases = recalibration.external()
    shipped = recalibration.shipped_scale(cases, 0.8)
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    lines = [("shipped scale (fitted on development)", None, NEUTRAL, None), ("refit on every earlier batch", "all", V2, None),
             ("refit on the last six batches", 6, V1, None)]
    for label, window, color, _ in lines:
        frame = recalibration.replay(cases, 0.8, None if window in (None, "all") else window)
        frame = frame.loc[frame.lab_scale.notna()]
        scale = shipped if window is None else frame.lab_scale
        hit = ((frame.target - frame.prediction).abs() <= scale * frame.trust_score)
        per_batch = hit.groupby(frame.order).mean()
        running = per_batch.expanding().mean()
        ax.plot(running.index, running.values, color=color, lw=1.8, label=label)
        ax.scatter(per_batch.index, per_batch.values, color=color, s=9, alpha=0.45, zorder=2)
    ax.axhline(0.8, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("external batch, in date order (the first two only build the history)")
    ax.set_ylabel("coverage of 80% intervals")
    ax.set_ylim(0.25, 1.03)
    ax.set_title("A lab's own day-12 checks lift pooled coverage; the worst batches stay low")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.text(0.01, 0.04, "lines: running coverage; dots: each batch", transform=ax.transAxes, fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "lab_recalibration.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    split_study()
    lab_recalibration()
    print("wrote docs/figures/split_study.png and docs/figures/lab_recalibration.png")
