"""Figures for the v3 development study (report section on the model study), drawn from its analysis outputs.

Writes docs/figures/v3_{models,ablation,calibration,twin,per_date}.png from
E:/kaggle/AI4S/experiments/neuroforecast_v3/analysis/{results.json,outer_case_table.csv}.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import report_external_figures as style  # noqa: E402,F401  (shared rcParams and palette)

FIG = ROOT / "docs/figures"
ANALYSIS = Path("E:/kaggle/AI4S/experiments/neuroforecast_v3/analysis")
SURFACE, INK, INK2, GRID = style.SURFACE, style.INK, style.INK2, style.GRID
V2, PERSIST, NEW, NEUTRAL = style.V2, style.PERSIST, style.V1, style.NEUTRAL

LABELS = {
    "persistence_day7": "Carry day 7 forward", "control_reference": "Predict no effect", "v2_incumbent": "v2 (shipped recipe)",
    "et_small_ctx": "Extra Trees + batch context", "et_full_ctx": "Extra Trees, all features + context",
    "et_small_chem": "Extra Trees + dose-series context", "ridge_full_ctx": "Ridge", "lgbm_small_ctx": "LightGBM (small)",
    "lgbm_full_ctx": "LightGBM (all)", "mlp_full_ctx": "MLP", "tabpfn_small_ctx": "TabPFN v2 (small)",
    "tabpfn_full_ctx": "TabPFN v2 (all)", "pct": "Plate-context transformer", "pct_no_context": "PCT without batch context",
    "pct_no_relations": "PCT without relation biases", "pct_no_aux": "PCT without auxiliary tasks",
    "ens_v2_pct": "v2 + PCT ensemble", "ens_tab": "v2 + LightGBM + TabPFN", "ens_all": "v2 + LightGBM + TabPFN + PCT",
    "nested_v3": "Nested v3 (selection included)",
}


def load():
    results = json.loads((ANALYSIS / "results.json").read_text(encoding="utf-8"))
    table = pd.read_csv(ANALYSIS / "outer_case_table.csv", dtype={"identity": str})
    return results, table


def color_of(name: str) -> str:
    if name == "v2_incumbent":
        return V2
    if name in ("persistence_day7", "control_reference"):
        return PERSIST
    if name.startswith("pct") or name.startswith("ens") or name == "nested_v3":
        return NEW
    return NEUTRAL


def models(results: dict) -> None:
    """Selectable candidates and the nested estimate as a dot plot, wide and short enough for a print page."""
    scores = dict(results["outer_date_macro_mae"])
    scores["nested_v3"] = results["nested_v3_date_macro_mae"]
    names = [n for n in sorted(scores, key=scores.get)
             if n != "control_reference" and not n.startswith("pct_no")]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    y = np.arange(len(names))[::-1]
    ax.hlines(y, 0.6, [scores[n] for n in names], color=GRID, lw=1.2)
    ax.scatter([scores[n] for n in names], y, s=26, color=[color_of(n) for n in names], zorder=3)
    for yi, n in zip(y, names):
        ax.text(scores[n] + 0.008, yi, f"{scores[n]:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [LABELS.get(n, n) for n in names], fontsize=7)
    ax.axvline(scores["v2_incumbent"], color=V2, lw=0.8, ls="--")
    ax.set_xlim(0.6, max(scores[n] for n in names) + 0.08)
    ax.set_xlabel("Date-macro MAE, 41 batches, outer folds (lower is better)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("Model study: every selectable candidate and the nested estimate")
    fig.tight_layout()
    fig.savefig(FIG / "v3_models.png", dpi=220)
    plt.close(fig)


def ablation(results: dict, table: pd.DataFrame) -> None:
    names = [n for n in ["pct", "pct_no_context", "pct_no_relations", "pct_no_aux"] if n in results["outer_date_macro_mae"]]
    if len(names) < 2:
        return
    scores = results["outer_date_macro_mae"]
    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, [scores[n] for n in names], color=[NEW] + [NEUTRAL] * (len(names) - 1), height=0.6)
    for yi, n in zip(y, names):
        cmp = results["comparisons"].get(f"{n}_vs_incumbent", {})
        ax.text(scores[n] + 0.005, yi, f"{scores[n]:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [LABELS[n] for n in names], fontsize=8)
    ax.axvline(scores["v2_incumbent"], color=V2, lw=0.8, ls="--", label="v2 recipe")
    ax.set_xlim(0, max(scores[n] for n in names) * 1.15)
    ax.set_xlabel("Date-macro MAE (lower is better)")
    ax.set_title("What each part of the transformer contributes")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "v3_ablation.png", dpi=200)
    plt.close(fig)


def calibration(results: dict) -> None:
    rows = []
    for tag in ["incumbent", "final"]:
        for method in ["pooled_scaled", "date_scaled"]:
            for level in ["0.8", "0.9"]:
                c = results["calibration"][tag][f"{method}_{level}"]
                rows.append((tag, method, float(level), c["date_mean"], c["worst_quintile_dates"]))
    frame = pd.DataFrame(rows, columns=["product", "method", "level", "date_mean", "worst"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, level in zip(axes, [0.8, 0.9]):
        part = frame.loc[frame.level == level]
        labels = [f"{'v2' if p == 'incumbent' else 'v3 pick'}\n{'pooled' if m == 'pooled_scaled' else 'batch-weighted'}" for p, m in zip(part["product"], part.method)]
        x = np.arange(len(part))
        ax.bar(x, part.date_mean, color=[V2 if p == "incumbent" else NEW for p in part["product"]], width=0.6)
        ax.scatter(x, part.worst, color=INK, s=14, zorder=3, label="worst fifth of batches")
        ax.axhline(level, color=INK2, lw=0.8, ls="--")
        ax.axhspan(level - 0.03, level + 0.03, color=GRID, zorder=0)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_title(f"Nominal {level:.0%}")
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("Coverage, averaged over batches")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "v3_calibration.png", dpi=200)
    plt.close(fig)


def twin(results: dict) -> None:
    """PCT twin head against carry-forward and the fair learned baseline (post hoc: one Extra Trees per target)."""
    posthoc = json.loads((ANALYSIS / "posthoc.json").read_text(encoding="utf-8"))["twin_baseline"]
    names = [n for n in posthoc if n.endswith("__d12") or n == "normalized_ns_d12"]
    names = sorted(names, key=lambda n: posthoc[n]["persistence"], reverse=True)
    readable = {"meanfiringrate": "firing rate", "burst.per.min": "burst rate", "mean.isis": "ISI within bursts",
                "per.spikes.in.burst": "% spikes in bursts", "mean.dur": "burst duration", "mean.IBIs": "inter-burst interval",
                "nAE": "active electrodes", "nABE": "bursting electrodes", "ns.n": "network spikes (contrast)",
                "ns.peak.m": "network spike peak", "ns.durn.m": "network spike duration",
                "ns.percent.of.spikes.in.ns": "% spikes in network spikes", "ns.mean.insis": "inter-network-spike interval",
                "ns.durn.sd": "network spike duration SD", "ns.mean.spikes.in.ns": "spikes per network spike",
                "r": "correlation", "mi": "mutual information"}
    pretty = ["network spikes (log2)" if n == "normalized_ns_d12" else readable[n.replace("contrast__", "").replace("__d12", "")]
              for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 4]})
    for ax, subset in [(axes[0], names[:1]), (axes[1], names[1:])]:
        y = np.arange(len(subset))[::-1]
        for key, color, label, marker in [("persistence", PERSIST, "carry day 7 forward", "o"),
                                          ("extra_trees_per_target", NEUTRAL, "Extra Trees, one per target (post hoc)", "s"),
                                          ("pct_twin", NEW, "PCT twin head (one model)", "D")]:
            ax.scatter([posthoc[n][key] for n in subset], y, s=18, color=color, marker=marker, label=label, zorder=3)
        ax.set_yticks(y, [pretty[names.index(n)] for n in subset], fontsize=6.5)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="x", color=GRID, lw=0.6)
    axes[0].set_xlim(0.6, 1.08)
    axes[1].set_xlim(0, 0.3)
    axes[1].set_xlabel("Date-macro MAE of the day-12 contrast to controls", fontsize=7)
    axes[0].set_xlabel("MAE (log2)", fontsize=7)
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle("Twin head: the whole day-12 profile from days 5 and 7", x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "v3_twin.png", dpi=220)
    plt.close(fig)


def per_date(results: dict, table: pd.DataFrame) -> None:
    import neuroforecast_gate as gate

    usable = table.usable.astype(bool)
    final = results["final_pick"]
    losses = {}
    for col in ["v2_incumbent", final, "persistence_day7"]:
        frame = table.loc[usable, ["date", "identity", "target"]].assign(prediction=table.loc[usable, f"pred__{col}"])
        losses[col] = gate.date_losses(frame)
    diff = (losses[final] - losses["v2_incumbent"]).sort_values()
    cohort = table.drop_duplicates("date").set_index("date").cohort
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(diff))
    ax.bar(x, diff.to_numpy(), color=[NEW if v < 0 else PERSIST for v in diff], width=0.75)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [f"{d}" for d in diff.index], rotation=90, fontsize=5.5)
    for xi, d in zip(x, diff.index):
        if cohort.get(d) == "archive17":
            ax.text(xi, ax.get_ylim()[0], "•", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_ylabel(f"{LABELS.get(final, final)} minus v2\n(per-batch MAE)")
    ax.set_title(f"Per batch: below zero, the v3 pick beats v2 ({int((diff < 0).sum())} of {len(diff)}; • = 2016-17 archive)")
    fig.tight_layout()
    fig.savefig(FIG / "v3_per_date.png", dpi=200)
    plt.close(fig)


def main() -> None:
    results, table = load()
    models(results)
    ablation(results, table)
    calibration(results)
    twin(results)
    per_date(results, table)
    print("figures written")


if __name__ == "__main__":
    main()
