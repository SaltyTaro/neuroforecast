"""Figures and a compact summary from the saved gate outputs; development and, if present, the single reserve scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import neuroforecast_data as data
import neuroforecast_gate as gate

LABELS = {
    "extra_trees_relative_day7_reliability": "Control-relative day 7 + reliability indicators",
    "extra_trees_relative_day7": "Control-relative, day 7",
    "extra_trees_activity_coordination": "Activity + coordination, days 5 + 7",
    "extra_trees_relative": "Control-relative, days 5 + 7",
    "extra_trees_all": "Original full-feature Extra Trees",
    "ridge_relative": "Control-relative Ridge",
    "control_reference": "Assume control-like day 12",
    "persistence_day7": "Carry day 7 forward",
    "extra_trees_target_only": "Target-only Extra Trees",
    "ridge_target_only": "Target-only Ridge",
    "ridge_all": "Original full-feature Ridge",
    "linear_trend": "Extrapolate day 5 to day 7 trend",
    "nested_selection": "Nested selection estimate",
    "product_model": "Product forecast",
    "persistence_rank": "Rank by day-7 change",
}
BLUE, ORANGE, GREEN, GREY, RED = "#176A9A", "#C67B25", "#3A7D44", "#BEC7CF", "#B03A2E"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig, root: Path, name: str) -> None:
    fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(root / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label(name: str) -> str:
    return LABELS.get(name, name)


def model_comparison(dev: dict, figures: Path) -> None:
    macro = dev["date_macro_mae"]
    product = dev["product_model"]
    names = list(macro) + ["nested_selection"]
    values = list(macro.values()) + [dev["nested_selection"]["nested_date_macro_mae"]]
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = [ORANGE if names[i] == product else (GREEN if names[i] == "nested_selection" else (BLUE if names[i] == dev["strongest_simple_comparator"] else GREY)) for i in order]
    ax.barh([label(names[i]) for i in order], [values[i] for i in order], color=colors)
    for k, i in enumerate(order):
        ax.text(values[i] + 0.01, k, f"{values[i]:.3f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Date-macro MAE, log2 relative network spikes (lower is better)")
    ax.set_title("Development comparison; orange = product model, green = nested estimate of the selection procedure", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, figures, "gate_model_comparison")


def warning_curves(summary: dict, figures: Path, name: str, title: str) -> None:
    warning = summary["warning"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, subgroup in zip(axes, ["all", "early_quiet"]):
        for policy, color in [("product_model", ORANGE), (summary["strongest_simple_comparator"], BLUE), ("persistence_rank", GREY)]:
            records = [r for r in warning[policy] if r["subgroup"] == subgroup]
            budgets = [r["budget"] for r in records]
            recall = [r["pooled"]["recall"] if r["pooled"]["recall"] is not None else np.nan for r in records]
            ax.plot(budgets, recall, marker="o", color=color, label=label(policy))
        records = [r for r in warning["product_model"] if r["subgroup"] == subgroup]
        ax.plot([r["budget"] for r in records], [r["pooled"]["random_expected_recall"] for r in records], linestyle="--", color=RED, label="Random within budget")
        positives = records[0]["pooled"]["positives"]
        ax.set_title(f"{'All cases' if subgroup == 'all' else 'Early-quiet cases'} ({positives} later large changes)", fontsize=10)
        ax.set_xlabel("Review budget (fraction of cases flagged per date)")
        ax.set_xticks([0.1, 0.2, 0.3])
        ax.set_ylim(0, 1.02)
        ax.axvline(0.2, color=GREY, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recall of later large changes")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=10)
    save_figure(fig, figures, name)


def coverage_figure(summary: dict, figures: Path, name: str, title: str) -> None:
    coverage = summary["coverage"]
    level = 0.8
    dates = list(coverage[f"scaled_{level}"]["per_date"])
    x = np.arange(len(dates))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for offset, variant, color in [(-0.2, "symmetric", GREY), (0.2, "scaled", ORANGE)]:
        values = [coverage[f"{variant}_{level}"]["per_date"][d]["observed_coverage"] for d in dates]
        ax.bar(x + offset, values, width=0.4, color=color, label=f"{variant} intervals, pooled {coverage[f'{variant}_{level}']['pooled_observed_coverage']:.2f}")
    ax.axhline(level, color=RED, linestyle="--", linewidth=1, label="nominal 0.80")
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Observed coverage")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, figures, name)


def abstention_figure(intervals: pd.DataFrame, threshold: float, figures: Path, name: str, title: str) -> None:
    error = abs(intervals.prediction - intervals.target)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    retained = intervals.sigma <= threshold
    ax.scatter(intervals.sigma[retained], error[retained], s=10, color=BLUE, alpha=0.6, label=f"retained ({int(retained.sum())})")
    ax.scatter(intervals.sigma[~retained], error[~retained], s=10, color=ORANGE, alpha=0.6, label=f"unreliable flag ({int((~retained).sum())})")
    ax.axvline(threshold, color=RED, linestyle="--", linewidth=1, label=f"locked threshold {threshold:.2f}")
    ax.set_xlabel("Predicted difficulty sigma(x), day-7 inputs only")
    ax.set_ylabel("Absolute forecast error")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, figures, name)


def reserve_figure(res: dict, figures: Path) -> None:
    models = [res["product_model"], res["strongest_simple_comparator"], "persistence_day7", "control_reference"]
    dates = list(res["per_date_losses"][models[0]])
    x = np.arange(len(dates) + 1)
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, (m, color) in enumerate(zip(models, [ORANGE, BLUE, GREY, "#8C8C8C"])):
        values = [res["per_date_losses"][m][d] for d in dates] + [res["date_macro_mae"][m]]
        ax.bar(x + (k - 1.5) * width, values, width=width, color=color, label=label(m))
    ax.set_xticks(x)
    ax.set_xticklabels([f"reserve {d}" for d in dates] + ["mean of dates"], fontsize=9)
    ax.set_ylabel("Date-macro MAE")
    ax.set_title("Single reserve scoring: three later dates with chemically disjoint identities", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, figures, "gate_reserve_comparison")


def run(out: Path) -> dict:
    lock = gate.verify_lock(out)
    dev = read_json(out / "development_summary.json")
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    intervals = pd.read_csv(out / "development_intervals.csv", dtype={"identity": str})
    threshold = lock["locked_thresholds"]["abstention_sigma_p80"]
    model_comparison(dev, figures)
    warning_curves(dev, figures, "gate_warning_development", "Development: recall of later large changes against review budget")
    coverage_figure(dev, figures, "gate_coverage_development", "Development: observed coverage of nominal 80% intervals on each held-out date")
    abstention_figure(intervals, threshold, figures, "gate_abstention_development", "Development: predicted difficulty against realized error")
    compact = {
        "protocol": dev["protocol"],
        "product_model": dev["product_model"],
        "strongest_simple_comparator": dev["strongest_simple_comparator"],
        "development": {
            "date_macro_mae": dev["date_macro_mae"],
            "nested_selection_mae": dev["nested_selection"]["nested_date_macro_mae"],
            "optimism_estimate": dev["nested_selection"]["optimism_estimate"],
            "chosen_by_outer_fold": dev["nested_selection"]["chosen_by_outer_fold"],
            "comparisons_product_vs": dev["comparisons_product_vs"],
            "warning_primary": {p: next(r["pooled"] for r in dev["warning"][p] if r["subgroup"] == "early_quiet" and r["budget"] == 0.2) for p in dev["warning"]},
            "coverage_pooled": {k: v["pooled_observed_coverage"] for k, v in dev["coverage"].items()},
            "mean_width": {k: v["mean_width"] for k, v in dev["coverage"].items()},
            "abstention": dev["abstention"],
            "measurement_flags": dev["measurement_flags"],
            "outcome_quality_sensitivity": dev["outcome_quality_sensitivity"],
            "selection_v1_consistency": dev["selection_v1_consistency"],
            "locked_thresholds": lock["locked_thresholds"],
            "seconds": dev["seconds"],
        },
        "reserve_outcomes_scored": False,
    }
    if (out / "reserve_summary.json").exists():
        res = read_json(out / "reserve_summary.json")
        manifest = read_json(out / "reserve_manifest.json")
        for name, digest in manifest["file_sha256"].items():
            if data.digest(out / name) != digest:
                raise ValueError(f"Reserve output changed: {name}")
        reserve_intervals = pd.read_csv(out / "reserve_intervals.csv", dtype={"identity": str})
        warning_curves(res, figures, "gate_warning_reserve", "Reserve: recall of later large changes against review budget")
        coverage_figure(res, figures, "gate_coverage_reserve", "Reserve: observed coverage of nominal 80% intervals on each reserved date")
        abstention_figure(reserve_intervals, threshold, figures, "gate_abstention_reserve", "Reserve: predicted difficulty against realized error")
        reserve_figure(res, figures)
        compact["reserve_outcomes_scored"] = True
        compact["reserve"] = {
            "cases": res["reserve_cases"], "identities": res["reserve_identities"], "cases_per_date": res["cases_per_date"],
            "date_macro_mae": res["date_macro_mae"], "per_date_losses": {m: res["per_date_losses"][m] for m in [res["product_model"], res["strongest_simple_comparator"], "persistence_day7", "control_reference"]},
            "warning_primary": {p: next(r["pooled"] for r in res["warning"][p] if r["subgroup"] == "early_quiet" and r["budget"] == 0.2) for p in res["warning"]},
            "warning_all_budgets": {p: [{"subgroup": r["subgroup"], "budget": r["budget"], **r["pooled"]} for r in res["warning"][p]] for p in res["warning"]},
            "coverage_pooled": {k: v["pooled_observed_coverage"] for k, v in res["coverage"].items()},
            "coverage_per_date": {k: v["per_date"] for k, v in res["coverage"].items()},
            "abstention": res["abstention"], "measurement_flags": res["measurement_flags"], "outcome_quality_sensitivity": res["outcome_quality_sensitivity"],
            "criteria": res["criteria"], "conformal_quantiles": res["conformal_quantiles"],
        }
    compact["report_code_sha256"] = data.digest(Path(__file__))
    (out / "gate_report_summary.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (data.ROOT / "docs/neuroforecast_gate_summary.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=gate.DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "reserve"}, indent=2, default=str), flush=True)
