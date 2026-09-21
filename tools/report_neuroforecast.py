"""Rebuild selection figures and summaries from saved development predictions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import neuroforecast_benchmark as benchmark
import neuroforecast_data as data
import neuroforecast_representation as representation

SELECTED = "extra_trees_relative_day7"
SMALL = "extra_trees_activity_coordination"
LABELS = {
    SELECTED: "Control-relative, day 7 (selected)",
    SMALL: "Activity + coordination, days 5 + 7",
    "extra_trees_relative": "Control-relative, days 5 + 7",
    "extra_trees_all": "Original full-feature Extra Trees",
    "ridge_relative": "Control-relative Ridge",
    "control_reference": "Assume control-like day 12",
    "persistence_day7": "Carry day 7 forward",
    "extra_trees_target_only": "Target-only Extra Trees",
    "ridge_target_only": "Target-only Ridge",
    "ridge_all": "Original full-feature Ridge",
    "linear_trend": "Extrapolate day 5 to day 7 trend",
}
BLUE, ORANGE, GREY = "#176A9A", "#C67B25", "#BEC7CF"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def diagnostics(frame: pd.DataFrame) -> dict:
    actual = abs(frame.target.to_numpy()) >= 1
    scores = abs(frame.prediction.to_numpy())
    alert = scores >= 1
    tp, fp = int((actual & alert).sum()), int((~actual & alert).sum())
    fn, tn = int((actual & ~alert).sum()), int((~actual & ~alert).sum())
    return {
        "n": len(frame), "positive": int(actual.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "auc": float(roc_auc_score(actual, scores)),
        "ap": float(average_precision_score(actual, scores)),
    }


def save_figure(fig, root: Path, name: str) -> None:
    fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(root / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(source: Path) -> dict:
    initial = read_json(source / "summary.json")
    followup = read_json(source / "representation_followup/summary.json")
    prepared = read_json(source / "prepared_manifest.json")
    run_record = read_json(source / "run_manifest.json")
    if prepared["code_hashes"] != benchmark.code_hashes():
        raise ValueError("Frozen initial code or protocol changed")
    if followup["code_sha256"] != data.digest(Path(representation.__file__)):
        raise ValueError("Frozen follow-up code changed")
    if followup["protocol_sha256"] != data.digest(representation.PROTOCOL):
        raise ValueError("Frozen follow-up protocol changed")
    if data.digest(source / "prepared_manifest.json") != followup["source_prepared_manifest_sha256"]:
        raise ValueError("Follow-up source manifest changed")
    original_hash = data.digest(source / "predictions.csv")
    if original_hash != run_record["predictions_sha256"] or original_hash != followup["source_predictions_sha256"]:
        raise ValueError("Original predictions changed")
    predictions = pd.concat([
        pd.read_csv(source / "predictions.csv", dtype={"identity": str}),
        pd.read_csv(source / "representation_followup/predictions.csv", dtype={"identity": str}),
    ], ignore_index=True)
    if predictions.duplicated(["model", "case_id"]).any():
        raise ValueError("Duplicate model/case prediction")
    reserve = read_json(source / "reserved_manifest.json")
    if set(predictions.date) & set(reserve["reserved_dates"]) or set(predictions.identity) & set(reserve["reserved_identities"]):
        raise ValueError("Reserve entered the development report")
    reference = predictions.loc[predictions.model == SELECTED].set_index("case_id").sort_index()
    for _, model_rows in predictions.groupby("model"):
        aligned = model_rows.set_index("case_id").sort_index()
        pd.testing.assert_frame_equal(aligned[["target", "day7", "date", "identity"]], reference[["target", "day7", "date", "identity"]])
    predictions["absolute_error"] = abs(predictions.prediction - predictions.target)
    chemicals = predictions.groupby(["model", "date", "identity"], as_index=False).absolute_error.mean()
    dates = chemicals.groupby(["model", "date"], as_index=False).absolute_error.mean()
    wide = dates.pivot(index="date", columns="model", values="absolute_error").sort_index()
    metrics = wide.mean().sort_values()
    for model, value in followup["metrics"].items():
        np.testing.assert_allclose(metrics[model], value, atol=1e-12, rtol=0)
    rng = np.random.default_rng(data.protocol()["seed"])
    indices = rng.integers(len(wide), size=(10000, len(wide)))
    selected_loss = wide[SELECTED].to_numpy()
    comparisons = []
    for model in ["control_reference", "persistence_day7", SMALL, "extra_trees_relative", "extra_trees_all"]:
        loss = wide[model].to_numpy()
        gain = 1 - selected_loss[indices].mean(axis=1) / loss[indices].mean(axis=1)
        comparisons.append({
            "comparator": model,
            "gain": float(1 - selected_loss.mean() / loss.mean()),
            "descriptive_interval": np.quantile(gain, [0.025, 0.975]).tolist(),
            "dates_improved": int((selected_loss < loss).sum()),
        })
    alerts = []
    for model, frame in predictions.groupby("model"):
        for subset, selected_frame in [("all", frame), ("early_quiet", frame.loc[abs(frame.day7) < 0.5])]:
            alerts.append({"model": model, "subset": subset, **diagnostics(selected_frame)})
    audit = read_json(source / "data_audit.json")
    summary = {
        "status": "working topic selected after development comparisons; independent validation pending",
        "topic": "NeuroForecast: early warning for later changes in neuronal function",
        "selected_model": SELECTED,
        "primary_mae": float(metrics[SELECTED]),
        "metrics": metrics.to_dict(),
        "comparisons": comparisons,
        "diagnostic_classifications": alerts,
        "cohort": {key: audit[key] for key in ["physical_recordings", "physical_wells", "identified_substances", "eligible_substances", "development_cases", "development_substances", "development_dates"]},
        "initial_primary_result": {key: initial[key] for key in ["primary_model", "relative_mae_reduction", "paired_date_descriptive_interval_95"]},
        "followup_primary_model": followup["primary_model"],
        "selected_model_was_a_followup_ablation": True,
        "reserve_outcomes_scored": False,
        "reserve_dates": reserve["reserved_dates"],
        "uncertainty_note": "Paired bootstrap of 12 development date losses; descriptive, after model selection. These are not independent-study or clinical intervals.",
        "initial_run_seconds": run_record["seconds"],
        "followup_run_seconds": followup["seconds"],
        "source_hashes": {name: data.digest(source / name) for name in ["prepared_manifest.json", "predictions.csv", "representation_followup/predictions.csv", "audit.json", "demo/manifest.json"]},
        "report_code_sha256": data.digest(Path(__file__)),
    }
    # The original run has an extra contemporaneous selection memo. A fresh
    # reproduction does not need that memo to compute the same report.
    if (source / "candidate_summary.json").exists():
        previous = read_json(source / "candidate_summary.json")
        np.testing.assert_allclose(previous["primary_mae"], summary["primary_mae"], atol=1e-12, rtol=0)
        for old, new in zip(previous["comparisons"], comparisons, strict=True):
            if old["comparator"] != new["comparator"] or old["dates_improved"] != new["dates_improved"]:
                raise ValueError("Selected-candidate comparison differs")
            np.testing.assert_allclose(old["descriptive_interval"], new["descriptive_interval"], atol=1e-12, rtol=0)
    out = source / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    colors = [BLUE if name == SELECTED else ORANGE if name == SMALL else GREY for name in metrics.index]
    ax.barh([LABELS[name] for name in metrics.index], metrics.to_numpy(), color=colors, height=0.68)
    ax.invert_yaxis()
    for i, value in enumerate(metrics):
        ax.text(value + 0.025, i, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, metrics.max() * 1.13)
    ax.set_xlabel("Mean absolute error of day-12 log2 relative network-spike readout")
    ax.set_title("Development comparison: lower error is better", loc="left", fontsize=14, pad=15)
    ax.xaxis.grid(True, alpha=0.18)
    ax.set_axisbelow(True)
    fig.text(0.02, 0.015, "101 substances; 819 dose-condition cases; 12 equally weighted held-out dates. Selection results, not final validation.", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_figure(fig, out, "model_comparison")

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    yy = np.arange(len(wide))
    for model, offset, color, label in [(SELECTED, -0.19, BLUE, "Selected day-7 model"), (SMALL, 0, ORANGE, "Activity + coordination baseline"), ("persistence_day7", 0.19, "#7F8B94", "Carry day 7 forward")]:
        ax.scatter(wide[model], yy + offset, label=label, color=color, s=35, zorder=3)
    ax.set_yticks(yy, [str(day) for day in wide.index])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Per-date MAE (log2 relative readout); lower is better")
    ax.set_ylabel("Held-out experiment date")
    ax.set_title("Performance varies substantially across batches", loc="left", fontsize=14, pad=15)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    fig.text(0.02, 0.015, "All test chemical identities are also removed from training, including repeats on other dates.", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_figure(fig, out, "batch_comparison")

    models = [SELECTED, SMALL, "extra_trees_all", "persistence_day7"]
    diagnostic = {row["model"]: row for row in alerts if row["subset"] == "early_quiet"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, gridspec_kw={"width_ratios": [1.3, 1]})
    yy = np.arange(len(models))
    tp = np.array([diagnostic[name]["tp"] for name in models])
    fn = np.array([diagnostic[name]["fn"] for name in models])
    fp = np.array([diagnostic[name]["fp"] for name in models])
    axes[0].barh(yy, tp, color=BLUE, label="Detected", height=0.6)
    axes[0].barh(yy, fn, left=tp, color="#E4E8EB", label="Missed", height=0.6)
    axes[1].barh(yy, fp, color=ORANGE, height=0.6)
    for i in yy:
        axes[0].text(tp[i] + 0.6, i, f"{tp[i]}/45", va="center", fontsize=9)
        axes[1].text(fp[i] + 0.8, i, str(fp[i]), va="center", fontsize=9)
    axes[0].set_yticks(yy, ["Selected day-7 model", "Activity + coordination", "Original full-feature model", "Carry day 7 forward"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 49)
    axes[1].set_xlim(0, max(fp) * 1.18)
    axes[0].set_title("Later changes detected (45 cases)", loc="left", fontsize=11)
    axes[1].set_title("False alerts (382 below-threshold cases)", loc="left", fontsize=11)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xlabel("Dose-condition cases")
        ax.xaxis.grid(True, alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("Early warning among 427 cases with little day-7 change", fontsize=14, x=0.02, ha="left")
    fig.text(0.02, 0.01, "Exploratory counts, correlated within chemicals/dates. Early: |day 7| < 0.5; later change and alert: |log2 readout| >= 1.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    save_figure(fig, out, "early_warning")

    examples = read_json(source / "demo/manifest.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    example_frames = [reference.loc[reference.date == record["held_out_date"]] for record in examples["examples"]]
    lower = min(min(frame.target.min(), frame.prediction.min()) for frame in example_frames) - 0.5
    upper = max(max(frame.target.max(), frame.prediction.max()) for frame in example_frames) + 0.5
    for ax, record, frame in zip(axes, examples["examples"], example_frames, strict=True):
        ax.plot([lower, upper], [lower, upper], color="#7F8B94", linestyle="--", linewidth=1)
        ax.scatter(frame.target, frame.prediction, color=BLUE, s=24, alpha=0.68, edgecolors="white", linewidths=0.3)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        loss = wide.loc[record["held_out_date"], SELECTED]
        title = "Upper-median batch" if record["example"] == "representative" else "Largest-error batch"
        ax.set_title(f"{title}: {record['held_out_date']}\nMAE {loss:.3f}; {len(frame)} cases", fontsize=11, loc="left")
        ax.set_xlabel("Observed day-12 log2 relative readout")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Forecast from day 7")
    fig.suptitle("Executable examples include the weakest batch", x=0.02, ha="left", fontsize=14)
    fig.text(0.02, 0.015, "Models exclude each example's date and chemical identities. Dashed line indicates a perfect prediction.", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    save_figure(fig, out, "demo_examples")

    benchmark.write_json(source / "report_summary.json", summary)
    return {"summary": str(source / "report_summary.json"), "figures": str(out), "models": len(metrics), "cases": len(reference), "reserve_outcomes_scored": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=data.DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.source), indent=2), flush=True)
