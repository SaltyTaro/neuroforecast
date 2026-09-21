"""Describe development control variability without changing the frozen benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import neuroforecast_data as data


def run(source: Path, raw_path: Path) -> dict:
    predictions = pd.read_csv(source / "representation_followup/predictions.csv", dtype={"identity": str})
    predictions = predictions.loc[predictions.model == "extra_trees_relative_day7"].copy()
    reserve = json.loads((source / "reserved_manifest.json").read_text(encoding="utf-8"))
    if set(predictions.date) & set(reserve["reserved_dates"]) or set(predictions.identity) & set(reserve["reserved_identities"]):
        raise ValueError("Reserve is outside this development-only audit")
    raw, _ = data.load_raw(raw_path)
    raw = raw.loc[raw.date.isin(predictions.date.unique())].copy()
    controls = raw.loc[raw.dose == 0].groupby(["date", "Plate.SN", "DIV"])["ns.n"].agg(["size", "count", "min", "median", "max"])
    controls.to_csv(source / "development_control_summary.csv")
    predictions["absolute_error"] = abs(predictions.prediction - predictions.target)
    predictions.nlargest(20, "absolute_error").to_csv(source / "largest_development_errors.csv", index=False)
    observed = raw.loc[raw.DIV == 12].copy()
    observed["case_id"] = [f"{d}|{i}|{c:.12g}" for d, i, c in observed[data.KEY].itertuples(index=False, name=None)]
    observed = observed.loc[observed.case_id.isin(predictions.case_id)]
    reference = controls["median"].rename("control_median")
    observed = observed.join(reference, on=["date", "Plate.SN", "DIV"], validate="many_to_one")
    impacted = set(observed.loc[observed.control_median == 0, "case_id"])
    affected = predictions.loc[predictions.case_id.isin(impacted)]
    worst_date = int(predictions.groupby(["date", "identity"]).absolute_error.mean().groupby("date").mean().idxmax())
    small_counts = []
    for model in ["extra_trees_relative_day7", "extra_trees_activity_coordination"]:
        frame = pd.read_csv(source / "representation_followup/predictions.csv")
        frame = frame.loc[(frame.model == model) & (abs(frame.day7) < 0.5)]
        alert = abs(frame.prediction) >= 1
        change = abs(frame.target) >= 1
        small_counts.append({"model": model, "detected": int((alert & change).sum()), "false_alerts": int((alert & ~change).sum())})
    result = {
        "scope": "Post-development failure description only; no fitting, exclusions, or metric changes",
        "reserve_outcomes_scored": False,
        "worst_date": worst_date,
        "worst_date_control_summary": controls.reset_index().loc[lambda x: (x.date == worst_date) & x.DIV.isin([7, 12])].to_dict("records"),
        "zero_day12_control_median_development_cases": len(affected),
        "zero_day12_control_median_cases_by_date": {str(k): int(v) for k, v in affected.groupby("date").size().items()},
        "zero_day12_control_median_later_large_changes": int((abs(affected.target) >= 1).sum()),
        "zero_day12_control_median_early_quiet_later_large_changes": int(((abs(affected.target) >= 1) & (abs(affected.day7) < 0.5)).sum()),
        "zero_day12_control_median_detected_early_quiet_changes": int(((abs(affected.target) >= 1) & (abs(affected.day7) < 0.5) & (abs(affected.prediction) >= 1)).sum()),
        "early_warning_comparison": small_counts,
        "interpretation": "A zero control median can inflate the pseudocount-regularized relative target. This association does not prove an assay failure or a chemical mechanism. Keep the original cases and results; evaluate control/replicate reliability prospectively in a separate protocol, using only available observations for any decision.",
        "code_sha256": data.digest(Path(__file__)),
        "source_prediction_sha256": data.digest(source / "representation_followup/predictions.csv"),
    }
    (source / "failure_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {key: value for key, value in result.items() if key != "worst_date_control_summary"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=data.DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=data.DEFAULT_RAW)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.raw), indent=2), flush=True)
