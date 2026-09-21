"""Recompute every headline number in the README and reports from the published per-case tables.

This reads only evaluation/*.csv and models/development_lock.json, never a summary file, so it is an
independent check that the prose matches the evidence. It needs no download and runs in a second.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_gate as gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = "extra_trees_relative_day7"
COMPARATOR = "extra_trees_activity_coordination"
def tolerance_for(published: float | int) -> float:
    """Compare at the precision the value is published to: 0.51 must match 51%, 0.93711 must match five decimals."""
    if isinstance(published, int):
        return 0.0
    text = repr(float(published))
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * 10 ** -decimals


# value as published, recomputed below
PUBLISHED = {
    "reserve product date-macro MAE": 0.7716,
    "reserve persistence date-macro MAE": 0.6088,
    "reserve comparator date-macro MAE": 0.8005,
    "reserve control-reference date-macro MAE": 0.9150,
    "reserve dates product beats persistence": 1,
    "reserve dates product beats comparator": 3,
    "reserve conditions": 224,
    "reserve chemicals": 32,
    "reserve declined conditions": 32,
    "reserve retained-case MAE": 0.446,
    "reserve all-case MAE": 0.784,
    "reserve abstained-case MAE": 2.813,
    "reserve share of error in declined conditions": 0.51,
    "reserve persistence retained-case MAE": 0.433,
    "reserve persistence all-case MAE": 0.595,
    "reserve observed coverage at nominal 0.80": 0.714,
    "reserve early-quiet positives": 10,
    "reserve early-quiet detected at 20% budget": 6,
    "reserve early-quiet detected by persistence ranking": 0,
    "reserve low-reference-activity flagged conditions": 84,
    "reserve flagged MAE": 1.057,
    "reserve unflagged MAE": 0.620,
    "development product date-macro MAE": 0.93711,
    "development persistence date-macro MAE": 1.10480,
    "development comparator date-macro MAE": 0.97339,
    "development conditions": 819,
    "development chemicals": 101,
    "development retained conditions": 655,
    "development retained-case MAE": 0.678,
    "development abstained-case MAE": 2.210,
    "development all-case MAE": 0.984,
    "development controls-disagree flagged conditions": 42,
    "development early-quiet positives": 45,
}


def early_quiet_recall(frame: pd.DataFrame, score: np.ndarray, budget: float = 0.2) -> tuple[int, int]:
    quiet = abs(frame.day7) < 0.5
    positive = abs(frame.target) >= 1
    work = pd.DataFrame({"date": frame.date, "score": score, "positive": positive, "quiet": quiet})
    record = next(r for r in gate.warning_metrics(work, [budget]) if r["subgroup"] == "early_quiet")
    return record["pooled"]["detected"], record["pooled"]["positives"]


def recompute() -> dict:
    found = {}
    threshold = json.loads((ROOT / "models/development_lock.json").read_text(encoding="utf-8"))["locked_thresholds"]["abstention_sigma_p80"]
    for stage in ["reserve", "development"]:
        predictions = pd.read_csv(ROOT / f"evaluation/{stage}_predictions.csv", dtype={"identity": str})
        intervals = pd.read_csv(ROOT / f"evaluation/{stage}_intervals.csv", dtype={"identity": str})
        losses = {m: gate.date_losses(g) for m, g in predictions.groupby("model")}
        found[f"{stage} product date-macro MAE"] = float(losses[PRODUCT].mean())
        found[f"{stage} persistence date-macro MAE"] = float(losses["persistence_day7"].mean())
        found[f"{stage} comparator date-macro MAE"] = float(losses[COMPARATOR].mean())
        found[f"{stage} conditions"] = int(intervals.case_id.nunique())
        found[f"{stage} chemicals"] = int(intervals.identity.nunique())

        error = abs(intervals.prediction - intervals.target)
        retained = intervals.sigma <= threshold
        found[f"{stage} all-case MAE"] = float(error.mean())
        found[f"{stage} retained-case MAE"] = float(error[retained].mean())
        found[f"{stage} abstained-case MAE"] = float(error[~retained].mean())
        detected, positives = early_quiet_recall(intervals, abs(intervals.prediction).to_numpy())
        found[f"{stage} early-quiet positives"] = positives

        if stage == "reserve":
            found["reserve control-reference date-macro MAE"] = float(losses["control_reference"].mean())
            found["reserve dates product beats persistence"] = int((losses[PRODUCT] < losses["persistence_day7"]).sum())
            found["reserve dates product beats comparator"] = int((losses[PRODUCT] < losses[COMPARATOR]).sum())
            found["reserve declined conditions"] = int((~retained).sum())
            found["reserve share of error in declined conditions"] = float(error[~retained].sum() / error.sum())
            persistence = predictions.loc[predictions.model == "persistence_day7"].set_index("case_id").reindex(intervals.case_id)
            simple_error = abs(persistence.prediction.to_numpy() - persistence.target.to_numpy())
            found["reserve persistence all-case MAE"] = float(simple_error.mean())
            found["reserve persistence retained-case MAE"] = float(simple_error[retained.to_numpy()].mean())
            inside = (intervals.target >= intervals["scaled_lower_0.8"]) & (intervals.target <= intervals["scaled_upper_0.8"])
            found["reserve observed coverage at nominal 0.80"] = float(inside.mean())
            found["reserve early-quiet detected at 20% budget"] = detected
            found["reserve early-quiet detected by persistence ranking"] = early_quiet_recall(intervals, abs(intervals.day7).to_numpy())[0]
            flagged = intervals.flag_low_reference_activity.to_numpy(dtype=bool)
            found["reserve low-reference-activity flagged conditions"] = int(flagged.sum())
            found["reserve flagged MAE"] = float(error[flagged].mean())
            found["reserve unflagged MAE"] = float(error[~flagged].mean())
        else:
            found["development retained conditions"] = int(retained.sum())
            flags = pd.read_csv(ROOT / "evaluation/development_flags.csv")
            found["development controls-disagree flagged conditions"] = int(flags.controls_disagree.sum())
    return found


if __name__ == "__main__":
    found = recompute()
    failures = []
    for label, published in PUBLISHED.items():
        actual = found[label]
        ok = abs(actual - published) <= tolerance_for(published)
        print(f"{'ok ' if ok else 'BAD'}  {label:58s} published {published:<10} recomputed {round(actual, 5)}")
        if not ok:
            failures.append(label)
    print()
    if failures:
        raise SystemExit(f"{len(failures)} published value(s) do not match the evidence: {failures}")
    print(f"All {len(PUBLISHED)} published values match the per-case evidence tables.")
