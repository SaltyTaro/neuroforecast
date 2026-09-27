"""Post-hoc analyses of the locked v2 tool: what drives the forecast, where it beats carry-forward, and how the
batch rule and the intervals behave. None of this was pre-registered, and none of it changes a sealed result.

Part 1 uses the 17 development batches only, with the same purged leave-one-batch-out folds as development:
linear models on the product's 13 inputs, and permutation importance of those inputs for the product model.
Part 2 describes predictions already scored once on the external batches; no model is refit or selected there.
It breaks errors down by the day-7 state, compares the trust verdict with declining the most extreme forecasts
at the same count, applies the batch reference rule to every batch, and recalibrates the interval width on the
first external batches in time order, as a lab would on its own first runs.

Runs from the repository alone (evaluation/ tables), in about a minute.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402

SEED = 20260928
REPEATS = 5
LEVEL = 0.8
CALIBRATION_BATCHES = [4, 8, 12]
# Day-7 states, in log2 units of the plate-normalized network-spike count.
STATES = ["strongly suppressed (day 7 <= -1)", "suppressed (-1 < day 7 <= -0.5)", "near control (|day 7| < 0.5)",
          "raised (0.5 <= day 7 < 1)", "hyperactive (day 7 >= 1)"]
OUT = ROOT / "evaluation/v2_posthoc"


def state_of(day7: pd.Series) -> np.ndarray:
    return np.select([day7 <= -1, day7 <= -0.5, day7 < 0.5, day7 < 1], STATES[:4], STATES[4])


def by_state(frame: pd.DataFrame, prediction: str) -> list[dict]:
    rows = []
    states = state_of(frame.day7)
    for label in STATES:
        part = frame.loc[states == label]
        rows.append({"state": label, "conditions": int(len(part)),
                     "forecast_mae": float(abs(part[prediction] - part.target).mean()),
                     "persistence_mae": float(abs(part.day7 - part.target).mean()),
                     "mean_observed_change": float((part.target - part.day7).mean()),
                     "mean_forecast_change": float((part[prediction] - part.day7).mean())})
    return rows


def development() -> tuple[dict, pd.DataFrame]:
    cfg = v2.v2_config()
    features = pd.read_csv(ROOT / "evaluation/v2_prepared/dev_features.csv")
    cases = pd.read_csv(ROOT / "evaluation/v2_prepared/dev_cases.csv", dtype={"identity": str, "spid": str})
    columns = v2.column_sets(features)["small"]
    groups = {c: [c] for c in columns}
    groups["all day-5 inputs"] = [c for c in columns if c.endswith("d5")]
    groups["all day-7 inputs"] = [c for c in columns if c.endswith("d7")]
    y = cases.target.to_numpy()
    usable = cases.usable.to_numpy(dtype=bool)
    oof = {name: np.full(len(cases), np.nan) for name in ["small_refit", "ols", "lad"]}
    permuted = {(g, r): np.full(len(cases), np.nan) for g in groups for r in range(REPEATS)}
    rng = np.random.default_rng(SEED)
    with threadpool_limits(limits=1):
        for _, train, test in v2.purged_folds(cases, np.arange(len(cases))):
            fit_idx = train[usable[train]]
            x_fit, x_test = features.iloc[fit_idx][columns], features.iloc[test][columns]
            model = v2.fit(features.iloc[fit_idx], columns, y[fit_idx], cfg)
            oof["small_refit"][test] = model.predict(x_test)
            oof["ols"][test] = LinearRegression().fit(x_fit, y[fit_idx]).predict(x_test)
            oof["lad"][test] = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(x_fit, y[fit_idx]).predict(x_test)
            for group, members in groups.items():
                for r in range(REPEATS):
                    shuffled = x_test.copy()
                    shuffled[members] = x_test[members].to_numpy()[rng.permutation(len(x_test))]
                    permuted[(group, r)][test] = model.predict(shuffled)
    saved = pd.read_csv(ROOT / "evaluation/v2_development/dev_oof_predictions.csv", dtype={"identity": str})
    replay = float(np.max(np.abs(saved.pred__small_refit.to_numpy() - oof["small_refit"])))
    if replay > 1e-12:
        raise ValueError(f"Product replay differs from the saved development predictions by {replay}")
    base = v2.macro(cases, oof["small_refit"], usable)
    importance = []
    for group in groups:
        scores = [v2.macro(cases, permuted[(group, r)], usable) for r in range(REPEATS)]
        importance.append({"input": group, "mae_increase": float(np.mean(scores) - base), "sd_over_repeats": float(np.std(scores, ddof=1))})
    table = cases[["date", "identity", "dose", "case_id", "target", "usable", "day7"]].assign(
        small_refit=oof["small_refit"], ols=oof["ols"], lad=oof["lad"])
    scored = table.loc[usable]
    result = {
        "folds": "17 development batches, purged leave-one-batch-out (as in development)",
        "product_replay_max_abs_difference": replay,
        "date_macro_mae": {"small_refit": base, "ols": v2.macro(cases, oof["ols"], usable), "lad": v2.macro(cases, oof["lad"], usable),
                           "persistence": v2.macro(cases, cases.day7.to_numpy(), usable)},
        "permutation_importance": sorted(importance, key=lambda r: -r["mae_increase"]),
        "permutation_repeats": REPEATS, "seed": SEED,
        "by_day7_state": by_state(scored, "small_refit"),
    }
    return result, table


def batch_rule(dev_table: pd.DataFrame) -> list[dict]:
    """The shipped display rule on every batch: half or more conditions below 1 day-7 control network spike."""
    rows = []
    for prefix, cases_file, scored in [("development", "dev_cases.csv", None), ("external", "ext_cases.csv", "arm_b")]:
        features = pd.read_csv(ROOT / f"evaluation/v2_prepared/{prefix[:3]}_features.csv")
        cases = pd.read_csv(ROOT / f"evaluation/v2_prepared/{cases_file}", dtype={"identity": str, "spid": str})
        cases["low_reference"] = np.sinh(features["rel__plate_ctrl_median_ns_d7"].to_numpy()) < 1
        if scored is None:
            outcome = dev_table.loc[dev_table.usable.astype(bool), ["case_id", "date", "identity", "target", "day7", "small_refit"]]
            outcome = outcome.rename(columns={"small_refit": "prediction"})
        else:
            arm = pd.read_csv(ROOT / "evaluation/external/arm_b_cases.csv", dtype={"identity": str})
            outcome = arm.loc[arm.primary.astype(bool) & arm.usable.astype(bool), ["case_id", "date", "identity", "target", "day7", "prediction"]]
        for date, group in cases.groupby("date"):
            fraction = float(group.low_reference.mean())
            part = outcome.loc[outcome.date == date]
            rows.append({"set": prefix, "batch": int(date), "conditions": int(len(group)), "low_reference_fraction": fraction,
                         "rule_fires": bool(fraction >= 0.5), "scored_conditions": int(len(part)),
                         "forecast_mae": float(gate.macro_mae(part)) if len(part) else None,
                         "persistence_mae": float(gate.macro_mae(part.assign(prediction=part.day7))) if len(part) else None})
    return rows


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    ordered = np.sort(scores)
    return float(ordered[min(math.ceil((len(ordered) + 1) * level), len(ordered)) - 1])


def external() -> dict:
    lock = json.loads((ROOT / "evaluation/external/results.json").read_text(encoding="utf-8"))["lock"]
    locked_q = lock["conformal_quantiles"][str(LEVEL)]
    arm = pd.read_csv(ROOT / "evaluation/external/arm_b_cases.csv", dtype={"identity": str})
    arm = arm.loc[arm.primary.astype(bool) & arm.usable.astype(bool)].copy()
    error = abs(arm.prediction - arm.target)
    declined = arm.declined.astype(bool).to_numpy()
    count = int(declined.sum())
    extreme = np.zeros(len(arm), dtype=bool)
    extreme[np.argsort(-np.abs(arm.prediction.to_numpy()), kind="stable")[:count]] = True
    comparison = {"declined": count, "kept_mae_trust_verdict": float(error[~declined].mean()),
                  "kept_mae_decline_most_extreme_forecasts": float(error[~extreme].mean()),
                  "declined_both_ways": int((declined & extreme).sum()),
                  "declined_max_day7": float(arm.day7[declined].max()),
                  "declined_share_day7_at_or_below_minus1": float((arm.day7[declined] <= -1).mean())}
    arm["sigma"] = (arm[f"high_{LEVEL}"] - arm[f"low_{LEVEL}"]) / (2 * locked_q)
    arm["score"] = error / arm.sigma
    dates = sorted(arm.date.unique())
    recalibration = []
    for k in CALIBRATION_BATCHES:
        calibrate, rest = arm.loc[arm.date.isin(dates[:k])], arm.loc[~arm.date.isin(dates[:k])]
        q = conformal_quantile(calibrate.score.to_numpy(), LEVEL)
        recalibration.append({"calibration_batches": k, "calibration_conditions": int(len(calibrate)), "remaining_batches": len(dates) - k,
                              "remaining_conditions": int(len(rest)), "recalibrated_quantile": q,
                              "coverage_locked": float((rest.score <= locked_q).mean()), "coverage_recalibrated": float((rest.score <= q).mean()),
                              "mean_width_locked": float((2 * locked_q * rest.sigma).mean()), "mean_width_recalibrated": float((2 * q * rest.sigma).mean())})
    return {"by_day7_state": by_state(arm, "prediction"), "trust_vs_extremity_same_count": comparison,
            "recalibration_nominal": LEVEL, "locked_quantile": locked_q, "recalibration": recalibration}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    dev, table = development()
    result = {"note": __doc__.strip().splitlines()[0], "development": dev, "external": external(), "batch_rule": batch_rule(table)}
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "dev_linear_oof.csv", index=False, lineterminator="\n")
    (args.output / "signal_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: dev[k] for k in ["date_macro_mae", "permutation_importance"]}, indent=1))
    print(pd.DataFrame(dev["by_day7_state"]).round(3).to_string(index=False))
    print(pd.DataFrame(result["external"]["by_day7_state"]).round(3).to_string(index=False))
    print(json.dumps(result["external"]["trust_vs_extremity_same_count"], indent=1))
    print(pd.DataFrame(result["external"]["recalibration"]).round(3).to_string(index=False))
    rules = pd.DataFrame(result["batch_rule"])
    print(rules.loc[rules.rule_fires | (rules.low_reference_fraction > 0)].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
