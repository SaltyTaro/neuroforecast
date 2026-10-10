"""Recompute every headline number in the README and reports from the published per-case tables.

This reads only per-case evidence (evaluation/**.csv) and models/development_lock.json, never a summary file, so it is an
independent check that the prose matches the evidence. It needs no download and runs in a second.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_gate as gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = "extra_trees_relative_day7"
COMPARATOR = "extra_trees_activity_coordination"
def written_decimals() -> dict[str, int]:
    """Decimals of each value as written in this file, so a trailing zero counts: 1.60 must round to 1.60, not 1.6."""
    source = Path(__file__).read_text(encoding="utf-8")
    return {m.group(1): len(m.group(2).split(".")[1]) for m in re.finditer(r'^\s+"([^"]+)":\s*(-?\d+\.\d+),?\s*(?:#.*)?$', source, re.M)}


def tolerance_for(published: float | int, decimals: int | None = None) -> float:
    """Compare at the precision the value is published to: 0.51 must match 51%, 0.93711 must match five decimals."""
    if isinstance(published, int):
        return 0.0
    if decimals is None:
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
    # Corrections of September 27, 2026: the clean-target explanation of the reserve result and the
    # post hoc trivial baselines. All are descriptive; none changes a locked criterion.
    "development usable-reference conditions": 777,
    "development usable-reference product date-macro MAE": 0.7635,
    "development usable-reference persistence date-macro MAE": 1.0571,
    "reserve 20170920 day-7/day-12 correlation": 0.97,
    "reserve 20171011 day-7/day-12 correlation": 0.92,
    "reserve 20171004 day-7/day-12 correlation": 0.12,
    "development lowest per-date day-7/day-12 correlation": 0.23,
    "development highest per-date day-7/day-12 correlation": 0.88,
    "reserve 20170920 day-12 slope on day 7": 1.31,
    "reserve 20171011 day-12 slope on day 7": 1.27,
    "reserve persistent-batch lowest forecast slope": 0.87,
    "reserve persistent-batch highest forecast slope": 0.94,
    "reserve unflagged product date-macro MAE": 0.629,
    "reserve unflagged persistence date-macro MAE": 0.573,
    "reserve 20171004 conditions with day-7 value exactly zero": 69,
    "reserve 20171004 early-quiet conditions": 76,
    "reserve early-quiet expected detections by persistence, random ties": 1.3,
    "reserve early-quiet expected detections by highest dose": 3.6,
    "development early-quiet expected detections by highest dose": 16.7,
    "development early-quiet detected at 20% budget": 22,
    "reserve early-quiet detections with the wrong direction": 2,
    "reserve retained-case MAE declining the most extreme forecasts": 0.456,
    "development retained-case MAE declining the most extreme forecasts": 0.623,
    "reserve abstention error reduction, product": 0.43,
    "reserve abstention error reduction, persistence": 0.27,
}


EXTERNAL = {
    # The sealed external test (scored once, September 27, 2026); primary cohort unless stated.
    "external primary conditions": 927,
    "external primary dates": 24,
    "external primary unseen chemicals": 105,
    "external secondary conditions": 56,
    "external v2 date-macro MAE": 0.735,
    "external persistence date-macro MAE": 0.980,
    "external frozen v1 date-macro MAE": 0.941,
    "external control-reference date-macro MAE": 1.403,
    "external v1 refit date-macro MAE": 0.706,
    "external v2 dates better than persistence": 19,
    "external frozen v1 dates better than persistence": 14,
    "external v2 minus persistence interval low": -0.396,
    "external v2 minus persistence interval high": -0.123,
    "external frozen v1 minus persistence interval low": -0.130,
    "external frozen v1 minus persistence interval high": 0.052,
    "external v2 relative error reduction vs persistence": 0.25,
    "external v2 declined": 135,
    "external v2 kept MAE": 0.541,
    "external v2 all MAE": 0.742,
    "external v2 kept reduction": 0.27,
    "external persistence kept MAE (v2 verdict)": 0.655,
    "external persistence all MAE": 0.932,
    "external persistence kept reduction (v2 verdict)": 0.30,
    "external v1 declined": 120,
    "external v1 kept MAE": 0.640,
    "external v1 all MAE": 0.907,
    "external v1 kept reduction": 0.29,
    "external persistence kept MAE (v1 verdict)": 0.717,
    "external persistence kept reduction (v1 verdict)": 0.23,
    "external v1 extremity rule kept MAE": 0.641,
    "external v2 risk-coverage area, difficulty": 0.542,
    "external v2 risk-coverage area, extremity": 0.562,
    "external early-quiet positives": 65,
    "external early-quiet detected, v1": 16.0,
    "external early-quiet detected, v2 locked score": 14.0,
    "external early-quiet detected, highest dose": 16.6,
    "external early-quiet detected, day-7 magnitude": 21.0,
    "external early-quiet detected, chance": 14.8,
    "external v1 coverage at nominal 0.80": 0.666,
    "external v2 coverage at nominal 0.80": 0.702,
    "external v2 coverage at nominal 0.90": 0.812,
    "external decision samples": 106,
    "external decision samples active": 72,
    "external decision EPA day-7 call agreement": 0.877,
    "external decision early share": 0.953,
    "external decision early agreement": 0.891,
    "external decision early agreement, secondary reference": 0.960,
    "external pooled day-7/day-12 correlation": 0.84,
    "external large-change share": 0.275,
    "reserve large-change share": 0.138,
    "v2 development pooled day-7/day-12 correlation": 0.66,
    "v2 development product date-macro MAE": 0.726,
    "v2 development persistence date-macro MAE": 1.064,
    "v2 development nested selection estimate": 0.770,
    "v2 development conditions": 1260,
    # Figures quoted in the rewritten technical report (September 27, 2026).
    "external v2 declined MAE": 1.921,
    "external v2 declined-to-kept error ratio": 3.5,
    "external v2 share of error in declined conditions": 0.38,
    "external v2 kept median absolute error": 0.41,
    "external v2 kept median fold error": 1.33,
    "external v2 share of conditions kept": 0.85,
    "external v2 fewest declined in a batch": 1,
    "external v2 most declined in a batch": 12,
    "external v2 mean declined per batch": 5.6,
    "external v2 batches where persistence won": 5,
    "external secondary v2 date-macro MAE": 0.724,
    "external secondary persistence date-macro MAE": 0.978,
    "external 20160120 v2 date-macro MAE": 0.286,
    "external 20160120 persistence date-macro MAE": 0.622,
    "v2 development v1 refit date-macro MAE": 0.745,
    "v2 development anchored residual date-macro MAE": 0.731,
    "v2 development batch blend date-macro MAE": 0.742,
    "v2 development anchored residual with batch summaries date-macro MAE": 0.810,
    "v2 development control-reference date-macro MAE": 1.385,
    "v2 development nested choices of the small model": 6,
    "v2 development nested choices of the blend": 9,
    "v2 development risk-coverage area, difficulty": 0.504,
    "v2 development risk-coverage area, extremity": 0.513,
    # Post hoc control-subsampling sensitivity (evaluation/v2_posthoc/control_subsampling.json).
    "subsampling 1 control v2 MAE": 0.733,
    "subsampling 1 control persistence MAE": 1.118,
    "subsampling 1 control declined share": 0.18,
    "subsampling 1 control false batch verdicts": 1.2,
    "subsampling 2 controls v2 MAE": 0.738,
    "subsampling 2 controls persistence MAE": 0.985,
    "subsampling 2 controls declined share": 0.15,
    "subsampling 2 controls false batch verdicts": 0,
    "subsampling 4 controls v2 MAE": 0.733,
    "subsampling 4 controls persistence MAE": 0.977,
    "subsampling 4 controls declined share": 0.12,
    "subsampling 4 controls false batch verdicts": 0,
}

SIGNAL = {
    # Post hoc analyses of September 28, 2026 (tools/posthoc_signal_analysis.py; evaluation/v2_posthoc).
    # Development: the same purged leave-one-batch-out folds; external: predictions already scored once.
    "posthoc development least-squares linear date-macro MAE": 0.949,
    "posthoc development median-regression linear date-macro MAE": 0.914,
    "posthoc importance, day-7 active electrodes": 0.23,
    "posthoc importance, day-7 network spikes": 0.14,
    "posthoc importance, day-7 firing rate": 0.08,
    "posthoc importance, all day-7 inputs": 0.88,
    "posthoc importance, all day-5 inputs": -0.003,
    "posthoc importance, dose": 0.000,
    "posthoc external strongly suppressed conditions": 207,
    "posthoc external strongly suppressed v2 MAE": 1.59,
    "posthoc external strongly suppressed persistence MAE": 2.27,
    "posthoc external suppressed conditions": 90,
    "posthoc external suppressed v2 MAE": 0.65,
    "posthoc external suppressed persistence MAE": 0.54,
    "posthoc external near-control conditions": 570,
    "posthoc external near-control v2 MAE": 0.476,
    "posthoc external near-control persistence MAE": 0.485,
    "posthoc external raised conditions": 47,
    "posthoc external raised v2 MAE": 0.44,
    "posthoc external raised persistence MAE": 0.87,
    "posthoc external hyperactive conditions": 13,
    "posthoc external hyperactive v2 MAE": 0.57,
    "posthoc external hyperactive persistence MAE": 2.11,
    "posthoc external raised or hyperactive conditions": 60,
    "posthoc external raised or hyperactive v2 MAE": 0.47,
    "posthoc external raised or hyperactive persistence MAE": 1.14,
    "posthoc external hyperactive mean observed change": -2.1,
    "posthoc external hyperactive mean forecast change": -1.9,
    "posthoc development strongly suppressed v2 MAE": 1.39,
    "posthoc development strongly suppressed persistence MAE": 2.15,
    "posthoc development near-control v2 MAE": 0.52,
    "posthoc development near-control persistence MAE": 0.58,
    "posthoc development raised or hyperactive conditions": 76,
    "posthoc development raised or hyperactive v2 MAE": 0.40,
    "posthoc development raised or hyperactive persistence MAE": 0.87,
    "posthoc external v2 kept MAE, trust verdict": 0.5413,
    "posthoc external v2 kept MAE, declining the most extreme forecasts": 0.5405,
    "posthoc external conditions declined by both rules": 102,
    "posthoc external highest day-7 value among declined": -1.30,
    "external secondary risk-coverage area, difficulty": 0.668,
    "external secondary risk-coverage area, extremity": 0.634,
    "external secondary v2 coverage at nominal 0.80": 0.607,
    "posthoc batches where the batch rule fires": 3,
    "posthoc batches checked by the batch rule": 41,
    "posthoc batch rule 20160907 v2 MAE": 0.88,
    "posthoc batch rule 20160907 persistence MAE": 1.85,
    "posthoc batch rule 20170628 v2 MAE": 0.53,
    "posthoc batch rule 20170628 persistence MAE": 1.99,
    "posthoc batch rule 20171004 v2 MAE": 1.10,
    "posthoc batch rule 20171004 persistence MAE": 0.68,
    "posthoc recalibrated on 4 batches, calibration conditions": 103,
    "posthoc recalibrated on 4 batches, locked coverage on the other 20": 0.711,
    "posthoc recalibrated on 4 batches, recalibrated coverage on the other 20": 0.876,
    "posthoc recalibrated on 4 batches, locked mean width": 1.87,
    "posthoc recalibrated on 4 batches, recalibrated mean width": 3.04,
    "posthoc recalibrated on 12 batches, locked coverage on the other 12": 0.855,
    "posthoc recalibrated on 12 batches, recalibrated coverage on the other 12": 0.975,
    "reserve pooled day-7/day-12 correlation": 0.87,
    "external per-batch v2 advantage vs batch persistence correlation": -0.27,
    "browser replay external conditions": 983,
    "browser replay reserve-batch conditions": 245,
}


V3 = {
    # The model study of October 7-8, 2026 (technical report section 9; evaluation/v3). Development only:
    # both sealed sets were spent before it. Outer folds of the locked nested, chemical-purged protocol.
    "v3 usable conditions": 2243,
    "v3 batches with usable conditions": 41,
    "v3 chemicals": 243,
    "v3 plates": 169,
    "v3 wells": 7900,
    "v3 v2-recipe date-macro MAE": 0.690,
    "v3 v2+PCT ensemble date-macro MAE": 0.668,
    "v3 dose-series Extra Trees date-macro MAE": 0.679,
    "v3 nested date-macro MAE": 0.695,
    "v3 PCT date-macro MAE": 0.711,
    "v3 LightGBM (all features) date-macro MAE": 0.718,
    "v3 TabPFN v2 (all features) date-macro MAE": 0.723,
    "v3 MLP date-macro MAE": 0.862,
    "v3 ridge date-macro MAE": 0.880,
    "v3 carry-forward date-macro MAE": 1.016,
    "v3 batch-blind PCT date-macro MAE": 0.742,
    "v3 batches ensemble beats v2": 30,
    "v3 batches nested beats v2": 25,
    "v3 batches PCT beats v2": 21,
    "v3 batches dose-series Extra Trees beats v2": 22,
    "v3 batches LightGBM beats v2": 21,
    "v3 batches TabPFN beats v2": 18,
    "v3 batches MLP beats v2": 9,
    "v3 batches ridge beats v2": 9,
    "v3 batches carry-forward beats v2": 4,
    "v3 nested minus v2, interval low": -0.038,
    "v3 nested minus v2, interval high": 0.050,
    "v3 ensemble minus v2, interval low": -0.060,
    "v3 ensemble minus v2, interval high": 0.020,
    "v3 external-cohort ensemble date-macro MAE": 0.614,
    "v3 external-cohort v2 date-macro MAE": 0.663,
    "v3 external-cohort ensemble minus v2, interval low": -0.092,
    "v3 external-cohort ensemble minus v2, interval high": -0.009,
    "v3 external-cohort batches ensemble beats v2": 20,
    "v3 batch-blind minus PCT, interval low": -0.012,
    "v3 batch-blind minus PCT, interval high": 0.079,
    "v3 batches where the batch-blind PCT is worse": 26,
    "v3 twin day-12 network spikes, PCT": 0.704,
    "v3 twin day-12 network spikes, carry-forward": 1.016,
    "v3 twin day-12 mutual information, PCT": 0.137,
    "v3 twin day-12 mutual information, carry-forward": 0.282,
    "v3 twin targets where PCT beats carry-forward": 36,
    "v3 twin targets where PCT beats per-target Extra Trees": 16,
    "v3 twin mean error ratio, PCT to per-target Extra Trees": 1.02,
    "v31 batch-weighted batch-mean coverage at 0.80": 0.81,
    "v31 batch-weighted batch-mean coverage at 0.90": 0.90,
    "v31 batch-weighted worst-fifth coverage at 0.80": 0.58,
    "v31 two-level worst-fifth coverage at 0.80": 0.68,
    "v31 two-level share of batches within 0.1 of nominal at 0.80": 0.90,
    "v31 two-level width increase over batch-weighted at 0.80": 0.25,
    "v31 batch-adaptive worst-fifth gain at 0.80": 0.026,
    "v31 rank correlation of predicted and realised batch difficulty": -0.07,
}


def v3_paired(a: pd.Series, b: pd.Series, seed: int = 20261007, n: int = 10000) -> tuple[float, float, int]:
    """Paired bootstrap over batches, exactly as neuroforecast_v3_analyze.paired_dates: (low, high, batches a better)."""
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy()
    boot = d[np.random.default_rng(seed).integers(len(d), size=(n, len(d)))].mean(1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), int((d < 0).sum())


def v3_coverage(frame: pd.DataFrame, method: str, level: str) -> dict:
    lo, hi = frame[f"lo__{method}__{level}"], frame[f"hi__{method}__{level}"]
    per = pd.DataFrame({"date": frame.date, "hit": (frame.target >= lo) & (frame.target <= hi), "width": hi - lo})
    per_date = per.groupby("date").hit.mean()
    return {"mean": float(per_date.mean()), "worst": float(per_date.sort_values().head(max(1, len(per_date) // 5)).mean()),
            "share": float((per_date >= float(level) - 0.1).mean()), "width": float(per.width.mean())}


def v3(found: dict) -> None:
    base = ROOT / "evaluation/v3"
    table = pd.read_csv(base / "analysis/outer_case_table.csv", dtype={"identity": str})
    cases = pd.read_csv(base / "prepared/cases.csv", dtype={"identity": str})
    wells = pd.read_csv(base / "prepared/wells.csv.gz", usecols=["date", "Plate.SN", "well"], dtype=str)
    usable = table.usable.astype(bool)

    def losses(column: str, mask: pd.Series = usable) -> pd.Series:
        part = table.loc[mask, ["date", "identity", "target"]].assign(prediction=table.loc[mask, column])
        return gate.date_losses(part)

    found["v3 usable conditions"] = int(usable.sum())
    found["v3 batches with usable conditions"] = int(table.loc[usable].date.nunique())
    found["v3 chemicals"] = int(cases.identity.nunique())
    found["v3 plates"] = int(wells[["date", "Plate.SN"]].drop_duplicates().shape[0])
    found["v3 wells"] = int(len(wells))
    names = {"v2-recipe": "v2_incumbent", "v2+PCT ensemble": "ens_v2_pct", "dose-series Extra Trees": "et_small_chem",
             "nested": "nested_v3", "PCT": "pct", "LightGBM (all features)": "lgbm_full_ctx", "TabPFN v2 (all features)": "tabpfn_full_ctx",
             "MLP": "mlp_full_ctx", "ridge": "ridge_full_ctx", "carry-forward": "persistence_day7", "batch-blind PCT": "pct_no_context"}
    loss = {label: losses(f"pred__{column}") for label, column in names.items()}
    for label in names:
        found[f"v3 {label} date-macro MAE"] = float(loss[label].mean())
    for label, key in [("ensemble", "v2+PCT ensemble"), ("nested", "nested"), ("PCT", "PCT"), ("dose-series Extra Trees", "dose-series Extra Trees"),
                       ("LightGBM", "LightGBM (all features)"), ("TabPFN", "TabPFN v2 (all features)"), ("MLP", "MLP"), ("ridge", "ridge"),
                       ("carry-forward", "carry-forward")]:
        found[f"v3 batches {label} beats v2"] = v3_paired(loss[key], loss["v2-recipe"])[2]
    for label, key in [("nested", "nested"), ("ensemble", "v2+PCT ensemble")]:
        low, high, _ = v3_paired(loss[key], loss["v2-recipe"])
        found[f"v3 {label} minus v2, interval low"], found[f"v3 {label} minus v2, interval high"] = low, high
    external_cohort = usable & (table.cohort == "external28")
    ens_ext, v2_ext = losses("pred__ens_v2_pct", external_cohort), losses("pred__v2_incumbent", external_cohort)
    found["v3 external-cohort ensemble date-macro MAE"] = float(ens_ext.mean())
    found["v3 external-cohort v2 date-macro MAE"] = float(v2_ext.mean())
    low, high, better = v3_paired(ens_ext, v2_ext)
    found["v3 external-cohort ensemble minus v2, interval low"] = low
    found["v3 external-cohort ensemble minus v2, interval high"] = high
    found["v3 external-cohort batches ensemble beats v2"] = better
    low, high, _ = v3_paired(loss["batch-blind PCT"], loss["PCT"])
    found["v3 batch-blind minus PCT, interval low"], found["v3 batch-blind minus PCT, interval high"] = low, high
    found["v3 batches where the batch-blind PCT is worse"] = int((loss["batch-blind PCT"] > loss["PCT"]).sum())

    twin = pd.read_csv(base / "analysis/posthoc_twin_cases.csv", dtype={"identity": str})
    targets = [c[len("truth__"):] for c in twin.columns if c.startswith("truth__")]
    scores = {}
    for name in targets:
        cols = {k: twin[f"{k}__{name}"] for k in ["truth", "pct", "et", "persistence"]}
        ok = twin.usable.astype(bool) & np.isfinite(cols["truth"]) & np.isfinite(cols["pct"]) & np.isfinite(cols["et"]) & np.isfinite(cols["persistence"])
        frame = twin.loc[ok, ["date", "identity"]].assign(target=cols["truth"][ok])
        scores[name] = {k: float(gate.date_losses(frame.assign(prediction=cols[k][ok])).mean()) for k in ["pct", "et", "persistence"]}
    found["v3 twin day-12 network spikes, PCT"] = scores["normalized_ns_d12"]["pct"]
    found["v3 twin day-12 network spikes, carry-forward"] = scores["normalized_ns_d12"]["persistence"]
    found["v3 twin day-12 mutual information, PCT"] = scores["contrast__mi__d12"]["pct"]
    found["v3 twin day-12 mutual information, carry-forward"] = scores["contrast__mi__d12"]["persistence"]
    found["v3 twin targets where PCT beats carry-forward"] = sum(s["pct"] < s["persistence"] for s in scores.values())
    found["v3 twin targets where PCT beats per-target Extra Trees"] = sum(s["pct"] < s["et"] for s in scores.values())
    found["v3 twin mean error ratio, PCT to per-target Extra Trees"] = float(np.mean([s["pct"] / s["et"] for s in scores.values()]))

    intervals = pd.read_csv(base / "analysis/v31_interval_cases.csv")
    kept = intervals.loc[intervals.usable.astype(bool)]
    weighted8, weighted9 = v3_coverage(kept, "batch_weighted", "0.8"), v3_coverage(kept, "batch_weighted", "0.9")
    two8, adaptive8 = v3_coverage(kept, "two_level", "0.8"), v3_coverage(kept, "batch_adaptive", "0.8")
    found["v31 batch-weighted batch-mean coverage at 0.80"] = weighted8["mean"]
    found["v31 batch-weighted batch-mean coverage at 0.90"] = weighted9["mean"]
    found["v31 batch-weighted worst-fifth coverage at 0.80"] = weighted8["worst"]
    found["v31 two-level worst-fifth coverage at 0.80"] = two8["worst"]
    found["v31 two-level share of batches within 0.1 of nominal at 0.80"] = two8["share"]
    found["v31 two-level width increase over batch-weighted at 0.80"] = two8["width"] / weighted8["width"] - 1
    found["v31 batch-adaptive worst-fifth gain at 0.80"] = adaptive8["worst"] - weighted8["worst"]
    realised = (kept.prediction - kept.target).abs().groupby(kept.date).median()
    predicted = kept.groupby("date").batch_difficulty.first()
    found["v31 rank correlation of predicted and realised batch difficulty"] = float(realised.rank().corr(predicted.rank()))


DAY12 = {
    # Post hoc, October 8, 2026: the calibrated day-7 baseline (tools/posthoc_calibrated_day7.py) and the three held-out
    # example batches the day-12 check ships, chosen by rule (best, median, largest loss to carry-forward).
    "calibrated day-7 41-batch date-macro MAE": 0.881,
    "41-batch v2 minus calibrated day-7 interval low": -0.29,
    "41-batch v2 minus calibrated day-7 interval high": -0.10,
    "41-batch batches where v2 beats calibrated day 7": 33,
    "41-batch share of v2 gain over carry-forward from calibration": 0.41,
    "41-batch v2 error reduction vs calibrated day 7": 0.22,
    "external calibrated day-7 date-macro MAE": 0.813,
    "external v2 minus calibrated day-7 interval low": -0.161,
    "external v2 minus calibrated day-7 interval high": 0.002,
    "external batches where v2 beats calibrated day 7": 15,
    "external share of v2 gain over carry-forward from calibration": 0.68,
    "external v2 error reduction vs calibrated day 7": 0.10,
    "external kept conditions": 792,
    "external kept v2 date-macro MAE": 0.541,
    "external kept calibrated day-7 date-macro MAE": 0.554,
    "external kept v2 minus calibrated day-7 interval low": -0.072,
    "external kept v2 minus calibrated day-7 interval high": 0.048,
    "external kept batches where v2 beats calibrated day 7": 14,
    "external share of v2 pooled gain from declined conditions": 0.49,
    "external share of v2 pooled gain from day 7 at most -1": 0.80,
    "41-batch rank correlation of batch-mean difficulty and batch MAE": 0.65,
    "external rank correlation of batch-mean difficulty and batch MAE": 0.60,
    "external median per-batch v2 MAE": 0.69,
    "external 20141231 v2 date-macro MAE": 0.678,
    "external 20141231 persistence date-macro MAE": 1.092,
    "external 20141231 v2 rank of 24": 12,
    "external 20140924 v2 date-macro MAE": 0.959,
    "external 20140924 persistence date-macro MAE": 0.725,
    "external 20140924 coverage at nominal 0.80": 0.38,
    "external 20160120 coverage at nominal 0.80": 0.93,
    "external 20160120 conditions where the forecast is closer": 27,
    "external 20160120 provisional forecast MAE": 0.27,
    "external 20160120 provisional persistence MAE": 0.42,
    "external 20160120 declined forecast MAE": 0.55,
    "external 20160120 declined persistence MAE": 3.20,
}


def day12(found: dict) -> None:
    import posthoc_calibrated_day7 as calibrated

    for prefix, cases, v2, carry, seed in [
            ("41-batch", calibrated.study_cases(), "pred__v2_incumbent", "pred__persistence_day7", 20261007),
            ("external", calibrated.external_cases(), "prediction", "day7", 20260927)]:
        result = calibrated.summary(cases, v2, carry, seed)
        if prefix == "41-batch":
            found["calibrated day-7 41-batch date-macro MAE"] = result["date_macro_mae"]["calibrated_day7"]
        else:
            found["external calibrated day-7 date-macro MAE"] = result["date_macro_mae"]["calibrated_day7"]
        found[f"{prefix} v2 minus calibrated day-7 interval low"], found[f"{prefix} v2 minus calibrated day-7 interval high"] = \
            result["v2_minus_calibrated_interval"]
        found[f"{prefix} batches where v2 beats calibrated day 7"] = result["batches_v2_better_than_calibrated"]
        found[f"{prefix} share of v2 gain over carry-forward from calibration"] = result["share_of_v2_gain_from_calibration"]
        found[f"{prefix} v2 error reduction vs calibrated day 7"] = result["v2_reduction_vs_calibrated"]
        sigma = "sigma__incumbent" if prefix == "41-batch" else "trust_score"
        found[f"{prefix} rank correlation of batch-mean difficulty and batch MAE"] = \
            calibrated.batch_difficulty(cases, sigma, v2)["spearman"]
        if prefix == "external":
            kept = calibrated.kept_external(cases)
            found["external kept conditions"] = kept["conditions"]
            found["external kept v2 date-macro MAE"] = kept["date_macro_mae"]["v2"]
            found["external kept calibrated day-7 date-macro MAE"] = kept["date_macro_mae"]["calibrated_day7"]
            found["external kept v2 minus calibrated day-7 interval low"], \
                found["external kept v2 minus calibrated day-7 interval high"] = kept["v2_minus_calibrated_interval"]
            found["external kept batches where v2 beats calibrated day 7"] = kept["batches_v2_better_than_calibrated"]
            found["external share of v2 pooled gain from declined conditions"] = kept["share_of_pooled_gain_from_declined"]
            found["external share of v2 pooled gain from day 7 at most -1"] = kept["share_of_pooled_gain_from_day7_at_most_minus_1"]

    cases = pd.read_csv(ROOT / "evaluation/external/arm_b_cases.csv", dtype={"identity": str, "spid": str})
    b = cases.loc[cases.primary.astype(bool) & cases.usable.astype(bool)].copy()
    b["persistence"] = b.day7
    v2_losses = gate.date_losses(b.assign(prediction=b.prediction))
    carry_losses = gate.date_losses(b.assign(prediction=b.persistence))
    found["external median per-batch v2 MAE"] = float(v2_losses.median())
    for date in [20141231, 20140924]:
        found[f"external {date} v2 date-macro MAE"] = float(v2_losses.loc[date])
        found[f"external {date} persistence date-macro MAE"] = float(carry_losses.loc[date])
    found["external 20141231 v2 rank of 24"] = int(v2_losses.rank().loc[20141231])
    inside = (b.target >= b["low_0.8"]) & (b.target <= b["high_0.8"])
    for date in [20140924, 20160120]:
        found[f"external {date} coverage at nominal 0.80"] = float(inside[b.date == date].mean())
    one = b.loc[b.date == 20160120]
    forecast, carry = (one.prediction - one.target).abs(), (one.persistence - one.target).abs()
    kept = ~one.declined.astype(bool)
    found["external 20160120 conditions where the forecast is closer"] = int((forecast < carry).sum())
    found["external 20160120 provisional forecast MAE"] = float(forecast[kept].mean())
    found["external 20160120 provisional persistence MAE"] = float(carry[kept].mean())
    found["external 20160120 declined forecast MAE"] = float(forecast[~kept].mean())
    found["external 20160120 declined persistence MAE"] = float(carry[~kept].mean())


HARNESS = {
    # The split-optimism study (protocol experiments/neuroforecast_split_study_protocol.json, committed publicly before
    # its first run) from its per-case out-of-fold table, and the post hoc replay of the lab's own recalibration.
    "split study ET date-macro MAE, random condition": 0.598,
    "split study ET date-macro MAE, chemical": 0.615,
    "split study ET date-macro MAE, batch": 0.688,
    "split study ET date-macro MAE, batch and chemical": 0.702,
    "split study LightGBM date-macro MAE, chemical": 0.616,
    "split study LightGBM date-macro MAE, batch and chemical": 0.690,
    "split study calibrated day-7 date-macro MAE, random condition": 0.855,
    "split study calibrated day-7 date-macro MAE, batch and chemical": 0.887,
    "split study ET optimism, random condition": 0.15,
    "split study ET optimism, chemical": 0.12,
    "split study ET optimism, batch": 0.02,
    "split study LightGBM optimism, random condition": 0.14,
    "split study LightGBM optimism, chemical": 0.11,
    "split study ridge optimism, random condition": 0.05,
    "split study calibrated day-7 optimism, random condition": 0.04,
    "split study ET chemical minus batch and chemical, interval low": -0.139,
    "split study ET chemical minus batch and chemical, interval high": -0.045,
    "split study ET random minus batch and chemical, interval low": -0.157,
    "split study ET random minus batch and chemical, interval high": -0.058,
    "split study ET batches where chemical design looks better": 33,
    "split study ET transfer coverage at 0.80, random condition": 0.747,
    "split study ET transfer coverage at 0.80, chemical": 0.761,
    "split study ET transfer coverage at 0.80, batch": 0.796,
    "split study ET transfer worst-fifth coverage at 0.80, chemical": 0.552,
    "split study ET apparent coverage at 0.80, batch and chemical": 0.812,
    "split study ET apparent worst-fifth coverage at 0.80, batch and chemical": 0.616,
    "split study ET advantage over calibrated day 7, random condition": 0.30,
    "split study ET advantage over calibrated day 7, batch and chemical": 0.21,
    "recalibration batches evaluated": 22,
    "recalibration shipped pooled coverage at 0.80": 0.694,
    "recalibration shipped worst-fifth coverage at 0.80": 0.338,
    "recalibration expanding pooled coverage at 0.80": 0.850,
    "recalibration expanding worst-fifth coverage at 0.80": 0.553,
    "recalibration expanding width increase at 0.80": 0.62,
    "recalibration last-6 pooled coverage at 0.80": 0.801,
    "recalibration last-6 worst-fifth coverage at 0.80": 0.532,
    "recalibration last-6 width increase at 0.80": 0.42,
    "recalibration shipped pooled coverage at 0.90": 0.807,
    "recalibration expanding pooled coverage at 0.90": 0.928,
    "recalibration last-6 pooled coverage at 0.90": 0.903,
}


CHIP = {
    # Chip ingestion (ingestion only) of Zenodo 4679064, from the shipped per-compartment readouts and domain table.
    "chip recordings": 10,
    "chip compartments read as wells": 40,
    "chip electrodes": 760,
    "chip recordings with network activity": 10,
    "chip readout values inside EPA control 1-99%": 0.59,
    "chip correlation values inside EPA control range": 0.90,
    "chip bursting-electrode values inside EPA control range": 0.88,
    "chip active-electrode values inside EPA control range": 0.75,
    "chip firing-rate values inside EPA control range": 0.42,
    "chip network-spike values inside EPA control range": 0.35,
    "chip burst-count rank correlation with the authors' detector": 0.90,
    "chip burst-count ratio, ours to the authors'": 0.27,
}


def chip(found: dict) -> None:
    base = ROOT / "evaluation/chip_ingestion"
    table = pd.read_csv(base / "zenodo_4679064_readouts.csv")
    domain = pd.read_csv(base / "zenodo_4679064_domain.csv")
    summary = json.loads((base / "zenodo_4679064_summary.json").read_text(encoding="utf-8"))
    found["chip recordings"] = int(table.recording.nunique())
    found["chip compartments read as wells"] = int(len(table))
    found["chip electrodes"] = int(table.electrodes.sum())
    found["chip recordings with network activity"] = int((table.groupby("recording")["ns.n"].median() >= 1).sum())
    inside = domain.dropna(subset=["inside_epa_1_99"]).assign(inside=lambda f: f.inside_epa_1_99.astype(str) == "True")
    found["chip readout values inside EPA control 1-99%"] = float(inside.inside.mean())
    share = inside.groupby("readout").inside.mean()
    for label, readout in [("correlation", "r"), ("bursting-electrode", "nABE"), ("active-electrode", "nAE"),
                           ("firing-rate", "meanfiringrate"), ("network-spike", "ns.n")]:
        found[f"chip {label} values inside EPA control range"] = float(share[readout])
    burst = summary["burst_sanity_check"]   # computed from the raw recordings, which are not shipped
    found["chip burst-count rank correlation with the authors' detector"] = burst["spearman_per_electrode_burst_count"]
    found["chip burst-count ratio, ours to the authors'"] = burst["bursts_ours"] / burst["bursts_authors"]


def harness(found: dict) -> None:
    import neuroforecast_harness as h
    import neuroforecast_split_study as study
    import posthoc_lab_recalibration as recalibration

    table = pd.read_csv(ROOT / "evaluation/harness/split_study/oof.csv.gz", dtype={"identity": str})
    by = {d: table.loc[table.design == d] for d in h.DESIGNS}
    names = {"random_condition": "random condition", "chemical": "chemical", "batch": "batch", "batch_chemical": "batch and chemical"}
    models = {"ET": "extra_trees_v2", "LightGBM": "lightgbm", "ridge": "ridge", "calibrated day-7": "calibrated_day7"}
    losses = {(d, m): study.batch_losses(by[d], m) for d in h.DESIGNS for m in models.values()}
    for label, model in models.items():
        for design, name in names.items():
            key = f"split study {label} date-macro MAE, {name}"
            if key in HARNESS:
                found[key] = float(losses[(design, model)].mean())
            key = f"split study {label} optimism, {name}"
            if key in HARNESS:
                found[key] = 1 - losses[(design, model)].mean() / losses[("batch_chemical", model)].mean()
    for design, name in [("chemical", "chemical"), ("random_condition", "random")]:
        diff = h.paired(losses[(design, "extra_trees_v2")], losses[("batch_chemical", "extra_trees_v2")])
        found[f"split study ET {name} minus batch and chemical, interval low"] = diff["low"]
        found[f"split study ET {name} minus batch and chemical, interval high"] = diff["high"]
        if design == "chemical":
            found["split study ET batches where chemical design looks better"] = diff["batches_better"]
    reference = by["batch_chemical"]
    for design, name in [("random_condition", "random condition"), ("chemical", "chemical"), ("batch", "batch")]:
        transfer = study.transfer_coverage(by[design], reference, "extra_trees_v2", 0.8)
        found[f"split study ET transfer coverage at 0.80, {name}"] = transfer["pooled"]
        if design == "chemical":
            found["split study ET transfer worst-fifth coverage at 0.80, chemical"] = transfer["worst_fifth"]
    apparent = study.apparent_coverage(reference, "extra_trees_v2", 0.8)
    found["split study ET apparent coverage at 0.80, batch and chemical"] = apparent["pooled"]
    found["split study ET apparent worst-fifth coverage at 0.80, batch and chemical"] = apparent["worst_fifth"]
    for design, name in [("random_condition", "random condition"), ("batch_chemical", "batch and chemical")]:
        found[f"split study ET advantage over calibrated day 7, {name}"] = \
            1 - losses[(design, "extra_trees_v2")].mean() / losses[(design, "calibrated_day7")].mean()

    replay = recalibration.run()
    for level in ["0.8", "0.9"]:
        entry, tag = replay["levels"][level], "0.80" if level == "0.8" else "0.90"
        found[f"recalibration shipped pooled coverage at {tag}"] = entry["expanding"]["shipped"]["pooled"]
        for name, key in [("expanding", "expanding"), ("last-6", "last_6_batches")]:
            found[f"recalibration {name} pooled coverage at {tag}"] = entry[key]["lab"]["pooled"]
            if level == "0.8":
                found[f"recalibration {name} worst-fifth coverage at 0.80"] = entry[key]["lab"]["worst_fifth"]
                found[f"recalibration {name} width increase at 0.80"] = entry[key]["lab"]["mean_width"] / entry[key]["shipped"]["mean_width"] - 1
        if level == "0.8":
            found["recalibration batches evaluated"] = entry["expanding"]["batches_evaluated"]
            found["recalibration shipped worst-fifth coverage at 0.80"] = entry["expanding"]["shipped"]["worst_fifth"]


REVIEW3 = {
    # Post hoc analyses added after the third mock review, from their shipped evidence: the temporal split
    # (evaluation/posthoc_baselines/temporal_split_predictions.csv.gz), the decision-value tables (saved predictions
    # only) and the oracle constants of the recalibration replay.
    "temporal external reference-split v2 date-macro MAE (927 conditions)": 0.666,
    "temporal period test batches": 24,
    "temporal period conditions": 983,
    "temporal period interleaved v2 date-macro MAE": 0.659,
    "temporal period size-matched v2 date-macro MAE": 0.678,
    "temporal period v2 date-macro MAE": 0.731,
    "temporal period interleaved calibrated day-7 date-macro MAE": 0.823,
    "temporal period calibrated day-7 date-macro MAE": 0.817,
    "temporal period optimism of interleaved": 0.10,
    "temporal period interleaved minus period, interval low": -0.113,
    "temporal period interleaved minus period, interval high": -0.034,
    "temporal period batches where interleaved looks better": 18,
    "temporal period optimism of size-matched": 0.07,
    "temporal period size-matched minus period, interval low": -0.100,
    "temporal period size-matched minus period, interval high": -0.009,
    "temporal period edge over calibrated, interleaved": 0.20,
    "temporal period edge over calibrated, period": 0.11,
    "temporal forward test batches": 31,
    "temporal forward interleaved v2 date-macro MAE": 0.647,
    "temporal forward size-matched v2 date-macro MAE": 0.658,
    "temporal forward v2 date-macro MAE": 0.686,
    "temporal forward optimism of interleaved": 0.06,
    "temporal forward interleaved minus forward, interval low": -0.092,
    "temporal forward interleaved minus forward, interval high": 0.016,
    "temporal forward batches where interleaved looks better": 22,
    "temporal forward optimism of size-matched": 0.04,
    "temporal forward size-matched minus forward, interval low": -0.069,
    "temporal forward size-matched minus forward, interval high": 0.015,
    "split study ET advantage over calibrated day 7, chemical": 0.29,
    "decision collapsed conditions": 433,
    "decision collapsed carry-forward date-macro MAE": 2.207,
    "decision collapsed calibrated date-macro MAE": 2.019,
    "decision collapsed v2 date-macro MAE": 1.407,
    "decision collapsed v2 date-macro MAE, rounded": 1.41,
    "decision collapsed calibrated date-macro MAE, rounded": 2.02,
    "decision collapsed v2 minus calibrated, interval low": -0.842,
    "decision collapsed v2 minus calibrated, interval high": -0.382,
    "decision lowered conditions": 341,
    "decision lowered carry-forward date-macro MAE": 0.873,
    "decision lowered calibrated date-macro MAE": 0.786,
    "decision lowered v2 date-macro MAE": 0.699,
    "decision lowered v2 minus calibrated, interval low": -0.202,
    "decision lowered v2 minus calibrated, interval high": 0.043,
    "decision lowered v2 gain over calibrated": 0.11,
    "decision near-control conditions": 1333,
    "decision near-control carry-forward date-macro MAE": 0.549,
    "decision near-control calibrated date-macro MAE": 0.519,
    "decision near-control v2 date-macro MAE": 0.473,
    "decision near-control v2 minus calibrated, interval low": -0.141,
    "decision near-control v2 minus calibrated, interval high": 0.017,
    "decision near-control v2 gain over calibrated": 0.09,
    "decision raised conditions": 136,
    "decision raised carry-forward date-macro MAE": 0.918,
    "decision raised calibrated date-macro MAE": 0.492,
    "decision raised v2 date-macro MAE": 0.450,
    "decision raised v2 minus calibrated, interval low": -0.095,
    "decision raised v2 minus calibrated, interval high": 0.010,
    "decision raised v2 gain over calibrated": 0.09,
    "decision three-line rule kept conditions": 1810,
    "decision three-line rule calibrated date-macro MAE": 0.577,
    "decision three-line rule v2 date-macro MAE": 0.510,
    "decision three-line rule v2 gain over calibrated": 0.12,
    "decision three-line rule batches where v2 is better": 30,
    "decision three-line rule batches": 40,
    "decision three-line rule v2 minus calibrated, interval low": -0.164,
    "decision three-line rule v2 minus calibrated, interval high": 0.016,
    "decision external large changes": 255,
    "decision external tool no-large-change NPV": 0.864,
    "decision external tool large changes missed": 103,
    "decision external day-7 no-large-change NPV": 0.876,
    "decision external day-7 large changes missed": 88,
    "decision external large changes already 2-fold at day 7": 0.65,
    "decision external large changes declined": 119,
    "decision external confident calls": 33,
    "decision external confident calls large and right sign": 33,
    "decision external confident share of kept": 0.04,
    "recalibration oracle same width as expanding, factor": 1.62,
    "recalibration oracle same width as expanding, pooled coverage": 0.862,
    "recalibration oracle same width as expanding, worst fifth": 0.648,
    "recalibration oracle same width as last-6, factor": 1.42,
    "recalibration oracle same width as last-6, pooled coverage": 0.816,
    "recalibration oracle same width as last-6, worst fifth": 0.576,
    "recalibration oracle nominal, factor": 1.32,
    "recalibration oracle nominal, pooled coverage": 0.801,
    "recalibration oracle nominal, worst fifth": 0.543,
    "recalibration oracle nominal, scale": 1.98,
    "recalibration expanding final scale": 1.97,
    "chip median network spikes per 15 minutes": 574,
}


def review3(found: dict) -> None:
    import neuroforecast_harness as h
    import posthoc_decision_value as decision
    import posthoc_lab_recalibration as recalibration
    import posthoc_temporal_split as temporal

    bench = h.load_benchmark()
    reference = h.reference_predictions(bench)
    arm = pd.read_csv(ROOT / "evaluation/external/arm_b_cases.csv", dtype={"identity": str})
    arm = arm.loc[arm.primary.astype(bool) & arm.usable.astype(bool)].merge(reference[["case_id", "v2"]], on="case_id")
    found["temporal external reference-split v2 date-macro MAE (927 conditions)"] = float(h.date_losses(arm, "v2").mean())

    result = temporal.analyze(pd.read_csv(temporal.PREDICTIONS, dtype={"identity": str}))
    et = "extra_trees_v2"
    for design in ("period", "forward"):
        r = result[design]
        found[f"temporal {design} test batches"] = r["test_batches"]
        found[f"temporal {design} interleaved v2 date-macro MAE"] = r["interleaved"]["date_macro_mae"][et]
        found[f"temporal {design} size-matched v2 date-macro MAE"] = r["size_matched"]["date_macro_mae"][et]
        found[f"temporal {design} v2 date-macro MAE"] = r[design]["date_macro_mae"][et]
        for name, key in [("interleaved", f"interleaved_vs_{design}"), ("size-matched", f"size_matched_vs_{design}")]:
            c = r[key][et]
            found[f"temporal {design} optimism of {name}"] = -c["relative"]
            found[f"temporal {design} {name} minus {design}, interval low"] = c["low"]
            found[f"temporal {design} {name} minus {design}, interval high"] = c["high"]
        found[f"temporal {design} batches where interleaved looks better"] = r[f"interleaved_vs_{design}"][et]["batches_better"]
    p = result["period"]
    found["temporal period conditions"] = p["conditions"]
    found["temporal period interleaved calibrated day-7 date-macro MAE"] = p["interleaved"]["date_macro_mae"]["calibrated_day7"]
    found["temporal period calibrated day-7 date-macro MAE"] = p["period"]["date_macro_mae"]["calibrated_day7"]
    found["temporal period edge over calibrated, interleaved"] = p["interleaved"]["edge_over_calibrated"]
    found["temporal period edge over calibrated, period"] = p["period"]["edge_over_calibrated"]

    table = pd.read_csv(ROOT / "evaluation/harness/split_study/oof.csv.gz", dtype={"identity": str})
    import neuroforecast_split_study as study
    chem = table.loc[table.design == "chemical"]
    found["split study ET advantage over calibrated day 7, chemical"] = \
        1 - study.batch_losses(chem, "extra_trees_v2").mean() / study.batch_losses(chem, "calibrated_day7").mean()

    value = decision.run()
    bands = dict(zip(["collapsed", "lowered", "near-control", "raised"], value["where_ml_helps"]["bands"].values()))
    for name, r in bands.items():
        m, d = r["date_macro_mae"], r["v2_minus_calibrated"]
        found[f"decision {name} conditions"] = r["conditions"]
        found[f"decision {name} carry-forward date-macro MAE"] = m["carry_forward"]
        found[f"decision {name} calibrated date-macro MAE"] = m["calibrated_day7"]
        found[f"decision {name} v2 date-macro MAE"] = m["v2"]
        found[f"decision {name} v2 minus calibrated, interval low"] = d["low"]
        found[f"decision {name} v2 minus calibrated, interval high"] = d["high"]
        found[f"decision {name} v2 gain over calibrated"] = 1 - m["v2"] / m["calibrated_day7"]
    found["decision collapsed v2 date-macro MAE, rounded"] = found["decision collapsed v2 date-macro MAE"]
    found["decision collapsed calibrated date-macro MAE, rounded"] = found["decision collapsed calibrated date-macro MAE"]
    keep = value["where_ml_helps"]["three_line_rule_keeps"]
    found["decision three-line rule kept conditions"] = keep["conditions"]
    found["decision three-line rule calibrated date-macro MAE"] = keep["date_macro_mae"]["calibrated_day7"]
    found["decision three-line rule v2 date-macro MAE"] = keep["date_macro_mae"]["v2"]
    found["decision three-line rule v2 gain over calibrated"] = 1 - keep["date_macro_mae"]["v2"] / keep["date_macro_mae"]["calibrated_day7"]
    found["decision three-line rule batches where v2 is better"] = keep["v2_minus_calibrated"]["batches_better"]
    found["decision three-line rule batches"] = keep["v2_minus_calibrated"]["batches"]
    found["decision three-line rule v2 minus calibrated, interval low"] = keep["v2_minus_calibrated"]["low"]
    found["decision three-line rule v2 minus calibrated, interval high"] = keep["v2_minus_calibrated"]["high"]
    b = value["binary_call"]
    found["decision external large changes"] = b["large_changes"]["anticipated_confidently"] + b["large_changes"]["declined"] \
        + b["large_changes"]["kept_not_confident"]
    found["decision external tool no-large-change NPV"] = b["tool_no_large_change_call"]["npv"]
    found["decision external tool large changes missed"] = b["tool_no_large_change_call"]["large_changes_missed"]
    found["decision external day-7 no-large-change NPV"] = b["day7_no_large_change_call"]["npv"]
    found["decision external day-7 large changes missed"] = b["day7_no_large_change_call"]["large_changes_missed"]
    found["decision external large changes already 2-fold at day 7"] = b["large_already_2_fold_at_day7"]
    found["decision external large changes declined"] = b["large_changes"]["declined"]
    found["decision external confident calls"] = b["kept_intervals_excluding_zero"]
    found["decision external confident calls large and right sign"] = b["confident_calls_large_and_right_sign"]
    found["decision external confident share of kept"] = b["kept_intervals_excluding_zero"] / b["kept"]

    replay = recalibration.run()["levels"]["0.8"]
    for name, key in [("expanding", "expanding"), ("last-6", "last_6_batches")]:
        o = replay[key]["oracle_constant_same_width"]
        found[f"recalibration oracle same width as {name}, factor"] = o["factor"]
        found[f"recalibration oracle same width as {name}, pooled coverage"] = o["pooled"]
        found[f"recalibration oracle same width as {name}, worst fifth"] = o["worst_fifth"]
    o = replay["oracle_constant_nominal"]
    found["recalibration oracle nominal, factor"] = o["factor"]
    found["recalibration oracle nominal, pooled coverage"] = o["pooled"]
    found["recalibration oracle nominal, worst fifth"] = o["worst_fifth"]
    found["recalibration oracle nominal, scale"] = o["factor"] * replay["shipped_scale"]
    found["recalibration expanding final scale"] = replay["expanding"]["lab_scale_last"]

    summary = json.loads((ROOT / "evaluation/chip_ingestion/zenodo_4679064_summary.json").read_text(encoding="utf-8"))
    table = pd.read_csv(ROOT / "evaluation/chip_ingestion/zenodo_4679064_readouts.csv")
    median = float(table.groupby("recording")["ns.n"].median().median())
    assert abs(median - summary["reference_check"]["median_network_spikes_per_15_min"]) < 1e-9
    found["chip median network spikes per 15 minutes"] = int(round(median))


def paired_interval(frame: pd.DataFrame, a: str, b: str) -> list[float]:
    """Same resampling as the scoring code: 10000 paired date resamples, seed 20260927."""
    la = gate.date_losses(frame.assign(prediction=frame[a]))
    lb = gate.date_losses(frame.assign(prediction=frame[b]))
    index = np.random.default_rng(20260927).integers(len(la), size=(10000, len(la)))
    diff = la.to_numpy()[index].mean(axis=1) - lb.to_numpy()[index].mean(axis=1)
    return np.quantile(diff, [0.025, 0.975]).tolist()


def external(found: dict) -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import neuroforecast_v2 as v2

    read = lambda p: pd.read_csv(ROOT / p, dtype={"identity": str, "spid": str})
    a_all, b_all = read("evaluation/external/arm_a_cases.csv"), read("evaluation/external/arm_b_cases.csv")
    keep = lambda f: f.loc[f.primary.astype(bool) & f.usable.astype(bool)].reset_index(drop=True)
    a, b = keep(a_all), keep(b_all)
    a["persistence"], b["persistence"] = a.day7, b.day7
    found["external primary conditions"] = len(b)
    found["external primary dates"] = int(b.date.nunique())
    found["external primary unseen chemicals"] = int(b.identity.nunique())
    found["external secondary conditions"] = int((~b_all.primary.astype(bool) & b_all.usable.astype(bool)).sum())
    losses = {name: gate.date_losses(frame.assign(prediction=frame[col])) for name, frame, col in
              [("v2", b, "prediction"), ("persistence", b, "persistence"), ("v1", a, "prediction"), ("refit", b, "v1_refit")]}
    losses["control"] = gate.date_losses(b.assign(prediction=0.0))
    found["external v2 date-macro MAE"] = float(losses["v2"].mean())
    found["external persistence date-macro MAE"] = float(losses["persistence"].mean())
    found["external frozen v1 date-macro MAE"] = float(losses["v1"].mean())
    found["external control-reference date-macro MAE"] = float(losses["control"].mean())
    found["external v1 refit date-macro MAE"] = float(losses["refit"].mean())
    found["external v2 dates better than persistence"] = int((losses["v2"] < losses["persistence"]).sum())
    found["external frozen v1 dates better than persistence"] = int((losses["v1"] < losses["persistence"]).sum())
    lo, hi = paired_interval(b, "prediction", "persistence")
    found["external v2 minus persistence interval low"], found["external v2 minus persistence interval high"] = lo, hi
    lo, hi = paired_interval(a, "prediction", "persistence")
    found["external frozen v1 minus persistence interval low"], found["external frozen v1 minus persistence interval high"] = lo, hi
    found["external v2 relative error reduction vs persistence"] = float(1 - losses["v2"].mean() / losses["persistence"].mean())

    def kept(frame, mask, column):
        error = abs(frame[column] - frame.target).to_numpy()
        return float(error[mask].mean()), float(error.mean())

    keep_b = ~b.declined.astype(bool).to_numpy()
    found["external v2 declined"] = int((~keep_b).sum())
    found["external v2 kept MAE"], found["external v2 all MAE"] = kept(b, keep_b, "prediction")
    found["external v2 kept reduction"] = 1 - found["external v2 kept MAE"] / found["external v2 all MAE"]
    found["external persistence kept MAE (v2 verdict)"], found["external persistence all MAE"] = kept(b, keep_b, "persistence")
    found["external persistence kept reduction (v2 verdict)"] = 1 - found["external persistence kept MAE (v2 verdict)"] / found["external persistence all MAE"]
    keep_a = a.forecast_trusted.astype(bool).to_numpy()
    found["external v1 declined"] = int((~keep_a).sum())
    found["external v1 kept MAE"], found["external v1 all MAE"] = kept(a, keep_a, "prediction")
    found["external v1 kept reduction"] = 1 - found["external v1 kept MAE"] / found["external v1 all MAE"]
    found["external persistence kept MAE (v1 verdict)"], _ = kept(a, keep_a, "persistence")
    found["external persistence kept reduction (v1 verdict)"] = 1 - found["external persistence kept MAE (v1 verdict)"] / found["external persistence all MAE"]
    order = np.argsort(-abs(a.prediction.to_numpy()), kind="stable")
    extreme = np.ones(len(a), dtype=bool)
    extreme[order[: int((~keep_a).sum())]] = False
    found["external v1 extremity rule kept MAE"], _ = kept(a, extreme, "prediction")
    error_b = abs(b.prediction - b.target).to_numpy()
    found["external v2 risk-coverage area, difficulty"] = v2.risk_coverage(b.trust_score.to_numpy(), error_b)
    found["external v2 risk-coverage area, extremity"] = v2.risk_coverage(abs(b.prediction.to_numpy()), error_b)

    warn = lambda frame, score: v2.expected_detections(frame, score)
    found["external early-quiet positives"] = warn(b, abs(b.prediction.to_numpy()))["positives"]
    found["external early-quiet detected, v1"] = warn(a, abs(a.prediction.to_numpy()))["expected_detected"]
    found["external early-quiet detected, v2 locked score"] = warn(b, abs(b.v1_refit.to_numpy()))["expected_detected"]
    found["external early-quiet detected, highest dose"] = warn(b, b.dose.to_numpy(dtype=float))["expected_detected"]
    found["external early-quiet detected, day-7 magnitude"] = warn(b, abs(b.day7.to_numpy()))["expected_detected"]
    found["external early-quiet detected, chance"] = warn(b, b.dose.to_numpy(dtype=float))["random_expected"]
    inside = lambda frame, lo, hi: float(((frame.target >= frame[lo]) & (frame.target <= frame[hi])).mean())
    found["external v1 coverage at nominal 0.80"] = inside(a, "forecast_low", "forecast_high")
    found["external v2 coverage at nominal 0.80"] = inside(b, "low_0.8", "high_0.8")
    found["external v2 coverage at nominal 0.90"] = inside(b, "low_0.9", "high_0.9")

    units = pd.read_csv(ROOT / "evaluation/external/decision_units.csv")
    reference = units.reference_active.astype(bool)
    early = units.call != "continue"
    agree = lambda calls, ref: float((calls[early] == np.where(ref[early], "active", "inactive")).mean())
    found["external decision samples"] = len(units)
    found["external decision samples active"] = int(reference.sum())
    day7_call = pd.Series(np.where(units["div7.hitsum"] >= 1, "active", "inactive"))
    found["external decision EPA day-7 call agreement"] = float((day7_call == np.where(reference, "active", "inactive")).mean())
    found["external decision early share"] = float(early.mean())
    found["external decision early agreement"] = agree(units.call, reference)
    found["external decision early agreement, secondary reference"] = agree(units.call, units["auc.hitsum"] >= 3)

    reserve = pd.read_csv(ROOT / "evaluation/reserve_intervals.csv", dtype={"identity": str})
    found["external pooled day-7/day-12 correlation"] = float(np.corrcoef(b.day7, b.target)[0, 1])
    found["external large-change share"] = float((abs(b.target) >= 1).mean())
    found["reserve large-change share"] = float((abs(reserve.target) >= 1).mean())
    dev = read("evaluation/v2_development/dev_oof_predictions.csv")
    dev = dev.loc[dev.usable.astype(bool)]
    found["v2 development pooled day-7/day-12 correlation"] = float(np.corrcoef(dev.day7, dev.target)[0, 1])
    found["v2 development product date-macro MAE"] = gate.macro_mae(dev.assign(prediction=dev.pred__small_refit))
    found["v2 development persistence date-macro MAE"] = gate.macro_mae(dev.assign(prediction=dev.day7))
    found["v2 development nested selection estimate"] = gate.macro_mae(dev.assign(prediction=dev.pred__nested))
    found["v2 development conditions"] = len(dev)

    declined = b.declined.astype(bool).to_numpy()
    error = abs(b.prediction - b.target).to_numpy()
    found["external v2 declined MAE"] = float(error[declined].mean())
    found["external v2 declined-to-kept error ratio"] = float(error[declined].mean() / error[~declined].mean())
    found["external v2 share of error in declined conditions"] = float(error[declined].sum() / error.sum())
    found["external v2 kept median absolute error"] = float(np.median(error[~declined]))
    found["external v2 kept median fold error"] = float(2 ** np.median(error[~declined]))
    found["external v2 share of conditions kept"] = float((~declined).mean())
    per_batch = b.assign(d=declined).groupby("date").d.sum()
    found["external v2 fewest declined in a batch"] = int(per_batch.min())
    found["external v2 most declined in a batch"] = int(per_batch.max())
    found["external v2 mean declined per batch"] = float(per_batch.mean())
    found["external v2 batches where persistence won"] = int((losses["persistence"] < losses["v2"]).sum())
    secondary = b_all.loc[~b_all.primary.astype(bool) & b_all.usable.astype(bool)]
    found["external secondary v2 date-macro MAE"] = gate.macro_mae(secondary)
    found["external secondary persistence date-macro MAE"] = gate.macro_mae(secondary.assign(prediction=secondary.day7))
    found["external 20160120 v2 date-macro MAE"] = float(losses["v2"].loc[20160120])
    found["external 20160120 persistence date-macro MAE"] = float(losses["persistence"].loc[20160120])
    dev_all = read("evaluation/v2_development/dev_oof_predictions.csv")
    dev_all = dev_all.loc[dev_all.usable.astype(bool)]
    for label, column in [("v1 refit", "v1_refit"), ("anchored residual", "anchored_residual"), ("batch blend", "batch_blend"),
                          ("anchored residual with batch summaries", "anchored_residual_batch"), ("control-reference", "control_reference")]:
        found[f"v2 development {label} date-macro MAE"] = gate.macro_mae(dev_all.assign(prediction=dev_all[f"pred__{column}"]))
    # Selection statistics come from the hashed development results (checked against the lock by the tests).
    results = json.loads((ROOT / "evaluation/v2_development/development_results.json").read_text(encoding="utf-8"))
    choices = list(results["nested_selected_by_fold"].values())
    found["v2 development nested choices of the small model"] = choices.count("small_refit")
    found["v2 development nested choices of the blend"] = choices.count("batch_blend")
    found["v2 development risk-coverage area, difficulty"] = results["trust_aurc"]["sigma_v2"]
    found["v2 development risk-coverage area, extremity"] = results["trust_aurc"]["extremity"]
    sub = json.loads((ROOT / "evaluation/v2_posthoc/control_subsampling.json").read_text(encoding="utf-8"))
    by_k = {str(r["controls_per_plate"]): r for r in sub["summary"]}
    for k, label in [("1", "1 control"), ("2", "2 controls"), ("4", "4 controls")]:
        r = by_k[k]
        found[f"subsampling {label} v2 MAE"] = r["product_date_macro"]
        found[f"subsampling {label} persistence MAE"] = r["persistence_date_macro"]
        found[f"subsampling {label} declined share"] = r["declined_share"]
        found[f"subsampling {label} false batch verdicts"] = r["low_reference_batches"] if k == "1" else int(round(r["low_reference_batches"]))


def signal(found: dict) -> None:
    """The post hoc analyses, recomputed from per-case tables; only permutation importance is read from its JSON."""
    sys.path.insert(0, str(ROOT / "tools"))
    import neuroforecast_v2 as v2

    read = lambda p: pd.read_csv(ROOT / p, dtype={"identity": str, "spid": str})
    oof = read("evaluation/v2_posthoc/dev_linear_oof.csv")
    saved = read("evaluation/v2_development/dev_oof_predictions.csv")
    if not np.array_equal(oof.small_refit.to_numpy(), saved.pred__small_refit.to_numpy()):
        raise SystemExit("dev_linear_oof.csv does not carry the saved development predictions")
    dev = oof.loc[oof.usable.astype(bool)]
    found["posthoc development least-squares linear date-macro MAE"] = gate.macro_mae(dev.assign(prediction=dev.ols))
    found["posthoc development median-regression linear date-macro MAE"] = gate.macro_mae(dev.assign(prediction=dev.lad))
    analysis = json.loads((ROOT / "evaluation/v2_posthoc/signal_analysis.json").read_text(encoding="utf-8"))
    importance = {r["input"]: r["mae_increase"] for r in analysis["development"]["permutation_importance"]}
    for label, key in [("day-7 active electrodes", "contrast__nAE__d7"), ("day-7 network spikes", "contrast__ns.n__d7"),
                       ("day-7 firing rate", "contrast__meanfiringrate__d7"), ("all day-7 inputs", "all day-7 inputs"),
                       ("all day-5 inputs", "all day-5 inputs"), ("dose", "log10_dose")]:
        found[f"posthoc importance, {label}"] = importance[key]

    arm = read("evaluation/external/arm_b_cases.csv")
    ext = arm.loc[arm.primary.astype(bool) & arm.usable.astype(bool)].reset_index(drop=True)
    states = {"strongly suppressed": lambda d: d <= -1, "suppressed": lambda d: (d > -1) & (d <= -0.5),
              "near-control": lambda d: abs(d) < 0.5, "raised": lambda d: (d >= 0.5) & (d < 1), "hyperactive": lambda d: d >= 1,
              "raised or hyperactive": lambda d: d >= 0.5}
    for prefix, frame, column in [("external", ext, "prediction"), ("development", dev, "small_refit")]:
        for name, rule in states.items():
            part = frame.loc[rule(frame.day7)]
            found[f"posthoc {prefix} {name} conditions"] = len(part)
            found[f"posthoc {prefix} {name} v2 MAE"] = float(abs(part[column] - part.target).mean())
            found[f"posthoc {prefix} {name} persistence MAE"] = float(abs(part.day7 - part.target).mean())
    hyper = ext.loc[ext.day7 >= 1]
    found["posthoc external hyperactive mean observed change"] = float((hyper.target - hyper.day7).mean())
    found["posthoc external hyperactive mean forecast change"] = float((hyper.prediction - hyper.day7).mean())

    declined = ext.declined.astype(bool).to_numpy()
    error = abs(ext.prediction - ext.target).to_numpy()
    extreme = np.zeros(len(ext), dtype=bool)
    extreme[np.argsort(-abs(ext.prediction.to_numpy()), kind="stable")[: declined.sum()]] = True
    found["posthoc external v2 kept MAE, trust verdict"] = float(error[~declined].mean())
    found["posthoc external v2 kept MAE, declining the most extreme forecasts"] = float(error[~extreme].mean())
    found["posthoc external conditions declined by both rules"] = int((declined & extreme).sum())
    found["posthoc external highest day-7 value among declined"] = float(ext.day7[declined].max())
    secondary = arm.loc[~arm.primary.astype(bool) & arm.usable.astype(bool)]
    secondary_error = abs(secondary.prediction - secondary.target).to_numpy()
    found["external secondary risk-coverage area, difficulty"] = v2.risk_coverage(secondary.trust_score.to_numpy(), secondary_error)
    found["external secondary risk-coverage area, extremity"] = v2.risk_coverage(abs(secondary.prediction.to_numpy()), secondary_error)
    found["external secondary v2 coverage at nominal 0.80"] = float(((secondary.target >= secondary["low_0.8"]) & (secondary.target <= secondary["high_0.8"])).mean())

    # The shipped batch rule: half or more conditions below one day-7 control network spike (plate median).
    fires, checked = [], 0
    for prefix in ["dev", "ext"]:
        features = pd.read_csv(ROOT / f"evaluation/v2_prepared/{prefix}_features.csv", usecols=["rel__plate_ctrl_median_ns_d7"])
        cases = read(f"evaluation/v2_prepared/{prefix}_cases.csv").assign(low=np.sinh(features.rel__plate_ctrl_median_ns_d7.to_numpy()) < 1)
        share = cases.groupby("date").low.mean()
        checked += len(share)
        fires += [int(d) for d in share.index[share >= 0.5]]
    found["posthoc batches where the batch rule fires"] = len(fires)
    found["posthoc batches checked by the batch rule"] = checked
    for date in [20160907, 20170628, 20171004]:
        if date not in fires:
            raise SystemExit(f"batch rule expected to fire on {date}")
        part = dev.loc[dev.date == date]
        found[f"posthoc batch rule {date} v2 MAE"] = gate.macro_mae(part.assign(prediction=part.small_refit))
        found[f"posthoc batch rule {date} persistence MAE"] = gate.macro_mae(part.assign(prediction=part.day7))

    reserve = read("evaluation/reserve_intervals.csv")
    found["browser replay external conditions"] = len(read("evaluation/v2_reference/external_triage_v2.csv"))
    found["browser replay reserve-batch conditions"] = len(read("evaluation/v2_reference/reserve_triage_v2.csv"))
    found["reserve pooled day-7/day-12 correlation"] = float(np.corrcoef(reserve.day7, reserve.target)[0, 1])

    v2_loss = gate.date_losses(ext)
    carry_loss = gate.date_losses(ext.assign(prediction=ext.day7))
    persistence_r = pd.Series({d: np.corrcoef(g.day7, g.target)[0, 1] for d, g in ext.groupby("date")})
    found["external per-batch v2 advantage vs batch persistence correlation"] = float(np.corrcoef(carry_loss - v2_loss, persistence_r.loc[v2_loss.index])[0, 1])

    locked_q = json.loads((ROOT / "evaluation/external/results.json").read_text(encoding="utf-8"))["lock"]["conformal_quantiles"]["0.8"]
    sigma = (ext["high_0.8"] - ext["low_0.8"]) / (2 * locked_q)
    score = error / sigma
    dates = sorted(ext.date.unique())
    for k in [4, 12]:
        first = ext.date.isin(dates[:k]).to_numpy()
        calibration = np.sort(score[first])
        q = calibration[int(np.ceil((len(calibration) + 1) * 0.8)) - 1]
        rest = ~first
        label = f"posthoc recalibrated on {k} batches"
        found[f"{label}, locked coverage on the other {len(dates) - k}"] = float((score[rest] <= locked_q).mean())
        found[f"{label}, recalibrated coverage on the other {len(dates) - k}"] = float((score[rest] <= q).mean())
        if k == 4:
            found[f"{label}, calibration conditions"] = int(first.sum())
            found[f"{label}, locked mean width"] = float((2 * locked_q * sigma[rest]).mean())
            found[f"{label}, recalibrated mean width"] = float((2 * q * sigma[rest]).mean())


def expected_detections(frame: pd.DataFrame, score: np.ndarray, budget: float = 0.2) -> float:
    """Early-quiet detections at the budget when tied scores are broken at random (expected value)."""
    work = frame.assign(score=score, positive=abs(frame.target) >= 1).loc[abs(frame.day7) < 0.5]
    total = 0.0
    for _, group in work.groupby("date"):
        k = int(np.ceil(budget * len(group)))
        cutoff = np.sort(group.score.to_numpy())[::-1][k - 1]
        above, tied = group[group.score > cutoff], group[group.score == cutoff]
        total += above.positive.sum() + tied.positive.sum() * (k - len(above)) / len(tied)
    return float(total)


def retained_after_declining_extremes(frame: pd.DataFrame, declined: int) -> float:
    kept = frame.assign(extremity=abs(frame.prediction)).sort_values("extremity", ascending=False).iloc[declined:]
    return float(abs(kept.prediction - kept.target).mean())


def post_hoc(found: dict, threshold: float) -> None:
    development = pd.read_csv(ROOT / "evaluation/development_intervals.csv", dtype={"identity": str})
    reserve = pd.read_csv(ROOT / "evaluation/reserve_intervals.csv", dtype={"identity": str})
    usable = development.loc[development.day12_ctrl_median_ns >= 5]
    found["development usable-reference conditions"] = len(usable)
    found["development usable-reference product date-macro MAE"] = gate.macro_mae(usable)
    found["development usable-reference persistence date-macro MAE"] = gate.macro_mae(usable.assign(prediction=usable.day7))

    def correlation(group):
        return float(np.corrcoef(group.day7, group.target)[0, 1])

    by_date = {int(d): g for d, g in reserve.groupby("date")}
    for date in [20170920, 20171011, 20171004]:
        found[f"reserve {date} day-7/day-12 correlation"] = correlation(by_date[date])
    development_r = [correlation(g) for _, g in development.groupby("date")]
    found["development lowest per-date day-7/day-12 correlation"] = min(development_r)
    found["development highest per-date day-7/day-12 correlation"] = max(development_r)
    persistent = [by_date[20170920], by_date[20171011]]
    found["reserve 20170920 day-12 slope on day 7"] = float(np.polyfit(by_date[20170920].day7, by_date[20170920].target, 1)[0])
    found["reserve 20171011 day-12 slope on day 7"] = float(np.polyfit(by_date[20171011].day7, by_date[20171011].target, 1)[0])
    forecast_slopes = [float(np.polyfit(g.day7, g.prediction, 1)[0]) for g in persistent]
    found["reserve persistent-batch lowest forecast slope"] = min(forecast_slopes)
    found["reserve persistent-batch highest forecast slope"] = max(forecast_slopes)

    unflagged = reserve.loc[~reserve.flag_low_reference_activity.astype(bool)]
    found["reserve unflagged product date-macro MAE"] = gate.macro_mae(unflagged)
    found["reserve unflagged persistence date-macro MAE"] = gate.macro_mae(unflagged.assign(prediction=unflagged.day7))
    collapsed = by_date[20171004]
    found["reserve 20171004 conditions with day-7 value exactly zero"] = int((collapsed.day7 == 0).sum())
    found["reserve 20171004 early-quiet conditions"] = int((abs(collapsed.day7) < 0.5).sum())

    found["reserve early-quiet expected detections by persistence, random ties"] = expected_detections(reserve, abs(reserve.day7).to_numpy())
    found["reserve early-quiet expected detections by highest dose"] = expected_detections(reserve, reserve.dose.to_numpy(dtype=float))
    found["development early-quiet expected detections by highest dose"] = expected_detections(development, development.dose.to_numpy(dtype=float))
    found["development early-quiet detected at 20% budget"] = early_quiet_recall(development, abs(development.prediction).to_numpy())[0]
    quiet = reserve.loc[abs(reserve.day7) < 0.5]
    flagged = pd.concat(g.assign(s=abs(g.prediction)).nlargest(int(np.ceil(0.2 * len(g))), "s") for _, g in quiet.groupby("date"))
    detected = flagged.loc[abs(flagged.target) >= 1]
    found["reserve early-quiet detections with the wrong direction"] = int((np.sign(detected.prediction) != np.sign(detected.target)).sum())

    reserve_declined = int((reserve.sigma > threshold).sum())
    development_declined = int((development.sigma > threshold).sum())
    found["reserve retained-case MAE declining the most extreme forecasts"] = retained_after_declining_extremes(reserve, reserve_declined)
    found["development retained-case MAE declining the most extreme forecasts"] = retained_after_declining_extremes(development, development_declined)
    retained = reserve.sigma <= threshold
    product_error, simple_error = abs(reserve.prediction - reserve.target), abs(reserve.day7 - reserve.target)
    found["reserve abstention error reduction, product"] = float(1 - product_error[retained].mean() / product_error.mean())
    found["reserve abstention error reduction, persistence"] = float(1 - simple_error[retained].mean() / simple_error.mean())


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
    post_hoc(found, threshold)
    return found


if __name__ == "__main__":
    found = recompute()
    external(found)
    signal(found)
    v3(found)
    day12(found)
    harness(found)
    chip(found)
    review3(found)
    PUBLISHED.update(EXTERNAL)
    PUBLISHED.update(SIGNAL)
    PUBLISHED.update(V3)
    PUBLISHED.update(DAY12)
    PUBLISHED.update(HARNESS)
    PUBLISHED.update(CHIP)
    PUBLISHED.update(REVIEW3)
    failures = []
    decimals = written_decimals()
    for label, published in PUBLISHED.items():
        actual = found[label]
        ok = abs(actual - published) <= tolerance_for(published, decimals.get(label)) + 1e-12
        print(f"{'ok ' if ok else 'BAD'}  {label:58s} published {published:<10} recomputed {round(actual, 5)}")
        if not ok:
            failures.append(label)
    print()
    if failures:
        raise SystemExit(f"{len(failures)} published value(s) do not match the evidence: {failures}")
    print(f"All {len(PUBLISHED)} published values match the per-case evidence tables.")
