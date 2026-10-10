"""NeuroForecast v3.1 interval study (protocol: experiments/neuroforecast_v31_interval_protocol.json).

Coverage per culture batch for the v2 recipe's forecasts on the 41 v3 batches: the v2 pooled method, the v3
batch-weighted method, batch-adaptive intervals scaled by a predicted batch difficulty, and a two-level
calibration that targets coverage in most batches. Reads the locked v3 predictions; writes
E:/kaggle/AI4S/experiments/neuroforecast_v3/analysis/v31_intervals.json and v31_interval_cases.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402
import neuroforecast_v3_analyze as an  # noqa: E402
import neuroforecast_v3_data as v3d  # noqa: E402
import neuroforecast_v3_run as run  # noqa: E402

PROTOCOL = data.ROOT / "experiments/neuroforecast_v31_interval_protocol.json"
LEVELS = [0.8, 0.9]
METHODS = ["pooled_scaled", "batch_weighted", "batch_adaptive", "two_level"]


def batch_table(features: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """One row per batch from day-5/day-7 information only."""
    cols = v3d.CONTEXT + v2.BATCH
    frame = features[cols + gate.RELIABILITY].assign(date=cases.date.to_numpy())
    table = frame.groupby("date")[cols].first()
    rel = frame.groupby("date")[gate.RELIABILITY].median().add_prefix("median_")
    return table.join(rel)


def difficulty_model():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=10.0))


def per_batch_target(dates: np.ndarray, scores: np.ndarray) -> pd.Series:
    return np.log(pd.Series(scores).groupby(dates).median().clip(lower=1e-3))


def coverage_table(cases: pd.DataFrame, mask: np.ndarray, low: np.ndarray, high: np.ndarray, level: float) -> dict:
    y = cases.target.to_numpy()
    frame = pd.DataFrame({"date": cases.date, "hit": (y >= low) & (y <= high), "width": high - low}).loc[mask]
    per_date = frame.groupby("date").hit.mean()
    worst = per_date.sort_values().head(max(1, len(per_date) // 5))
    return {"batch_mean": float(per_date.mean()), "worst_fifth": float(worst.mean()),
            "share_batches_within_0.1": float((per_date >= level - 0.1).mean()), "pooled": float(frame.hit.mean()),
            "mean_width": float(frame.width.mean()), "batches": int(len(per_date)),
            "per_batch": {str(k): float(v) for k, v in per_date.items()}}


def run_study(out: Path) -> dict:
    run.verify_lock(out)
    cfg = data.protocol()
    loaded = v3d.load(out)
    x, cases = loaded["features"], loaded["cases"]
    cols = run.column_sets(x)["sigma"]
    batches = batch_table(x, cases)
    n = len(cases)
    y = cases.target.to_numpy()
    usable = cases.usable.to_numpy(dtype=bool)
    preds = pd.read_csv(out / "predictions/v2_incumbent.csv")
    point = np.full(n, np.nan)
    bounds = {(m, lv): (np.full(n, np.nan), np.full(n, np.nan)) for m in METHODS for lv in LEVELS}
    difficulty_test = np.full(n, np.nan)
    feats = an.sigma_features(x, cols, None)
    for k, dates, train, test in v3d.outer_splits(cases):
        inner = an.assemble(preds, n, k, "inner")
        outer = an.assemble(preds, n, k, "outer")
        point[test] = outer[test]
        cal = train[usable[train]]
        res = np.abs(inner - y)
        sig_cal = np.full(n, np.nan)
        folds = v3d.interleave(cases.date.iloc[train], 4)
        with threadpool_limits(limits=1):
            for fold_dates in folds:
                hold = cal[cases.date.iloc[cal].isin(fold_dates).to_numpy()]
                fit = cal[~cases.date.iloc[cal].isin(fold_dates).to_numpy()]
                m = an.sigma_model(cfg).fit(feats.iloc[fit], np.round(res[fit], gate.RESIDUAL_DECIMALS))
                sig_cal[hold] = np.maximum(m.predict(feats.iloc[hold]), gate.SIGMA_FLOOR)
            m = an.sigma_model(cfg).fit(feats.iloc[cal], np.round(res[cal], gate.RESIDUAL_DECIMALS))
            sig_test = np.maximum(m.predict(feats.iloc[test]), gate.SIGMA_FLOOR)
        s = res[cal] / sig_cal[cal]
        cal_dates = cases.date.iloc[cal].to_numpy()
        w = an.date_weights(cal_dates)

        # Batch difficulty: cross-fitted for calibration scores, refitted on all calibration batches for the test batches.
        b_cal = np.full(len(cal), np.nan)
        for fold_dates in folds:
            hold = np.isin(cal_dates, fold_dates)
            target = per_batch_target(cal_dates[~hold], s[~hold])
            model = difficulty_model().fit(batches.loc[target.index], target.to_numpy())
            held = sorted(set(cal_dates[hold]))
            pred = pd.Series(np.exp(model.predict(batches.loc[held])), index=held)
            b_cal[hold] = pd.Series(cal_dates[hold]).map(pred).to_numpy()
        target = per_batch_target(cal_dates, s)
        model = difficulty_model().fit(batches.loc[target.index], target.to_numpy())
        test_dates = cases.date.iloc[test].to_numpy()
        b_test = pd.Series(test_dates).map(pd.Series(np.exp(model.predict(batches.loc[sorted(set(test_dates))])),
                                                     index=sorted(set(test_dates)))).to_numpy()
        difficulty_test[test] = b_test
        per_batch_q = {lv: pd.Series(s).groupby(cal_dates).quantile(lv, interpolation="higher").to_numpy() for lv in LEVELS}

        for lv in LEVELS:
            half = {
                "pooled_scaled": gate.conformal_quantile(s, lv) * sig_test,
                "batch_weighted": an.weighted_quantile(s, w, lv) * sig_test,
                "batch_adaptive": an.weighted_quantile(s / b_cal, w, lv) * sig_test * b_test,
                "two_level": gate.conformal_quantile(per_batch_q[lv], 0.8) * sig_test,
            }
            for method, h in half.items():
                bounds[(method, lv)][0][test] = outer[test] - h
                bounds[(method, lv)][1][test] = outer[test] + h

    result = {"protocol_sha256": data.digest(PROTOCOL), "label": "development comparison on 41 batches; not a sealed test",
              "coverage": {}}
    for (method, lv), (lo, hi) in bounds.items():
        result["coverage"].setdefault(method, {})[str(lv)] = coverage_table(cases, usable, lo, hi, lv)
    # Pre-registered success checks
    ad, bw = result["coverage"]["batch_adaptive"]["0.8"], result["coverage"]["batch_weighted"]["0.8"]
    result["success_batch_adaptive"] = {
        "batch_mean_in_band": 0.77 <= ad["batch_mean"] <= 0.83,
        "worst_fifth_gain": ad["worst_fifth"] - bw["worst_fifth"],
        "worst_fifth_ok": ad["worst_fifth"] - bw["worst_fifth"] >= 0.05,
        "width_ratio": ad["mean_width"] / bw["mean_width"],
        "width_ok": ad["mean_width"] / bw["mean_width"] <= 1.10,
    }
    result["success_batch_adaptive"]["pass"] = all(result["success_batch_adaptive"][k] for k in ["batch_mean_in_band", "worst_fifth_ok", "width_ok"])
    # How well does predicted batch difficulty track realised batch difficulty on the test batches?
    realised = pd.Series(np.abs(point - y)[usable]).groupby(cases.date.to_numpy()[usable]).median()
    predicted = pd.Series(difficulty_test[usable]).groupby(cases.date.to_numpy()[usable]).first()
    result["difficulty_rank_correlation"] = float(realised.rank().corr(predicted.rank()))
    table = cases[["case_id", "date", "target", "usable"]].assign(prediction=point, batch_difficulty=difficulty_test)
    for (method, lv), (lo, hi) in bounds.items():
        table[f"lo__{method}__{lv}"], table[f"hi__{method}__{lv}"] = lo, hi
    table.to_csv(out / "analysis/v31_interval_cases.csv", index=False)
    v3d.intake.write_json(out / "analysis/v31_intervals.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=v3d.OUT)
    args = parser.parse_args()
    r = run_study(args.out)
    for method, levels in r["coverage"].items():
        for lv, c in levels.items():
            print(f"{method:15s} {lv}: batch-mean {c['batch_mean']:.3f} worst-fifth {c['worst_fifth']:.3f} "
                  f"share>=nom-0.1 {c['share_batches_within_0.1']:.2f} width {c['mean_width']:.2f}")
    print(json.dumps(r["success_batch_adaptive"]), "difficulty rank r:", round(r["difficulty_rank_correlation"], 3))


if __name__ == "__main__":
    main()
