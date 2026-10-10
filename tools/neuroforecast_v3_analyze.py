"""NeuroForecast v3 analysis: nested selection, calibration under batch shift, trust, warning, twin, swap rule.

Reads the candidate predictions written by `neuroforecast_v3_run.py` and the locked protocol. Every
quantity used to choose anything (model, calibration quantile, sigma model) comes from inner
out-of-fold predictions on the outer training dates; outer predictions are only scored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402
import neuroforecast_v3_data as v3d  # noqa: E402
import neuroforecast_v3_run as run  # noqa: E402

OUT = v3d.OUT
LEVELS = [0.8, 0.9]
QCOLS = ["q10", "q90", "seed_sd"]


# --------------------------------------------------------------------------- loading

def load_predictions(out: Path, proto: dict, allow_missing: bool) -> dict[str, pd.DataFrame]:
    preds = {}
    for name, spec in proto["candidates"].items():
        if spec["kind"] == "ensemble":
            continue
        path = out / "predictions" / f"{name}.csv"
        if not path.exists():
            if allow_missing:
                continue
            raise FileNotFoundError(path)
        preds[name] = pd.read_csv(path)
    for name, spec in proto["candidates"].items():
        if spec["kind"] == "ensemble" and all(m in preds for m in spec["members"]):
            key = ["outer", "level", "inner", "position"]
            merged = None
            for m in spec["members"]:
                part = preds[m][key + ["pred"]].rename(columns={"pred": m})
                merged = part if merged is None else merged.merge(part, on=key, validate="one_to_one")
            merged["pred"] = merged[spec["members"]].mean(axis=1)
            merged["candidate"] = name
            preds[name] = merged[key + ["candidate", "pred"]]
    return preds


def assemble(frame: pd.DataFrame, n: int, outer: int, level: str, column: str = "pred") -> np.ndarray:
    part = frame.loc[(frame.outer == outer) & (frame.level == level)]
    values = np.full(n, np.nan)
    if part.position.duplicated().any():
        raise ValueError("A position was predicted twice at one level")
    values[part.position.to_numpy()] = part[column].to_numpy()
    return values


# --------------------------------------------------------------------------- calibration

def weighted_quantile(scores: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Conformal quantile with per-point weights (normalised); the test point's weight is the mean weight."""
    order = np.argsort(scores)
    s, w = scores[order], weights[order]
    w = w / w.sum()
    test_w = 1.0 / len(s)
    cum = np.cumsum(w) * (1 - test_w)
    idx = np.searchsorted(cum, level - 1e-12)
    return float(s[min(idx, len(s) - 1)]) if idx < len(s) else float("inf")


def date_weights(dates: np.ndarray) -> np.ndarray:
    counts = pd.Series(dates).map(pd.Series(dates).value_counts()).to_numpy(dtype=float)
    return 1.0 / counts


def sigma_model(cfg: dict):
    return run.make_model({"kind": "extra_trees"}, cfg)


def sigma_features(x: pd.DataFrame, cols: list[str], extra: dict[str, np.ndarray] | None) -> pd.DataFrame:
    frame = x[cols].copy()
    for k, v in (extra or {}).items():
        frame[k] = v
    return frame


def calibrate(name, frame, x, cases, cols, cfg, outer, train, test, inner_pred, outer_pred, inner_extra, outer_extra):
    """Cross-fitted sigma on the calibration (outer-train) positions, sigma for the test positions,
    and the four interval variants. Returns per-test-position arrays."""
    y = cases.target.to_numpy()
    usable = cases.usable.to_numpy(dtype=bool)
    cal = train[usable[train]]
    res = np.abs(inner_pred - y)
    sig_cal = np.full(len(cases), np.nan)
    feats_cal = sigma_features(x, cols, inner_extra)
    with threadpool_limits(limits=1):
        for j, dates in enumerate(v3d.interleave(cases.date.iloc[train], 4)):
            hold = cal[cases.date.iloc[cal].isin(dates).to_numpy()]
            fit = cal[~cases.date.iloc[cal].isin(dates).to_numpy()]
            m = sigma_model(cfg).fit(feats_cal.iloc[fit], np.round(res[fit], gate.RESIDUAL_DECIMALS))
            sig_cal[hold] = np.maximum(m.predict(feats_cal.iloc[hold]), gate.SIGMA_FLOOR)
        m = sigma_model(cfg).fit(feats_cal.iloc[cal], np.round(res[cal], gate.RESIDUAL_DECIMALS))
        sig_test = np.maximum(m.predict(sigma_features(x, cols, outer_extra).iloc[test]), gate.SIGMA_FLOOR)
    scores = res[cal] / sig_cal[cal]
    wts = date_weights(cases.date.iloc[cal].to_numpy())
    out = {"sigma": sig_test}
    for level in LEVELS:
        q_pool = gate.conformal_quantile(scores, level)
        q_date = weighted_quantile(scores, wts, level)
        out[f"pooled_scaled_{level}"] = q_pool * sig_test
        out[f"date_scaled_{level}"] = q_date * sig_test
    return out


def cqr(frame_inner_q10, frame_inner_q90, frame_outer_q10, frame_outer_q90, cases, train, test):
    y = cases.target.to_numpy()
    usable = cases.usable.to_numpy(dtype=bool)
    cal = train[usable[train]]
    scores = np.maximum(frame_inner_q10[cal] - y[cal], y[cal] - frame_inner_q90[cal])
    wts = date_weights(cases.date.iloc[cal].to_numpy())
    out = {}
    for level in LEVELS:
        for tag, q in [("pooled", gate.conformal_quantile(scores, level)), ("date", weighted_quantile(scores, wts, level))]:
            out[f"{tag}_cqr_{level}"] = (frame_outer_q10[test] - q, frame_outer_q90[test] + q)
    return out


def coverage(cases: pd.DataFrame, mask: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    y = cases.target.to_numpy()
    hit = (y >= low) & (y <= high)
    frame = pd.DataFrame({"date": cases.date, "hit": hit, "width": high - low}).loc[mask]
    per_date = frame.groupby("date").hit.mean()
    return {"date_mean": float(per_date.mean()), "pooled": float(frame.hit.mean()),
            "worst_quintile_dates": float(per_date.sort_values().head(max(1, len(per_date) // 5)).mean()),
            "mean_width": float(frame.width.mean()), "dates": int(len(per_date))}


# --------------------------------------------------------------------------- statistics

def paired_dates(a: pd.Series, b: pd.Series, seed: int, n: int = 10000) -> dict:
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(len(d), size=(n, len(d)))].mean(1)
    return {"mean_difference": float(d.mean()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "a_better_dates": int((d < 0).sum()), "b_better_dates": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "dates": int(len(d))}


# --------------------------------------------------------------------------- main analysis

def analyze(out: Path, allow_missing: bool) -> dict:
    run.verify_lock(out)
    proto = v3d.protocol()
    cfg = data.protocol()
    loaded = v3d.load(out)
    x, cases, twin = loaded["features"], loaded["cases"], loaded["twin"]
    cols = run.column_sets(x)
    n = len(cases)
    y = cases.target.to_numpy()
    usable = cases.usable.to_numpy(dtype=bool)
    preds = load_predictions(out, proto, allow_missing)
    order = [c for c in proto["candidates"] if c in preds]
    selectable = [c for c in order if proto["candidates"][c].get("selectable", True)]
    incumbent = proto["incumbent"]

    outer_oof = {c: np.full(n, np.nan) for c in order}
    nested = np.full(n, np.nan)
    folds = []
    splits = list(v3d.outer_splits(cases))
    for k, dates, train, test in splits:
        inner_scores = {}
        for c in order:
            inner = assemble(preds[c], n, k, "inner")
            if c in selectable:
                inner_scores[c] = v3d.macro(cases, inner, np.isin(np.arange(n), train) & usable)
            outer_oof[c][test] = assemble(preds[c], n, k, "outer")[test]
        chosen = min(inner_scores, key=lambda c: (inner_scores[c], selectable.index(c)))
        nested[test] = outer_oof[chosen][test]
        folds.append({"outer": k, "test_dates": [int(d) for d in dates], "train_cases": int(len(train)),
                      "test_cases": int(len(test)), "selected": chosen, "inner_scores": inner_scores})

    scores = {c: v3d.macro(cases, outer_oof[c], usable) for c in order}
    nested_score = v3d.macro(cases, nested, usable)
    final = min(selectable, key=lambda c: (scores[c], selectable.index(c)))
    per_date = {c: v3d.date_losses(cases, outer_oof[c], usable) for c in order}
    per_date["nested_v3"] = v3d.date_losses(cases, nested, usable)
    seed = proto["seed"]
    comparisons = {
        "nested_v3_vs_incumbent": paired_dates(per_date["nested_v3"], per_date[incumbent], seed),
        "nested_v3_vs_persistence": paired_dates(per_date["nested_v3"], per_date["persistence_day7"], seed),
        "incumbent_vs_persistence": paired_dates(per_date[incumbent], per_date["persistence_day7"], seed),
        "final_vs_incumbent_descriptive": paired_dates(per_date[final], per_date[incumbent], seed),
    }
    for c in order:
        comparisons[f"{c}_vs_incumbent"] = paired_dates(per_date[c], per_date[incumbent], seed)
    by_cohort = {c: {cohort: v3d.macro(cases, outer_oof[c], usable & (cases.cohort == cohort).to_numpy())
                     for cohort in ["archive17", "external28"]} for c in order}
    by_cohort["nested_v3"] = {cohort: v3d.macro(cases, nested, usable & (cases.cohort == cohort).to_numpy())
                              for cohort in ["archive17", "external28"]}

    # ---------------- calibration and trust: incumbent, nested v3 and the final pick
    calib, trust, abstain = {}, {}, {}
    product_sets = {"incumbent": {k: incumbent for k in range(len(splits))},
                    "nested_v3": {f["outer"]: f["selected"] for f in folds},
                    "final": {k: final for k in range(len(splits))}}
    interval_tables = {}
    for tag, chosen_by_fold in product_sets.items():
        point = np.full(n, np.nan)
        intervals = {}
        sigma = np.full(n, np.nan)
        for k, dates, train, test in splits:
            c = chosen_by_fold[k]
            inner = assemble(preds[c], n, k, "inner")
            outer = assemble(preds[c], n, k, "outer")
            inner_extra = outer_extra = q = None
            if "q10" in preds[c]:
                q = {col: (assemble(preds[c], n, k, "inner", col), assemble(preds[c], n, k, "outer", col)) for col in QCOLS}
                inner_extra = {"spread": q["q90"][0] - q["q10"][0], "seed_sd": q["seed_sd"][0]}
                outer_extra = {"spread": q["q90"][1] - q["q10"][1], "seed_sd": q["seed_sd"][1]}
            res = calibrate(c, preds[c], x, cases, cols["sigma"], cfg, k, train, test, inner, outer, inner_extra, outer_extra)
            point[test] = outer[test]
            sigma[test] = res.pop("sigma")
            for key, half in res.items():
                intervals.setdefault(key, (np.full(n, np.nan), np.full(n, np.nan)))
                intervals[key][0][test] = outer[test] - half
                intervals[key][1][test] = outer[test] + half
            if q is not None:
                for key, (lo, hi) in cqr(q["q10"][0], q["q90"][0], q["q10"][1], q["q90"][1], cases, train, test).items():
                    intervals.setdefault(key, (np.full(n, np.nan), np.full(n, np.nan)))
                    intervals[key][0][test] = lo
                    intervals[key][1][test] = hi
        calib[tag] = {key: coverage(cases, usable & np.isfinite(lo), lo, hi) for key, (lo, hi) in intervals.items()}
        error = np.abs(point - y)
        d7 = x.normalized_ns_d7.to_numpy()
        scores_t = {"sigma": sigma, "extremity": np.abs(point), "predicted_change": np.abs(point - d7),
                    "day7_magnitude": np.abs(d7)}
        trust[tag] = {s: v2.risk_coverage(v[usable], error[usable]) for s, v in scores_t.items()}
        trust[tag]["random_expected"] = float(error[usable].mean())
        keep = usable & (sigma <= np.quantile(sigma[usable], v2.RETAIN))
        abstain[tag] = {"retained_mae": float(error[keep].mean()), "declined_mae": float(error[usable & ~keep].mean()),
                        "all_mae": float(error[usable].mean()), "retained": int(keep.sum()), "declined": int((usable & ~keep).sum()),
                        "retained_date_macro": v3d.macro(cases, point, keep), "all_date_macro": v3d.macro(cases, point, usable)}
        interval_tables[tag] = (point, sigma, intervals)

    # ---------------- early warning among early-quiet conditions (20% budget per date)
    frame = cases.assign(day7=x.normalized_ns_d7.to_numpy())
    warning = {}
    for tag, (point, _, _) in interval_tables.items():
        warning[f"{tag}_abs_predicted_change"] = v2.expected_detections(frame.loc[usable], np.abs(point - frame.day7.to_numpy())[usable])
    warning["highest_dose"] = v2.expected_detections(frame.loc[usable], frame.dose.to_numpy(dtype=float)[usable])
    warning["day7_magnitude"] = v2.expected_detections(frame.loc[usable], np.abs(frame.day7.to_numpy())[usable])

    # ---------------- twin (functional profile) for any candidate that writes twin columns
    twin_results = {}
    for c in order:
        tcols = [col for col in preds[c].columns if col.startswith("twin__")]
        if not tcols:
            continue
        part = preds[c].loc[preds[c].level == "outer"].set_index("position")
        res = {}
        for col in tcols:
            name = col[len("twin__"):]
            truth = twin[name].to_numpy()
            pred = np.full(n, np.nan)
            pred[part.index.to_numpy()] = part[col].to_numpy()
            if name.startswith("normalized_ns"):
                base = x.normalized_ns_d7.to_numpy()
            else:
                field = name.split("__")[1]
                base = x.get(f"contrast__{field}__d7", pd.Series(np.full(n, np.nan))).to_numpy()
            ok = usable & np.isfinite(truth) & np.isfinite(pred) & np.isfinite(base)
            frame_t = cases.loc[ok, ["date", "identity"]]
            res[name] = {
                "model": float(gate.date_losses(frame_t.assign(target=truth[ok], prediction=pred[ok])).mean()),
                "persistence": float(gate.date_losses(frame_t.assign(target=truth[ok], prediction=base[ok])).mean()),
                "cases": int(ok.sum()),
            }
        twin_results[c] = res

    # ---------------- swap rule
    rule = proto["swap_rule"]
    cmp = comparisons["nested_v3_vs_incumbent"]
    inc = scores[incumbent]
    checks = {
        "relative_gain": 1 - nested_score / inc,
        "gain_ok": nested_score <= (1 - rule["min_relative_gain"]) * inc,
        "interval_ok": cmp["ci95"][1] < 0,
        "dates_ok": cmp["a_better_dates"] >= math.ceil(rule["min_date_share"] * cmp["dates"]),
        "trust_ok": trust["final"]["sigma"] <= (1 + rule["max_trust_aurc_increase"]) * trust["incumbent"]["sigma"],
        "final_is_learned_change": final != incumbent,
    }
    checks["swap"] = all(checks[k] for k in ["gain_ok", "interval_ok", "dates_ok", "trust_ok", "final_is_learned_change"])

    results = {
        "cases": int(n), "usable": int(usable.sum()), "dates": int(cases.date.nunique()),
        "dates_with_usable": int(cases.loc[usable].date.nunique()), "identities": int(cases.identity.nunique()),
        "candidates_scored": order, "selectable": selectable, "incumbent": incumbent,
        "outer_date_macro_mae": scores, "by_cohort": by_cohort, "nested_v3_date_macro_mae": nested_score,
        "folds": folds, "final_pick": final, "comparisons": comparisons,
        "calibration": calib, "trust_aurc": trust, "abstention_at_80pct_retention": abstain,
        "warning_early_quiet_20pct": warning, "twin": twin_results, "swap_rule": checks,
    }
    case_table = cases[["case_id", "date", "identity", "dose", "cohort", "outer_fold", "target", "usable"]].copy()
    case_table["day7"] = x.normalized_ns_d7.to_numpy()
    for c in order:
        case_table[f"pred__{c}"] = outer_oof[c]
    case_table["pred__nested_v3"] = nested
    for tag, (point, sigma, intervals) in interval_tables.items():
        case_table[f"sigma__{tag}"] = sigma
        for key in ["date_scaled_0.8", "pooled_scaled_0.8"]:
            case_table[f"lo__{tag}__{key}"], case_table[f"hi__{tag}__{key}"] = intervals[key]
    target = out / "analysis"
    target.mkdir(exist_ok=True)
    case_table.to_csv(target / "outer_case_table.csv", index=False)
    v3d.intake.write_json(target / "results.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    r = analyze(args.out, args.allow_missing)
    print(json.dumps({k: r[k] for k in ["outer_date_macro_mae", "nested_v3_date_macro_mae", "final_pick", "swap_rule"]}, indent=2))
    print(json.dumps({"selected_by_fold": [f["selected"] for f in r["folds"]], "calibration": r["calibration"]}, indent=2))


if __name__ == "__main__":
    main()
