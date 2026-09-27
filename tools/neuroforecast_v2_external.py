"""Score the sealed external dates exactly once (Arm A: frozen v1 tool; Arm B: locked v2), then audit.

Order of operations, enforced here:
  1. `freeze`   records this file's hash beside the development lock (before any external outcome exists)
  2. `external --unseal`   verifies every hash, builds external targets, scores both arms once
  3. `audit`    replays the saved v2 models and checks separation of development and external rows

Wording for each criterion is taken from the protocol and printed whether it passes or fails.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_triage as triage_tool  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402

MIN_DATE_CASES = 10
SEED = 20260927


def verify_lock(out: Path) -> dict:
    lock = json.loads((out / "development_lock.json").read_text(encoding="utf-8"))
    problems = []
    if data.digest(v2.PROTOCOL) != lock["protocol_sha256"]:
        problems.append("protocol")
    current = v2.code_hashes()
    problems += [f"code:{k}" for k, v in lock["code"].items() if current.get(k) != v]
    for key, path in [("prepared_manifest_sha256", out / "prepared/manifest.json"),
                      ("development_results_sha256", out / "development_results.json"),
                      ("dev_oof_sha256", out / "dev_oof_predictions.csv")]:
        if data.digest(path) != lock[key]:
            problems.append(key)
    problems += [f"model:{k}" for k, v in lock["models"].items() if data.digest(out / "models" / k) != v]
    if problems:
        raise ValueError(f"The development lock does not match: {problems}")
    return lock


def arm_a_hashes() -> dict:
    """The shipped v1 tool and bundle that Arm A must run unchanged."""
    bundle = triage_tool.DEFAULT_BUNDLE
    files = [Path(triage_tool.__file__), triage_tool.DEFAULT_LOCK, triage_tool.DEFAULT_RESERVE,
             bundle / "final_extra_trees_relative_day7.joblib", bundle / "final_extra_trees_activity_coordination.joblib", bundle / "final_sigma.joblib"]
    return {str(p): data.digest(p) for p in files}


def freeze(out: Path) -> None:
    path = out / "external_code_lock.json"
    if path.exists():
        raise ValueError("External code is already frozen")
    lock = verify_lock(out)
    record = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "external_code_sha256": data.digest(Path(__file__)),
              "development_lock_sha256": data.digest(out / "development_lock.json"), "locked": lock["locked"],
              "arm_a_unchanged_v1": arm_a_hashes(), "external_outcomes_read": False}
    v2.write(path, record)
    print(json.dumps(record, indent=2))


# --------------------------------------------------------------------------- scoring helpers

def date_table(frame: pd.DataFrame, a: str, b: str) -> dict:
    la = gate.date_losses(frame.assign(prediction=frame[a]))
    lb = gate.date_losses(frame.assign(prediction=frame[b]))
    counts = frame.groupby("date").size()
    eligible = counts[counts >= MIN_DATE_CASES].index
    wins = int((la.loc[eligible] < lb.loc[eligible]).sum())
    rng = np.random.default_rng(SEED)
    index = rng.integers(len(la), size=(10000, len(la)))
    diff = la.to_numpy()[index].mean(axis=1) - lb.to_numpy()[index].mean(axis=1)
    return {"a": a, "b": b, "a_date_macro": float(la.mean()), "b_date_macro": float(lb.mean()), "dates": int(len(la)),
            "dates_with_at_least_10": int(len(eligible)), "a_wins_on_those": wins,
            "paired_difference_a_minus_b_interval_95": np.quantile(diff, [0.025, 0.975]).tolist(),
            "per_date": {str(d): {"cases": int(counts[d]), a: float(la[d]), b: float(lb[d])} for d in la.index}}


def retained_stats(frame: pd.DataFrame, keep: np.ndarray, column: str) -> dict:
    error = np.abs(frame[column].to_numpy() - frame.target.to_numpy())
    return {"all_case_mae": float(error.mean()), "retained_case_mae": float(error[keep].mean()),
            "declined": int((~keep).sum()), "relative_reduction": float(1 - error[keep].mean() / error.mean())}


def extremity_keep(frame: pd.DataFrame, column: str, declined: int) -> np.ndarray:
    order = np.argsort(-np.abs(frame[column].to_numpy()), kind="stable")
    keep = np.ones(len(frame), dtype=bool)
    keep[order[:declined]] = False
    return keep


def coverage(frame: pd.DataFrame, low: str, high: str) -> dict:
    inside = (frame.target >= frame[low]) & (frame.target <= frame[high])
    return {"pooled": float(inside.mean()), "mean_width": float((frame[high] - frame[low]).mean()),
            "per_date": {str(d): float(inside.loc[g.index].mean()) for d, g in frame.groupby("date")}}


def verdict(passed: bool | None, spec: dict) -> dict:
    if passed is None:
        return {"result": "NOT TESTABLE"}
    return {"result": "PASS" if passed else "FAIL", "statement": spec["if_pass"] if passed else spec["if_fail"]}


# --------------------------------------------------------------------------- stage

def external(out: Path, unseal: bool) -> None:
    if not unseal:
        raise SystemExit("External scoring requires --unseal and runs once.")
    target_dir = out / "external"
    if (target_dir / "results.json").exists():
        raise ValueError("The external dates have already been scored")
    lock = verify_lock(out)
    frozen = json.loads((out / "external_code_lock.json").read_text(encoding="utf-8"))
    if frozen["external_code_sha256"] != data.digest(Path(__file__)) or frozen["development_lock_sha256"] != data.digest(out / "development_lock.json"):
        raise ValueError("External code or development lock changed after freezing")
    if frozen["arm_a_unchanged_v1"] != arm_a_hashes():
        raise ValueError("The v1 tool or bundle changed after freezing; Arm A would not be a replication")
    protocol = json.loads(v2.PROTOCOL.read_text(encoding="utf-8"))
    target_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    cfg = v2.v2_config()
    raw = v2.release_frame()
    ext_raw = raw.loc[~raw.date.isin(v2.DEV_DATES)]
    features, cases = v2.load(out, "ext")
    index = pd.MultiIndex.from_frame(cases[["date", "identity", "dose"]])

    # ---- unsealing: the only place external day-12 values are read
    targets = data.build_targets(ext_raw, cfg)
    quality = gate.outcome_quality(ext_raw, cfg)
    cases["target"] = targets.reindex(index).to_numpy()
    cases["day12_ctrl_median_ns"] = quality.reindex(index).to_numpy()
    cases["usable"] = cases.target.notna() & (cases.day12_ctrl_median_ns >= v2.USABLE_MIN)

    # ---- Arm A: the shipped v1 triage function and bundle, unchanged
    bundle = triage_tool.load_bundle(triage_tool.DEFAULT_BUNDLE, triage_tool.DEFAULT_LOCK, triage_tool.DEFAULT_RESERVE)
    table_a, summary_a = triage_tool.triage(ext_raw.loc[ext_raw.DIV.isin([5, 7])].copy(), bundle)
    a = cases.merge(table_a[["case_id", "forecast_log2", "forecast_low", "forecast_high", "persistence_log2", "difficulty",
                             "forecast_trusted", "quality_low_reference_activity"]], on="case_id", how="left", validate="one_to_one")
    a["prediction"] = a.forecast_log2
    if a.loc[a.target.notna(), "prediction"].isna().any():
        raise ValueError("The v1 tool produced no forecast for a scored external condition")

    # ---- Arm B: locked v2 components, predictions only
    locked = lock["locked"]
    models = {name: joblib.load(out / "models" / f"final_{name}.joblib") for name in ["v1_refit", "small_refit", "g", "g_batch", "sigma_v2"]}
    with threadpool_limits(limits=1):
        base = {name: models[name]["model"].predict(features[models[name]["columns"]]) for name in ["v1_refit", "small_refit", "g", "g_batch"]}
        sigma = np.maximum(models["sigma_v2"]["model"].predict(features[models["sigma_v2"]["columns"]]), gate.SIGMA_FLOOR)
    params = dict(locked["final_params"])
    params["blend"] = tuple(params["blend"])
    candidates = v2.combine(base, features, params)
    b = cases.copy()
    b["prediction"] = candidates[locked["product"]]
    b["persistence"] = features.normalized_ns_d7.to_numpy()
    b["v1_refit"] = base["v1_refit"]
    trust = v2.trust_scores(locked["trust"], features, b.prediction.to_numpy(), sigma)
    b["trust_score"] = trust
    b["declined"] = trust > locked["trust_threshold"]
    scale = sigma if locked["interval_variant"] == "scaled" else np.ones(len(b))
    for level in v2.LEVELS:
        half = locked["conformal_quantiles"][str(level)] * scale
        b[f"low_{level}"], b[f"high_{level}"] = b.prediction - half, b.prediction + half
    b["flag_low_reference_activity"] = np.sinh(features["rel__plate_ctrl_median_ns_d7"].to_numpy()) < 1

    a.to_csv(target_dir / "arm_a_cases.csv", index=False)
    b.to_csv(target_dir / "arm_b_cases.csv", index=False)

    spec_a, spec_b = protocol["arm_A_v1_replication"]["criteria"], protocol["arm_B_v2"]["criteria"]
    results = {"unsealed_utc": started, "lock": lock["locked"], "cohort": {}, "arm_A": {}, "arm_B": {}}
    cohorts = {
        "primary": (a.primary & a.usable).to_numpy(),
        "all_scored_primary": (a.primary & a.target.notna()).to_numpy(),
        "secondary_shared_chemicals": (~a.primary & a.usable).to_numpy(),
    }
    results["cohort"] = {k: {"conditions": int(m.sum()), "dates": int(a.loc[m, "date"].nunique()), "chemicals": int(a.loc[m, "identity"].nunique())} for k, m in cohorts.items()}
    results["cohort"]["input_eligible"] = int(len(a))
    results["cohort"]["without_target"] = int(a.target.isna().sum())
    results["cohort"]["unusable_reference"] = int((a.target.notna() & ~a.usable).sum())

    for label, mask in cohorts.items():
        if not mask.any():
            continue
        fa = a.loc[mask].reset_index(drop=True)
        fb = b.loc[mask].reset_index(drop=True)
        fa = fa.assign(persistence=fa.day7)
        # Arm A
        A = {}
        A["forecast_vs_persistence"] = date_table(fa, "prediction", "persistence")
        A["forecast_vs_control_reference"] = date_table(fa.assign(control=0.0), "prediction", "control")
        keep = fa.forecast_trusted.to_numpy(dtype=bool)
        A["abstention_v1"] = retained_stats(fa, keep, "prediction")
        A["abstention_persistence"] = retained_stats(fa, keep, "persistence")
        A["extremity_rule_same_count"] = retained_stats(fa, extremity_keep(fa, "prediction", int((~keep).sum())), "prediction")
        flagged = fa.quality_low_reference_activity.to_numpy(dtype=bool)
        A["low_reference_activity"] = {"flagged": int(flagged.sum()), "dates_flagged": sorted(int(d) for d in fa.loc[flagged, "date"].unique())}
        if flagged.any() and (~flagged).any():
            for col in ["prediction", "persistence"]:
                err = np.abs(fa[col] - fa.target).to_numpy()
                A["low_reference_activity"][col] = {"flagged_mae": float(err[flagged].mean()), "unflagged_mae": float(err[~flagged].mean())}
        work = fa.assign(prediction=fa.prediction)
        A["warning_v1"] = v2.expected_detections(work, np.abs(fa.prediction.to_numpy()))
        A["warning_dose"] = v2.expected_detections(work, fa.dose.to_numpy(dtype=float))
        A["warning_day7_random_ties"] = v2.expected_detections(work, np.abs(fa.day7.to_numpy()))
        A["warning_v1_direction"] = v2.direction_agreement(work, np.abs(fa.prediction.to_numpy()), (fa.prediction - fa.day7).to_numpy())
        A["coverage_scaled_0.8"] = coverage(fa, "forecast_low", "forecast_high")
        results["arm_A"][label] = A
        # Arm B
        B = {}
        B["product"] = locked["product"]
        B["forecast_vs_persistence"] = date_table(fb, "prediction", "persistence")
        B["forecast_vs_arm_a_v1"] = date_table(fb.assign(v1=fa.prediction.to_numpy()), "prediction", "v1")
        B["forecast_vs_v1_refit"] = date_table(fb, "prediction", "v1_refit")
        keep_b = ~fb.declined.to_numpy(dtype=bool)
        B["abstention_product"] = retained_stats(fb, keep_b, "prediction")
        B["abstention_persistence"] = retained_stats(fb, keep_b, "persistence")
        err_b = np.abs(fb.prediction - fb.target).to_numpy()
        B["aurc"] = {"locked_trust": v2.risk_coverage(fb.trust_score.to_numpy(), err_b),
                     "extremity": v2.risk_coverage(np.abs(fb.prediction.to_numpy()), err_b),
                     "random_expected": float(err_b.mean())}
        warn_score = v2.warning_score(locked["warning"], fb.assign(day7=fb.day7), fb.prediction.to_numpy(), fb.v1_refit.to_numpy())
        B["warning_locked"] = v2.expected_detections(fb, warn_score)
        B["warning_dose"] = v2.expected_detections(fb, fb.dose.to_numpy(dtype=float))
        B["warning_direction"] = v2.direction_agreement(fb, warn_score, (fb.prediction - fb.day7).to_numpy())
        B["coverage_0.8"] = coverage(fb, "low_0.8", "high_0.8")
        B["coverage_0.9"] = coverage(fb, "low_0.9", "high_0.9")
        flagged_b = fb.flag_low_reference_activity.to_numpy(dtype=bool)
        B["low_reference_activity_flagged"] = int(flagged_b.sum())
        results["arm_B"][label] = B

    # ---- Decision D on primary-cohort samples (unit = EPA sample), unsealed here
    primary_cases = b.loc[b.primary & b.target.notna()]
    spids = set(primary_cases.spid.dropna())
    bio = v2.bioactivity(["div7.hitsum", "auc.hitsum"], spids)
    units = v2.d_units(primary_cases, primary_cases.prediction.to_numpy(), primary_cases.declined.to_numpy(dtype=bool),
                       primary_cases.flag_low_reference_activity.to_numpy(dtype=bool)).join(bio, how="inner")
    reference = units["auc.hitsum"] >= 1
    d0 = pd.Series(np.where(units["div7.hitsum"] >= 1, "active", "inactive"), index=units.index)
    D = {"units": int(len(units)), "reference_active": int(reference.sum()), "baseline_D0": v2.d_score(units, d0, reference)}
    if locked["decision_D"]:
        calls = v2.d_policy(units, locked["decision_D"]["h"], locked["decision_D"]["c"])
        D["policy_D1"] = v2.d_score(units, calls, reference)
        D["secondary_reference_auc_hitsum_ge_3"] = v2.d_score(units, calls, units["auc.hitsum"] >= 3)
        # Descriptive ablation, written before unsealing: the same thresholds on EPA's day-7 call alone,
        # without the forecast, trust and quality gate on the early inactive call.
        plain = pd.Series(np.select([units["div7.hitsum"] >= locked["decision_D"]["h"], units["div7.hitsum"] == 0], ["active", "inactive"], "continue"), index=units.index)
        D["ablation_without_forecast_gate"] = v2.d_score(units, plain, reference)
        units.assign(call=calls, reference_active=reference).to_csv(target_dir / "decision_units.csv")
    results["decision_D"] = D

    # ---- Criteria on the primary cohort, with the protocol's wording
    pa, pb = results["arm_A"]["primary"], results["arm_B"]["primary"]
    fvp = pa["forecast_vs_persistence"]
    crit = {}
    crit["A1"] = verdict(fvp["a_date_macro"] < fvp["b_date_macro"] and fvp["a_wins_on_those"] > fvp["dates_with_at_least_10"] / 2, spec_a["A1_forecast_vs_persistence"])
    crit["A2"] = verdict(pa["abstention_v1"]["relative_reduction"] >= 0.20 and pa["abstention_persistence"]["relative_reduction"] >= 0.10, spec_a["A2_abstention"])
    crit["A2b"] = verdict(pa["abstention_v1"]["retained_case_mae"] < pa["extremity_rule_same_count"]["retained_case_mae"], spec_a["A2b_difficulty_model_vs_extremity_rule"])
    lra = pa["low_reference_activity"]
    crit["A3"] = verdict(None if "prediction" not in lra else (lra["prediction"]["flagged_mae"] >= 1.5 * lra["prediction"]["unflagged_mae"] and lra["persistence"]["flagged_mae"] >= 1.5 * lra["persistence"]["unflagged_mae"]), spec_a["A3_quality_flag"])
    w = pa["warning_v1"]
    crit["A4"] = verdict(w["expected_detected"] > 2 * w["random_expected"] and w["expected_detected"] > pa["warning_dose"]["expected_detected"], spec_a["A4_early_warning"])
    crit["A5"] = verdict(pa["coverage_scaled_0.8"]["pooled"] >= 0.70, spec_a["A5_interval_coverage"])
    bvp = pb["forecast_vs_persistence"]
    crit["B1"] = verdict(None if locked["product"] == "persistence_day7" else (bvp["a_date_macro"] < bvp["b_date_macro"] and bvp["a_wins_on_those"] > bvp["dates_with_at_least_10"] / 2), spec_b["B1_forecast_vs_persistence"])
    crit["B1b"] = {"product_below_arm_a_v1": pb["forecast_vs_arm_a_v1"]["a_date_macro"] < pb["forecast_vs_arm_a_v1"]["b_date_macro"],
                   "product_below_v1_refit": pb["forecast_vs_v1_refit"]["a_date_macro"] < pb["forecast_vs_v1_refit"]["b_date_macro"],
                   "note": spec_b["B1b_forecast_vs_v1"]["note"]}
    au = pb["aurc"]
    crit["B2"] = verdict(au["locked_trust"] < au["random_expected"] and (locked["trust"] == "extremity" or au["locked_trust"] < au["extremity"]), spec_b["B2_trust"])
    wb = pb["warning_locked"]
    crit["B3"] = verdict(wb["expected_detected"] > 2 * wb["random_expected"] and wb["expected_detected"] > pb["warning_dose"]["expected_detected"], spec_b["B3_warning"])
    if locked["decision_D"]:
        d1, d0s = D["policy_D1"], D["baseline_D0"]
        crit["B4"] = verdict((d1["early_agreement"] or 0) >= 0.90 and d1["early_share"] >= 0.25 and (d1["early_agreement"] or 0) > (d0s["early_agreement"] or 0), spec_b["B4_decision_D"])
    else:
        crit["B4"] = {"result": "NOT LOCKED", "statement": "No development operating point met 0.95 agreement; D is descriptive only."}
    crit["B5"] = verdict(pb["coverage_0.8"]["pooled"] >= 0.75, spec_b["B5_interval_coverage"])
    results["criteria"] = crit
    results["arm_A_handicap"] = protocol["arm_A_v1_replication"]["disclosed_handicaps"]
    results["arm_A_tool_summary_batches"] = summary_a["batches"]
    results["external_code_sha256"] = data.digest(Path(__file__))
    v2.write(target_dir / "results.json", results)
    print(json.dumps({"cohort": results["cohort"], "criteria": crit}, indent=2, default=str))


def audit(out: Path) -> None:
    lock = verify_lock(out)
    features, cases = v2.load(out, "ext")
    dev_features, dev_cases = v2.load(out, "dev")
    checks = {"external_case_ids_absent_from_development": not set(cases.case_id) & set(dev_cases.case_id),
              "external_dates_absent_from_development": not set(cases.date) & set(v2.DEV_DATES)}
    replays = {}
    with threadpool_limits(limits=1):
        for name in ["v1_refit", "small_refit", "g", "g_batch", "sigma_v2"]:
            saved = joblib.load(out / "models" / f"final_{name}.joblib")
            first = saved["model"].predict(features[saved["columns"]])
            again = joblib.load(out / "models" / f"final_{name}.joblib")["model"].predict(features[saved["columns"]])
            replays[name] = float(np.max(np.abs(first - again)))
            checks[f"{name}_feature_count_matches_saved_columns"] = int(getattr(saved["model"][-1], "n_features_in_", 0)) == len(saved["columns"])
    b = pd.read_csv(out / "external/arm_b_cases.csv", dtype={"identity": str, "spid": str})
    locked = lock["locked"]
    base = {}
    with threadpool_limits(limits=1):
        for name in ["v1_refit", "small_refit", "g", "g_batch"]:
            saved = joblib.load(out / "models" / f"final_{name}.joblib")
            base[name] = saved["model"].predict(features[saved["columns"]])
    params = dict(locked["final_params"])
    params["blend"] = tuple(params["blend"])
    recomputed = v2.combine(base, features, params)[locked["product"]]
    checks["saved_external_forecasts_replay"] = float(np.max(np.abs(recomputed - b.prediction.to_numpy())))
    report = {"checks": checks, "replay_max_difference": replays, "audited_utc": datetime.now(timezone.utc).isoformat()}
    v2.write(out / "external/audit.json", report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["freeze", "external", "audit"])
    parser.add_argument("--out", type=Path, default=v2.OUT)
    parser.add_argument("--unseal", action="store_true")
    args = parser.parse_args()
    {"freeze": lambda: freeze(args.out), "external": lambda: external(args.out, args.unseal), "audit": lambda: audit(args.out)}[args.stage]()


if __name__ == "__main__":
    main()
