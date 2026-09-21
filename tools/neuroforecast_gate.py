"""Locked NeuroForecast validation gate: development analyses, one reserve scoring, and audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import neuroforecast_benchmark as bench
import neuroforecast_data as data
import neuroforecast_representation as rep

PROTOCOL = data.ROOT / "experiments/neuroforecast_gate_protocol.json"
DEFAULT_OUT = Path("E:/kaggle/AI4S/experiments/neuroforecast_gate_v1_1")
RELIABILITY = [
    "rel__rep_count_d7",
    "rel__rep_sd_log2_ns_d7",
    "rel__plate_ctrl_count_d7",
    "rel__plate_ctrl_median_ns_d7",
    "rel__plate_ctrl_zero_ns_frac_d7",
    "rel__plate_ctrl_iqr_ratio_d7",
    "rel__plate_ctrl_silent_frac_d7",
]
TRIVIAL = ["control_reference", "persistence_day7", "linear_trend"]
SIGMA_FLOOR = 0.05
RESIDUAL_DECIMALS = 9
PLATE = ["date", "Plate.SN"]


def plan() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def code_hashes() -> dict:
    files = [Path(__file__), Path(data.__file__), Path(bench.__file__), Path(rep.__file__), data.PROTOCOL_PATH, rep.PROTOCOL, PROTOCOL]
    return {p.name: data.digest(p) for p in files}


def case_ids(index: pd.MultiIndex) -> list[str]:
    return [f"{d}|{i}|{c:.12g}" for d, i, c in index]


# --------------------------------------------------------------------------- inputs

def reliability_indicators(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Day-7 replicate and plate-control reliability indicators; nothing later than day 7 is read."""
    day = raw.loc[raw.DIV == 7]
    controls = day.loc[day.dose == 0]
    by_plate = controls.groupby(PLATE)
    ns = by_plate["ns.n"]
    plate = pd.DataFrame({
        "plate_ctrl_count_d7": ns.count().astype(float),
        "plate_ctrl_median_ns_d7": np.arcsinh(ns.median()),
        "plate_ctrl_zero_ns_frac_d7": ns.agg(lambda x: float((x.dropna() == 0).mean()) if x.notna().any() else np.nan),
        "plate_ctrl_iqr_ratio_d7": ns.agg(lambda x: (x.quantile(0.75) - x.quantile(0.25)) / (x.median() + 1.0)),
        "plate_ctrl_silent_frac_d7": by_plate["nAE"].agg(lambda x: float((x.dropna() == 0).mean()) if x.notna().any() else np.nan),
    })
    treated = day.loc[day.eligible_identity & (day.dose > 0)].join(ns.median().rename("control_median"), on=PLATE, validate="many_to_one")
    treated = treated.join(plate, on=PLATE, validate="many_to_one")
    pseudo = cfg["target_pseudocount"]
    treated["log2_relative"] = np.log2((treated["ns.n"] + pseudo) / (treated["control_median"] + pseudo))
    by_case = treated.groupby(data.KEY, sort=True)
    out = pd.DataFrame({"rep_count_d7": by_case.size().astype(float), "rep_sd_log2_ns_d7": by_case.log2_relative.std(ddof=1)})
    for column in plate.columns:
        out[column] = by_case[column].mean()
    out["plates_spanned"] = by_case["Plate.SN"].nunique()
    return out.add_prefix("rel__")


def outcome_quality(raw: pd.DataFrame, cfg: dict) -> pd.Series:
    """Day-12 plate control median per case (minimum over its wells). Reporting only; never a feature."""
    late = raw.loc[raw.DIV == cfg["target_day"]]
    control = late.loc[late.dose == 0].groupby(PLATE)["ns.n"].median().rename("day12_ctrl_median_ns")
    treated = late.loc[late.eligible_identity & (late.dose > 0)].join(control, on=PLATE, validate="many_to_one")
    return treated.groupby(data.KEY).day12_ctrl_median_ns.min()


def assemble(inputs: pd.DataFrame, raw_subset: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rel = reliability_indicators(raw_subset, cfg).reindex(inputs.index)
    spanned = rel.pop("rel__plates_spanned")
    if spanned.isna().any() or rel[RELIABILITY].isna().any().any():
        raise ValueError("Reliability indicators are missing for eligible cases")
    full = inputs.join(rel[RELIABILITY], validate="one_to_one")
    full.attrs["plates_spanned"] = spanned
    return full


def candidate_frames(full: pd.DataFrame, cfg: dict) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    v1 = full[[c for c in full.columns if not c.startswith("rel__")]]
    transformed = rep.representation(v1, cfg)
    combined = transformed.join(full[RELIABILITY], validate="one_to_one")
    frames = {}
    for name in ["extra_trees_all", "ridge_all", "extra_trees_target_only", "ridge_target_only"]:
        frames[name] = (v1, bench.columns_for(name, v1))
    for name in ["extra_trees_relative", "ridge_relative", "extra_trees_activity_coordination", "extra_trees_relative_day7"]:
        frames[name] = (transformed, rep.selected_columns(name, transformed))
    frames["extra_trees_relative_day7_reliability"] = (combined, rep.selected_columns("extra_trees_relative_day7", transformed) + RELIABILITY)
    frames["sigma"] = (combined, None)
    return frames


def trivial_prediction(name: str, v1: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    d7 = v1.iloc[idx].normalized_ns_d7.to_numpy()
    if name == "control_reference":
        return np.zeros(len(idx))
    if name == "persistence_day7":
        return d7
    if name == "linear_trend":
        return d7 + 2.5 * (d7 - v1.iloc[idx].normalized_ns_d5.to_numpy())
    raise KeyError(name)


def fit_predict(name: str, frames: dict, train: np.ndarray, test: np.ndarray, y: np.ndarray, cfg: dict):
    if name in TRIVIAL:
        return trivial_prediction(name, frames["extra_trees_all"][0], test), None
    frame, columns = frames[name]
    model = single_threaded(bench.fitted_model(name, cfg))
    model.fit(frame.iloc[train][columns], y[train])
    return model.predict(frame.iloc[test][columns]), (model, columns)


def single_threaded(model):
    """Same trees and seeds as the frozen configuration; a fixed summation order makes predictions bit-reproducible."""
    if "extratreesregressor" in model.named_steps:
        model.set_params(extratreesregressor__n_jobs=1)
    return model


def fit_sigma(frames: dict, columns: list[str], idx: np.ndarray, abs_residual: np.ndarray, cfg: dict):
    frame = frames["sigma"][0]
    model = single_threaded(bench.fitted_model("extra_trees_sigma", cfg))
    # Rounding removes the dependence of split choices on floating-point summation order in the residuals.
    model.fit(frame.iloc[idx][columns], np.round(abs_residual, RESIDUAL_DECIMALS))
    return model


def sigma_predict(model, frames: dict, columns: list[str], idx: np.ndarray) -> np.ndarray:
    return np.maximum(model.predict(frames["sigma"][0].iloc[idx][columns]), SIGMA_FLOOR)


def sigma_columns(product: str, frames: dict) -> list[str]:
    return list(dict.fromkeys(frames[product][1] + RELIABILITY))


# --------------------------------------------------------------------------- metrics

def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample split-conformal quantile for a nominal coverage level such as 0.8."""
    ordered = np.sort(np.asarray(scores, dtype=float))
    k = math.ceil((len(ordered) + 1) * level)
    return float(ordered[min(k, len(ordered)) - 1])


def date_losses(frame: pd.DataFrame) -> pd.Series:
    error = abs(frame.prediction - frame.target)
    chemical = error.groupby([frame.date, frame.identity]).mean()
    return chemical.groupby(level=0).mean()


def macro_mae(frame: pd.DataFrame) -> float:
    return float(date_losses(frame).mean())


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    index = rng.integers(len(a), size=(10000, len(a)))
    gain = 1 - a[index].mean(axis=1) / b[index].mean(axis=1)
    return {"relative_mae_reduction": float(1 - a.mean() / b.mean()), "descriptive_paired_date_interval_95": np.quantile(gain, [0.025, 0.975]).tolist(), "dates_improved": int((a < b).sum()), "date_groups": int(len(a))}


def warning_metrics(frame: pd.DataFrame, budgets: list[float]) -> list[dict]:
    """Rank within each date by score and flag the top ceil(budget * n); frame needs date, score, positive, quiet."""
    records = []
    for subgroup in ["all", "early_quiet"]:
        for budget in budgets:
            per_date, expected_random = [], 0.0
            for date, group in frame.groupby("date", sort=True):
                if subgroup == "early_quiet":
                    group = group.loc[group.quiet]
                n = len(group)
                k = math.ceil(budget * n) if n else 0
                order = np.argsort(-group.score.to_numpy(), kind="stable")
                flagged = np.zeros(n, dtype=bool)
                flagged[order[:k]] = True
                positive = group.positive.to_numpy(dtype=bool)
                expected_random += positive.sum() * (k / n if n else 0.0)
                per_date.append({"date": int(date), "cases": n, "positives": int(positive.sum()), "flagged": int(k), "detected": int((flagged & positive).sum()), "false_alerts": int((flagged & ~positive).sum()), "missed": int((~flagged & positive).sum())})
            pooled: dict[str, object] = {key: int(sum(d[key] for d in per_date)) for key in ["cases", "positives", "flagged", "detected", "false_alerts", "missed"]}
            pooled.update(recall=pooled["detected"] / pooled["positives"] if pooled["positives"] else None, precision=pooled["detected"] / pooled["flagged"] if pooled["flagged"] else None, alerts_per_detection=pooled["flagged"] / pooled["detected"] if pooled["detected"] else None, random_expected_recall=expected_random / pooled["positives"] if pooled["positives"] else None)
            records.append({"subgroup": subgroup, "budget": budget, "pooled": pooled, "per_date": per_date})
    return records


def threshold_rule(frame: pd.DataFrame) -> dict:
    alert = abs(frame.prediction) >= 1
    positive = abs(frame.target) >= 1
    quiet = abs(frame.day7) < 0.5
    def block(mask):
        return {"cases": int(mask.sum()), "positives": int((positive & mask).sum()), "detected": int((alert & positive & mask).sum()), "false_alerts": int((alert & ~positive & mask).sum()), "missed": int((~alert & positive & mask).sum())}
    return {"all": block(pd.Series(True, index=frame.index)), "early_quiet": block(quiet)}


def coverage_table(frame: pd.DataFrame, levels: list[float]) -> dict:
    result = {}
    for variant in ["symmetric", "scaled"]:
        for level in levels:
            lower, upper = frame[f"{variant}_lower_{level}"], frame[f"{variant}_upper_{level}"]
            inside = (frame.target >= lower) & (frame.target <= upper)
            result[f"{variant}_{level}"] = {"pooled_observed_coverage": float(inside.mean()), "mean_width": float((upper - lower).mean()), "per_date": {str(d): {"cases": int(len(g)), "observed_coverage": float(inside.loc[g.index].mean())} for d, g in frame.groupby("date")}}
    return result


def abstention_table(frame: pd.DataFrame, threshold: float, level: float) -> dict:
    retained = frame.sigma <= threshold
    error = abs(frame.prediction - frame.target)
    inside = (frame.target >= frame[f"scaled_lower_{level}"]) & (frame.target <= frame[f"scaled_upper_{level}"])
    return {"sigma_threshold": threshold, "retained_fraction": float(retained.mean()), "retained_cases": int(retained.sum()), "abstained_cases": int((~retained).sum()), "all_case_mae": float(error.mean()), "retained_case_mae": float(error.loc[retained].mean()) if retained.any() else None, "abstained_case_mae": float(error.loc[~retained].mean()) if (~retained).any() else None, "retained_observed_coverage": float(inside.loc[retained].mean()) if retained.any() else None, "retained_macro_mae": macro_mae(frame.loc[retained]) if retained.any() else None}


def flag_table(frame: pd.DataFrame, thresholds: dict) -> tuple[dict, pd.DataFrame]:
    flags = {
        "low_reference_activity": np.sinh(frame["rel__plate_ctrl_median_ns_d7"]) < 1,
        "few_controls": frame["rel__plate_ctrl_count_d7"] < 4,
        "replicates_disagree": frame["rel__rep_sd_log2_ns_d7"] > thresholds["rep_sd_log2_ns_d7_p90"],
        "controls_disagree": frame["rel__plate_ctrl_iqr_ratio_d7"] > thresholds["plate_ctrl_iqr_ratio_d7_p90"],
    }
    error = abs(frame.prediction - frame.target)
    result = {}
    for name, mask in flags.items():
        mask = mask.to_numpy(dtype=bool)
        result[name] = {"flagged_cases": int(mask.sum()), "flagged_fraction": float(mask.mean()), "flagged_mae": float(error[mask].mean()) if mask.any() else None, "unflagged_mae": float(error[~mask].mean()) if (~mask).any() else None}
    any_flag = np.column_stack([m.to_numpy(dtype=bool) for m in flags.values()]).any(axis=1)
    result["any_flag"] = {"flagged_cases": int(any_flag.sum()), "flagged_fraction": float(any_flag.mean()), "flagged_mae": float(error[any_flag].mean()) if any_flag.any() else None, "unflagged_mae": float(error[~any_flag].mean()) if (~any_flag).any() else None}
    return result, pd.DataFrame({k: v.to_numpy(dtype=bool) for k, v in flags.items()}, index=frame.index)


# --------------------------------------------------------------------------- stages

def prepare(raw_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "gate_prepared_manifest.json").exists():
        raise ValueError("Prepared gate data already exist; preserve this run")
    cfg = data.protocol()
    raw, audit = data.load_raw(raw_path)
    inputs, targets, reserve = data.development(raw, cfg)
    inputs = inputs.sort_index()
    targets = targets.reindex(inputs.index)
    dev_raw = raw.loc[~raw.date.isin(cfg["reserved_dates"])]
    full = assemble(inputs, dev_raw, cfg)
    quality = outcome_quality(dev_raw, cfg).reindex(full.index)
    cases = full.index.to_frame(index=False)
    cases["case_id"] = case_ids(full.index)
    names = raw.loc[raw.identity.notna()].groupby("identity").trt.agg(lambda x: sorted(set(x))[0])
    cases["name"] = cases.identity.map(names)
    cases["plates_spanned"] = full.attrs["plates_spanned"].to_numpy()
    cases.to_csv(out / "gate_cases.csv", index=False)
    np.savez_compressed(out / "gate_development_data.npz", X=full.to_numpy(dtype=float), y=targets.to_numpy(dtype=float), columns=np.array(full.columns, dtype=str))
    pd.DataFrame({"case_id": cases.case_id, "day12_ctrl_median_ns": quality.to_numpy()}).to_csv(out / "outcome_quality.csv", index=False)
    consistency = {"selection_v1_available": False}
    source = data.DEFAULT_OUT / "development_data.npz"
    if source.exists():
        payload = np.load(source, allow_pickle=False)
        v1_columns = payload["columns"].tolist()
        difference = float(np.nanmax(abs(full[v1_columns].to_numpy(dtype=float) - payload["X"]))) if payload["X"].shape == full[v1_columns].shape else None
        consistency = {"selection_v1_available": True, "same_shape": payload["X"].shape == full[v1_columns].shape, "max_abs_difference_v1_features": difference, "max_abs_difference_targets": float(np.max(abs(payload["y"] - targets.to_numpy())))}
    audit.update(development_cases=len(full), development_substances=int(cases.identity.nunique()), development_dates=int(cases.date.nunique()), features=len(full.columns), reliability_features=RELIABILITY, cases_spanning_multiple_plates=int((cases.plates_spanned > 1).sum()), cases_with_day12_control_median_below_5=int((quality < 5).sum()), selection_v1_consistency=consistency)
    bench.write_json(out / "gate_data_audit.json", audit)
    bench.write_json(out / "gate_reserved_manifest.json", reserve)
    manifest = {"protocol_version": plan()["version"], "code_hashes": code_hashes(), "source_sha256": data.digest(raw_path), "data_sha256": data.digest(out / "gate_development_data.npz"), "cases_sha256": data.digest(out / "gate_cases.csv"), "quality_sha256": data.digest(out / "outcome_quality.csv"), "reserve_outcomes_scored": False}
    bench.write_json(out / "gate_prepared_manifest.json", manifest)
    print(json.dumps({"prepared": str(out), "cases": len(full), "features": len(full.columns), "consistency": consistency}), flush=True)


def load_prepared(out: Path):
    manifest = json.loads((out / "gate_prepared_manifest.json").read_text(encoding="utf-8"))
    if manifest["code_hashes"] != code_hashes():
        raise ValueError("Code or protocol changed since preparation; create a separately documented run")
    for name, key in [("gate_development_data.npz", "data_sha256"), ("gate_cases.csv", "cases_sha256"), ("outcome_quality.csv", "quality_sha256")]:
        if data.digest(out / name) != manifest[key]:
            raise ValueError(f"Prepared file hash mismatch: {name}")
    payload = np.load(out / "gate_development_data.npz", allow_pickle=False)
    cases = pd.read_csv(out / "gate_cases.csv", dtype={"identity": str})
    index = pd.MultiIndex.from_frame(cases[data.KEY])
    full = pd.DataFrame(payload["X"], columns=payload["columns"].tolist(), index=index)
    quality = pd.read_csv(out / "outcome_quality.csv").set_index("case_id").day12_ctrl_median_ns.reindex(cases.case_id).to_numpy()
    return full, payload["y"], cases, quality


def prediction_frame(cases: pd.DataFrame, idx: np.ndarray, v1: pd.DataFrame, y: np.ndarray, name: str, prediction: np.ndarray) -> pd.DataFrame:
    result = cases.iloc[idx][["case_id", "date", "identity", "dose", "name"]].copy()
    result["model"] = name
    result["prediction"] = prediction
    result["target"] = y[idx]
    result["day7"] = v1.iloc[idx].normalized_ns_d7.to_numpy()
    result["day5"] = v1.iloc[idx].normalized_ns_d5.to_numpy()
    return result


def develop(out: Path) -> None:
    if (out / "development_lock.json").exists():
        raise ValueError("A development lock already exists; preserve this run")
    started = time.monotonic()
    cfg, gate = data.protocol(), plan()
    full, y, cases, quality = load_prepared(out)
    frames = candidate_frames(full, cfg)
    v1 = frames["extra_trees_all"][0]
    learned = gate["candidates"]["learned"]
    models = TRIVIAL + learned
    n = len(full)
    outer = {m: np.full(n, np.nan) for m in models}
    inner = {}
    inner_rows, folds = [], []
    with threadpool_limits(limits=1):
        for date, train, test in data.split_indices(full):
            fold_start = time.monotonic()
            for m in models:
                outer[m][test], _ = fit_predict(m, frames, train, test, y, cfg)
                inner[(date, m)] = np.full(n, np.nan)
            sub_folds = []
            for inner_date, itrain, itest in data.split_indices(full.iloc[train]):
                g_train, g_test = train[itrain], train[itest]
                if set(cases.iloc[g_train].identity) & set(cases.iloc[g_test].identity) or set(cases.iloc[g_train].date) & {inner_date} or date in set(cases.iloc[g_train].date) | set(cases.iloc[g_test].date):
                    raise ValueError("Inner fold leaks a date or identity")
                for m in models:
                    prediction, _ = fit_predict(m, frames, g_train, g_test, y, cfg)
                    inner[(date, m)][g_test] = prediction
                    inner_rows.append(pd.DataFrame({"outer_date": date, "inner_date": inner_date, "case_id": cases.iloc[g_test].case_id.to_numpy(), "identity": cases.iloc[g_test].identity.to_numpy(), "model": m, "prediction": prediction, "target": y[g_test]}))
                sub_folds.append({"inner_date": inner_date, "train_case_ids": cases.iloc[g_train].case_id.tolist(), "test_case_ids": cases.iloc[g_test].case_id.tolist()})
            folds.append({"outer_date": date, "train_case_ids": cases.iloc[train].case_id.tolist(), "test_case_ids": cases.iloc[test].case_id.tolist(), "inner_folds": sub_folds})
            print(json.dumps({"outer_date_complete": date, "train_cases": len(train), "test_cases": len(test), "inner_folds": len(sub_folds), "seconds": round(time.monotonic() - fold_start, 1)}), flush=True)
    predictions = pd.concat([prediction_frame(cases, np.arange(n), v1, y, m, outer[m]) for m in models], ignore_index=True)
    predictions.to_csv(out / "development_predictions.csv", index=False)
    inner_table = pd.concat(inner_rows, ignore_index=True)
    inner_table.to_csv(out / "inner_predictions.csv", index=False)
    bench.write_json(out / "gate_folds.json", folds)

    # Full-development comparison and the product-model rule.
    losses = {m: date_losses(predictions.loc[predictions.model == m]) for m in models}
    macro = {m: float(losses[m].mean()) for m in models}
    product = min(learned, key=lambda m: macro[m])
    comparator = gate["candidates"]["strongest_simple_comparator"]

    # Nested estimate of the selection procedure.
    nested = []
    for fold in folds:
        date = fold["outer_date"]
        block = inner_table.loc[inner_table.outer_date == date]
        inner_macro = {}
        for m in models:
            part = block.loc[block.model == m]
            error = abs(part.prediction - part.target)
            inner_macro[m] = float(error.groupby([part.inner_date, part.identity]).mean().groupby(level=0).mean().mean())
        chosen = min(models, key=lambda m: inner_macro[m])
        nested.append({"outer_date": date, "chosen": chosen, "inner_macro_mae": inner_macro, "outer_loss_of_chosen": float(losses[chosen].loc[date]), "outer_loss_of_product": float(losses[product].loc[date])})
    nested_mae = float(np.mean([r["outer_loss_of_chosen"] for r in nested]))

    # Conformal intervals and difficulty model for the product model, calibrated inside each outer fold.
    levels = gate["reliability"]["nominal_levels"]
    columns = sigma_columns(product, frames)
    interval = pd.DataFrame(index=cases.index)
    interval["case_id"] = cases.case_id
    interval["sigma"] = np.nan
    for level in levels:
        for variant in ["symmetric", "scaled"]:
            interval[f"{variant}_lower_{level}"] = np.nan
            interval[f"{variant}_upper_{level}"] = np.nan
    fold_quantiles = []
    (out / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for date, train, test in data.split_indices(full):
            residual = y[train] - inner[(date, product)][train]
            if np.isnan(residual).any():
                raise ValueError("Missing inner residuals")
            sigma_oof = np.full(len(train), np.nan)
            for inner_date, itrain, itest in data.split_indices(full.iloc[train]):
                model = fit_sigma(frames, columns, train[itrain], abs(residual[itrain]), cfg)
                sigma_oof[itest] = sigma_predict(model, frames, columns, train[itest])
            sigma_model = fit_sigma(frames, columns, train, abs(residual), cfg)
            sigma_test = sigma_predict(sigma_model, frames, columns, test)
            forecast = outer[product][test]
            record: dict[str, object] = {"outer_date": date, "calibration_cases": int(len(train))}
            for level in levels:
                q_sym = conformal_quantile(abs(residual), level)
                q_scaled = conformal_quantile(abs(residual) / sigma_oof, level)
                record[f"symmetric_q_{level}"] = q_sym
                record[f"scaled_q_{level}"] = q_scaled
                interval.loc[interval.index[test], f"symmetric_lower_{level}"] = forecast - q_sym
                interval.loc[interval.index[test], f"symmetric_upper_{level}"] = forecast + q_sym
                interval.loc[interval.index[test], f"scaled_lower_{level}"] = forecast - q_scaled * sigma_test
                interval.loc[interval.index[test], f"scaled_upper_{level}"] = forecast + q_scaled * sigma_test
            interval.loc[interval.index[test], "sigma"] = sigma_test
            fold_quantiles.append(record)
            # Save the product and difficulty models for this held-out date, and check the replay.
            replay, bundle = fit_predict(product, frames, train, test, y, cfg)
            np.testing.assert_allclose(replay, forecast, atol=1e-12, rtol=0)
            joblib.dump({"model": bundle[0], "columns": bundle[1]}, out / "models" / f"{date}_{product}.joblib", compress=3)
            joblib.dump({"model": sigma_model, "columns": columns, "floor": SIGMA_FLOOR}, out / "models" / f"{date}_sigma.joblib", compress=3)
    product_frame = predictions.loc[predictions.model == product].reset_index(drop=True)
    comparator_frame = predictions.loc[predictions.model == comparator].reset_index(drop=True)
    evaluation = product_frame.join(interval.drop(columns="case_id"))
    evaluation = evaluation.join(full[RELIABILITY].reset_index(drop=True))
    evaluation["day12_ctrl_median_ns"] = quality
    evaluation.to_csv(out / "development_intervals.csv", index=False)

    thresholds = {"rep_sd_log2_ns_d7_p90": float(np.quantile(full["rel__rep_sd_log2_ns_d7"], 0.9)), "plate_ctrl_iqr_ratio_d7_p90": float(np.quantile(full["rel__plate_ctrl_iqr_ratio_d7"], 0.9)), "abstention_sigma_p80": float(np.quantile(evaluation.sigma, 0.8))}
    flags, flag_frame = flag_table(evaluation, thresholds)
    flag_frame.insert(0, "case_id", cases.case_id.to_numpy())
    flag_frame.to_csv(out / "development_flags.csv", index=False)

    # Warning decision: rank by score within date; product, comparator, persistence.
    positive = abs(product_frame.target) >= 1
    quiet = abs(product_frame.day7) < 0.5
    warning = {}
    for label, score in [("product_model", abs(product_frame.prediction)), (comparator, abs(comparator_frame.prediction)), ("persistence_rank", abs(product_frame.day7))]:
        frame = pd.DataFrame({"date": product_frame.date, "score": score.to_numpy(), "positive": positive.to_numpy(), "quiet": quiet.to_numpy()})
        warning[label] = warning_metrics(frame, gate["warning_decision"]["budgets"])
    threshold_diagnostics = {product: threshold_rule(product_frame), comparator: threshold_rule(comparator_frame)}

    subset = evaluation.day12_ctrl_median_ns >= 5
    sensitivity = {m: macro_mae(predictions.loc[(predictions.model == m) & predictions.case_id.isin(evaluation.loc[subset, "case_id"])]) for m in models}
    comparisons = {m: paired_bootstrap(losses[product].to_numpy(), losses[m].to_numpy(), cfg["seed"]) for m in models if m != product}
    consistency = v1_consistency(predictions)
    summary = {
        "protocol": gate["version"],
        "product_model": product,
        "product_model_rule": gate["candidates"]["product_model_rule"],
        "strongest_simple_comparator": comparator,
        "date_macro_mae": dict(sorted(macro.items(), key=lambda kv: kv[1])),
        "per_date_losses": {m: {str(k): float(v) for k, v in losses[m].items()} for m in models},
        "comparisons_product_vs": comparisons,
        "nested_selection": {"nested_date_macro_mae": nested_mae, "product_plain_date_macro_mae": macro[product], "optimism_estimate": macro[product] - nested_mae, "chosen_by_outer_fold": {str(r["outer_date"]): r["chosen"] for r in nested}, "details": nested},
        "warning": warning,
        "threshold_rule_diagnostics": threshold_diagnostics,
        "coverage": coverage_table(evaluation, levels),
        "abstention": abstention_table(evaluation, thresholds["abstention_sigma_p80"], gate["reliability"]["primary_level"]),
        "fold_quantiles": fold_quantiles,
        "measurement_flags": flags,
        "outcome_quality_sensitivity": {"rule": gate["target"]["outcome_quality_sensitivity_subset"], "retained_cases": int(subset.sum()), "date_macro_mae": dict(sorted(sensitivity.items(), key=lambda kv: kv[1]))},
        "selection_v1_consistency": consistency,
        "locked_thresholds": thresholds,
        "cases": n,
        "seconds": time.monotonic() - started,
        "reserve_outcomes_scored": False,
    }
    bench.write_json(out / "development_summary.json", summary)
    lock = {"protocol_version": gate["version"], "locked_at_seconds_since_epoch": time.time(), "product_model": product, "strongest_simple_comparator": comparator, "sigma_columns": columns, "locked_thresholds": thresholds, "primary_budget": gate["warning_decision"]["primary_budget"], "primary_level": gate["reliability"]["primary_level"], "code_hashes": code_hashes(), "file_sha256": {name: data.digest(out / name) for name in ["gate_development_data.npz", "gate_cases.csv", "outcome_quality.csv", "development_predictions.csv", "inner_predictions.csv", "development_intervals.csv", "development_flags.csv", "development_summary.json", "gate_folds.json"]}, "reserve_outcomes_scored": False}
    bench.write_json(out / "development_lock.json", lock)
    print(json.dumps({k: v for k, v in summary.items() if k in {"product_model", "date_macro_mae", "nested_selection", "abstention", "locked_thresholds", "seconds"}}, indent=2, default=str), flush=True)


def v1_consistency(predictions: pd.DataFrame) -> dict:
    """The gate must reproduce the frozen selection_v1 and follow-up predictions exactly."""
    result = {}
    for label, path in [("selection_v1", data.DEFAULT_OUT / "predictions.csv"), ("representation_followup", data.DEFAULT_OUT / "representation_followup/predictions.csv")]:
        if not path.exists():
            result[label] = {"available": False}
            continue
        old = pd.read_csv(path, dtype={"identity": str})
        merged = old.merge(predictions, on=["case_id", "model"], suffixes=("_old", "_new"))
        result[label] = {"available": True, "compared_rows": int(len(merged)), "models": sorted(merged.model.unique()), "max_abs_prediction_difference": float(np.max(abs(merged.prediction_old - merged.prediction_new))), "max_abs_target_difference": float(np.max(abs(merged.target_old - merged.target_new)))}
    return result


def verify_lock(out: Path) -> dict:
    lock = json.loads((out / "development_lock.json").read_text(encoding="utf-8"))
    if lock["code_hashes"] != code_hashes():
        raise ValueError("Code or protocol changed since the development lock")
    for name, digest in lock["file_sha256"].items():
        if data.digest(out / name) != digest:
            raise ValueError(f"Locked file changed: {name}")
    return lock


def reserve(out: Path, raw_path: Path, unseal: bool) -> None:
    if not unseal:
        raise ValueError("Reserve scoring requires --unseal after the development lock has been reviewed")
    lock = verify_lock(out)
    if (out / "reserve_predictions.csv").exists():
        raise ValueError("The reserve has already been scored; a rerun requires a new protocol version")
    started = time.monotonic()
    cfg, gate = data.protocol(), plan()
    dev_full, dev_y, dev_cases, _ = load_prepared(out)
    raw, _ = data.load_raw(raw_path)
    reserved_dates = cfg["reserved_dates"]
    res_raw = raw.loc[raw.date.isin(reserved_dates)]
    inputs = data.build_inputs(res_raw, cfg).sort_index()
    targets = data.build_targets(res_raw, cfg)
    common = inputs.index.intersection(targets.index, sort=False)
    inputs = inputs.loc[common].sort_index()
    res_y = targets.reindex(inputs.index).to_numpy(dtype=float)
    res_full = assemble(inputs, res_raw, cfg)
    if set(res_full.index.get_level_values("identity")) & set(dev_cases.identity) or set(res_full.index.get_level_values("date")) & set(dev_cases.date):
        raise ValueError("Reserve overlaps development dates or identities")
    quality = outcome_quality(res_raw, cfg).reindex(res_full.index).to_numpy()
    n_dev = len(dev_full)
    combined = pd.concat([dev_full, res_full])
    y_all = np.concatenate([dev_y, res_y])
    frames = candidate_frames(combined, cfg)
    v1 = frames["extra_trees_all"][0]
    train, test = np.arange(n_dev), np.arange(n_dev, len(combined))
    models = TRIVIAL + gate["candidates"]["learned"]
    product, comparator = lock["product_model"], lock["strongest_simple_comparator"]
    res_cases = res_full.index.to_frame(index=False)
    res_cases["case_id"] = case_ids(res_full.index)
    names = raw.loc[raw.identity.notna()].groupby("identity").trt.agg(lambda x: sorted(set(x))[0])
    res_cases["name"] = res_cases.identity.map(names)
    all_cases = pd.concat([dev_cases[["case_id", "date", "identity", "dose", "name"]], res_cases[["case_id", "date", "identity", "dose", "name"]]], ignore_index=True)
    outputs = {}
    (out / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for m in models:
            prediction, bundle = fit_predict(m, frames, train, test, y_all, cfg)
            outputs[m] = prediction_frame(all_cases, test, v1, y_all, m, prediction)
            if bundle is not None:
                joblib.dump({"model": bundle[0], "columns": bundle[1]}, out / "models" / f"final_{m}.joblib", compress=3)
        # Calibrate on the full-development out-of-fold residuals of the product model.
        dev_predictions = pd.read_csv(out / "development_predictions.csv", dtype={"identity": str})
        dev_product = dev_predictions.loc[dev_predictions.model == product].set_index("case_id").reindex(dev_cases.case_id)
        residual = dev_y - dev_product.prediction.to_numpy()
        columns = lock["sigma_columns"]
        sigma_oof = np.full(n_dev, np.nan)
        for inner_date, itrain, itest in data.split_indices(dev_full):
            model = fit_sigma(frames, columns, itrain, abs(residual[itrain]), cfg)
            sigma_oof[itest] = sigma_predict(model, frames, columns, itest)
        sigma_model = fit_sigma(frames, columns, train, abs(residual), cfg)
        joblib.dump({"model": sigma_model, "columns": columns, "floor": SIGMA_FLOOR}, out / "models" / "final_sigma.joblib", compress=3)
        sigma_test = sigma_predict(sigma_model, frames, columns, test)
    levels = gate["reliability"]["nominal_levels"]
    evaluation = outputs[product].reset_index(drop=True)
    evaluation["sigma"] = sigma_test
    quantiles = {}
    for level in levels:
        q_sym = conformal_quantile(abs(residual), level)
        q_scaled = conformal_quantile(abs(residual) / sigma_oof, level)
        quantiles[f"symmetric_q_{level}"] = q_sym
        quantiles[f"scaled_q_{level}"] = q_scaled
        evaluation[f"symmetric_lower_{level}"] = evaluation.prediction - q_sym
        evaluation[f"symmetric_upper_{level}"] = evaluation.prediction + q_sym
        evaluation[f"scaled_lower_{level}"] = evaluation.prediction - q_scaled * sigma_test
        evaluation[f"scaled_upper_{level}"] = evaluation.prediction + q_scaled * sigma_test
    evaluation = evaluation.join(res_full[RELIABILITY].reset_index(drop=True))
    evaluation["day12_ctrl_median_ns"] = quality
    thresholds = lock["locked_thresholds"]
    evaluation["abstain"] = evaluation.sigma > thresholds["abstention_sigma_p80"]
    flags, flag_frame = flag_table(evaluation, thresholds)
    for column in flag_frame.columns:
        evaluation[f"flag_{column}"] = flag_frame[column].to_numpy()
    predictions = pd.concat(outputs.values(), ignore_index=True)
    predictions.to_csv(out / "reserve_predictions.csv", index=False)
    evaluation.to_csv(out / "reserve_intervals.csv", index=False)

    losses = {m: date_losses(predictions.loc[predictions.model == m]) for m in models}
    macro = {m: float(losses[m].mean()) for m in models}
    positive = abs(evaluation.target) >= 1
    quiet = abs(evaluation.day7) < 0.5
    comparator_frame = outputs[comparator].reset_index(drop=True)
    warning = {}
    for label, score in [("product_model", abs(evaluation.prediction)), (comparator, abs(comparator_frame.prediction)), ("persistence_rank", abs(evaluation.day7))]:
        frame = pd.DataFrame({"date": evaluation.date, "score": score.to_numpy(), "positive": positive.to_numpy(), "quiet": quiet.to_numpy()})
        warning[label] = warning_metrics(frame, gate["warning_decision"]["budgets"])
    budget, level = lock["primary_budget"], lock["primary_level"]
    def pooled_recall(label):
        record = next(r for r in warning[label] if r["subgroup"] == "early_quiet" and r["budget"] == budget)
        return record["pooled"]["recall"]
    coverage = coverage_table(evaluation, levels)
    abstention = abstention_table(evaluation, thresholds["abstention_sigma_p80"], level)
    subset = evaluation.day12_ctrl_median_ns >= 5
    sensitivity = {m: (macro_mae(predictions.loc[(predictions.model == m) & predictions.case_id.isin(evaluation.loc[subset, "case_id"])]) if subset.any() else None) for m in models}
    product_recall, persistence_recall, comparator_recall = pooled_recall("product_model"), pooled_recall("persistence_rank"), pooled_recall(comparator)
    random_recall = next(r for r in warning["product_model"] if r["subgroup"] == "early_quiet" and r["budget"] == budget)["pooled"]["random_expected_recall"]
    per_date_product_vs_comparator = int((losses[product] < losses[comparator]).sum())
    per_date_product_vs_persistence = int((losses[product] < losses["persistence_day7"]).sum())
    recalls_defined = all(v is not None for v in [product_recall, persistence_recall, comparator_recall, random_recall])
    criteria = {
        "C1a_forecast_vs_simple_baselines": {"pass": bool(macro[product] < macro["persistence_day7"] and macro[product] < macro["control_reference"] and per_date_product_vs_persistence >= 2), "product_mean_mae": macro[product], "persistence_mean_mae": macro["persistence_day7"], "control_reference_mean_mae": macro["control_reference"], "dates_product_beats_persistence": per_date_product_vs_persistence, "date_groups": int(len(losses[product]))},
        "C1b_forecast_vs_small_model": {"pass": bool(macro[product] < macro[comparator] and per_date_product_vs_comparator >= 2), "product_mean_mae": macro[product], "comparator_mean_mae": macro[comparator], "dates_product_beats_comparator": per_date_product_vs_comparator, "date_groups": int(len(losses[product]))},
        "C2a_warning_vs_naive_rule": {"pass": bool(recalls_defined and product_recall > persistence_recall and product_recall > random_recall), "budget": budget, "subgroup": "early_quiet", "product_recall": product_recall, "persistence_rank_recall": persistence_recall, "random_expected_recall": random_recall},
        "C2b_warning_vs_small_model": {"pass": bool(recalls_defined and product_recall >= comparator_recall), "budget": budget, "subgroup": "early_quiet", "product_recall": product_recall, "comparator_recall": comparator_recall},
        "C3_reliability": {"pass": bool(coverage[f"scaled_{level}"]["pooled_observed_coverage"] >= 0.70 and abstention["retained_case_mae"] is not None and abstention["retained_case_mae"] < abstention["all_case_mae"]), "nominal_level": level, "pooled_observed_coverage_scaled": coverage[f"scaled_{level}"]["pooled_observed_coverage"], "retained_case_mae": abstention["retained_case_mae"], "all_case_mae": abstention["all_case_mae"]},
        "C4_measurement_display": {"pass": None, "note": "Descriptive only"},
    }
    summary = {
        "protocol": gate["version"],
        "scored_once": True,
        "reserve_dates": reserved_dates,
        "reserve_cases": int(len(evaluation)),
        "reserve_identities": int(evaluation.identity.nunique()),
        "cases_per_date": {str(k): int(v) for k, v in evaluation.groupby("date").size().items()},
        "identities_per_date": {str(k): int(v) for k, v in evaluation.groupby("date").identity.nunique().items()},
        "product_model": product,
        "strongest_simple_comparator": comparator,
        "date_macro_mae": dict(sorted(macro.items(), key=lambda kv: kv[1])),
        "per_date_losses": {m: {str(k): float(v) for k, v in losses[m].items()} for m in models},
        "warning": warning,
        "threshold_rule_diagnostics": {product: threshold_rule(evaluation), comparator: threshold_rule(comparator_frame)},
        "coverage": coverage,
        "abstention": abstention,
        "conformal_quantiles": quantiles,
        "calibration_cases": int(n_dev),
        "measurement_flags": flags,
        "outcome_quality_sensitivity": {"retained_cases": int(subset.sum()), "date_macro_mae": sensitivity},
        "criteria": criteria,
        "seconds": time.monotonic() - started,
        "interpretation": "Three later experiment dates from the same public rat-culture study with chemically disjoint identities; not living-chip, clinical, or broad external validation.",
    }
    bench.write_json(out / "reserve_summary.json", summary)
    manifest = {"protocol_version": gate["version"], "development_lock_sha256": data.digest(out / "development_lock.json"), "code_hashes": code_hashes(), "source_sha256": data.digest(raw_path), "file_sha256": {name: data.digest(out / name) for name in ["reserve_predictions.csv", "reserve_intervals.csv", "reserve_summary.json"]}, "reserve_outcomes_scored": True, "unsealed_at_seconds_since_epoch": time.time()}
    bench.write_json(out / "reserve_manifest.json", manifest)
    print(json.dumps({k: v for k, v in summary.items() if k in {"reserve_cases", "cases_per_date", "date_macro_mae", "criteria", "abstention", "seconds"}}, indent=2, default=str), flush=True)


def audit(out: Path, raw_path: Path) -> None:
    cfg, gate = data.protocol(), plan()
    lock = verify_lock(out)
    full, y, cases, _ = load_prepared(out)
    raw, _ = data.load_raw(raw_path)
    dev_raw = raw.loc[~raw.date.isin(cfg["reserved_dates"])]
    # Reliability indicators must be reproducible and blind to everything after day 7.
    rebuilt = reliability_indicators(dev_raw, cfg).reindex(full.index)[RELIABILITY]
    pd.testing.assert_frame_equal(rebuilt, full[RELIABILITY], check_exact=True, check_names=False)
    mutated = dev_raw.copy()
    future = mutated.DIV > 7
    for column in cfg["feature_readouts"]:
        mutated.loc[future, column] = 987654.0
    pd.testing.assert_frame_equal(reliability_indicators(mutated, cfg).reindex(full.index)[RELIABILITY], full[RELIABILITY], check_exact=True, check_names=False)
    past_only = reliability_indicators(dev_raw.loc[dev_raw.DIV <= 7], cfg).reindex(full.index)[RELIABILITY]
    pd.testing.assert_frame_equal(past_only, full[RELIABILITY], check_exact=True, check_names=False)
    # Fold purity, recorded membership, and saved-model replays.
    folds = json.loads((out / "gate_folds.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(out / "development_predictions.csv", dtype={"identity": str})
    product = lock["product_model"]
    frames = candidate_frames(full, cfg)
    replays, max_difference, inner_checked = 0, 0.0, 0
    identity_of = dict(zip(cases.case_id, cases.identity))
    date_of = dict(zip(cases.case_id, cases.date))
    for date, train, test in data.split_indices(full):
        fold = next(f for f in folds if f["outer_date"] == date)
        assert set(fold["train_case_ids"]) == set(cases.iloc[train].case_id) and set(fold["test_case_ids"]) == set(cases.iloc[test].case_id)
        assert not set(cases.iloc[train].identity) & set(cases.iloc[test].identity) and date not in set(cases.iloc[train].date)
        for inner in fold["inner_folds"]:
            tr, te = inner["train_case_ids"], inner["test_case_ids"]
            assert not {identity_of[c] for c in tr} & {identity_of[c] for c in te}
            assert not {date_of[c] for c in tr} & {date_of[c] for c in te}
            assert date not in {date_of[c] for c in tr} | {date_of[c] for c in te}
            inner_checked += 1
        recorded = predictions.loc[(predictions.model == product) & (predictions.date == date)].set_index("case_id").reindex(cases.iloc[test].case_id)
        bundle = joblib.load(out / "models" / f"{date}_{product}.joblib")
        replay = bundle["model"].predict(frames[product][0].iloc[test][bundle["columns"]])
        max_difference = max(max_difference, float(np.max(abs(replay - recorded.prediction.to_numpy()))))
        np.testing.assert_allclose(replay, recorded.prediction, atol=1e-10, rtol=0)
        replays += 1
    result = {"passed": True, "reliability_rebuild_exact": True, "future_mutation_invariance": True, "past_only_invariance": True, "outer_folds": replays, "inner_folds_checked": inner_checked, "saved_product_model_replays": replays, "maximum_replay_difference": max_difference, "lock_verified": True, "reserve_outcomes_scored": (out / "reserve_manifest.json").exists()}
    if (out / "reserve_manifest.json").exists():
        manifest = json.loads((out / "reserve_manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["file_sha256"].items():
            if data.digest(out / name) != digest:
                raise ValueError(f"Reserve output changed: {name}")
        reserve_predictions = pd.read_csv(out / "reserve_predictions.csv", dtype={"identity": str})
        assert not set(reserve_predictions.identity) & set(cases.identity)
        assert set(reserve_predictions.date) <= set(cfg["reserved_dates"]) and not set(reserve_predictions.date) & set(cases.date)
        assert manifest["development_lock_sha256"] == data.digest(out / "development_lock.json")
        result.update(reserve_hashes_verified=True, reserve_identity_and_date_disjointness=True, reserve_cases=int(reserve_predictions.case_id.nunique()))
    bench.write_json(out / "gate_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "develop", "reserve", "audit"], required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=data.DEFAULT_RAW)
    parser.add_argument("--unseal", action="store_true", help="Required for the single reserve scoring")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.raw, args.out)
    elif args.stage == "develop":
        develop(args.out)
    elif args.stage == "reserve":
        reserve(args.out, args.raw, args.unseal)
    else:
        audit(args.out, args.raw)
