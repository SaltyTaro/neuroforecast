"""Frozen exploratory, culture-grouped forecasting benchmark; reserve is not scored."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import neuroforecast_data as data


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def code_hashes() -> dict:
    return {p.name: data.digest(p) for p in [Path(__file__), Path(data.__file__), data.PROTOCOL_PATH]}


def prepare(raw_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = data.protocol()
    raw, audit = data.load_raw(raw_path)
    inputs, targets, reserve = data.development(raw, cfg)
    inputs = inputs.sort_index()
    targets = targets.reindex(inputs.index)
    case_table = inputs.index.to_frame(index=False)
    case_table["case_id"] = [f"{d}|{i}|{c:.12g}" for d, i, c in inputs.index]
    names = raw.loc[raw.identity.notna()].groupby("identity").trt.agg(lambda x: sorted(set(x))[0])
    case_table["name"] = case_table.identity.map(names)
    case_table.to_csv(out / "development_cases.csv", index=False)
    np.savez_compressed(out / "development_data.npz", X=inputs.to_numpy(dtype=float), y=targets.to_numpy(dtype=float), columns=np.array(inputs.columns, dtype=str))
    audit.update(
        development_cases=len(inputs),
        development_substances=int(case_table.identity.nunique()),
        development_dates=int(case_table.date.nunique()),
        features=len(inputs.columns),
        case_count_by_date={str(k): int(v) for k, v in case_table.groupby("date").size().items()},
    )
    write_json(out / "data_audit.json", audit)
    write_json(out / "reserved_manifest.json", reserve)
    write_json(out / "prepared_manifest.json", {"code_hashes": code_hashes(), "source_sha256": data.digest(raw_path), "data_sha256": data.digest(out / "development_data.npz"), "cases_sha256": data.digest(out / "development_cases.csv")})
    print(json.dumps({"prepared": str(out), "cases": len(inputs), "substances": int(case_table.identity.nunique()), "dates": int(case_table.date.nunique()), "features": len(inputs.columns), "reserve_scored": False}), flush=True)


def load_prepared(out: Path):
    manifest = json.loads((out / "prepared_manifest.json").read_text(encoding="utf-8"))
    if manifest["code_hashes"] != code_hashes():
        raise ValueError("Code/protocol changed since preparation; create a separately documented run")
    for name, key in [("development_data.npz", "data_sha256"), ("development_cases.csv", "cases_sha256")]:
        if data.digest(out / name) != manifest[key]:
            raise ValueError("Prepared data hash mismatch")
    payload = np.load(out / "development_data.npz", allow_pickle=False)
    cases = pd.read_csv(out / "development_cases.csv", dtype={"identity": str})
    index = pd.MultiIndex.from_frame(cases[data.KEY])
    inputs = pd.DataFrame(payload["X"], columns=payload["columns"].tolist(), index=index)
    return inputs, payload["y"], cases


def fitted_model(name: str, cfg: dict):
    if name.startswith("ridge"):
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=cfg["ridge"]["alpha"]))
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), ExtraTreesRegressor(**cfg["extra_trees"], random_state=cfg["seed"]))


def columns_for(name: str, inputs: pd.DataFrame) -> list[str]:
    if name.endswith("target_only"):
        return ["normalized_ns_d5", "normalized_ns_d7", "log10_dose"]
    return list(inputs.columns)


def evaluate(out: Path) -> None:
    if (out / "predictions.csv").exists():
        raise ValueError("Completed predictions already exist; preserve this run")
    started = time.monotonic()
    cfg = data.protocol()
    inputs, targets, cases = load_prepared(out)
    (out / "models").mkdir(exist_ok=True)
    outputs, folds = [], []
    with threadpool_limits(limits=1):
        for date, train, test in data.split_indices(inputs):
            fold_start = time.monotonic()
            folds.append({"test_date": date, "train_case_ids": cases.iloc[train].case_id.tolist(), "test_case_ids": cases.iloc[test].case_id.tolist(), "train_identities": sorted(set(cases.iloc[train].identity)), "test_identities": sorted(set(cases.iloc[test].identity))})
            for name in cfg["models"]:
                if name == "control_reference":
                    prediction = np.zeros(len(test))
                elif name == "persistence_day7":
                    prediction = inputs.iloc[test].normalized_ns_d7.to_numpy()
                elif name == "linear_trend":
                    prediction = (inputs.iloc[test].normalized_ns_d7 + 2.5 * (inputs.iloc[test].normalized_ns_d7 - inputs.iloc[test].normalized_ns_d5)).to_numpy()
                else:
                    columns = columns_for(name, inputs)
                    model = fitted_model(name, cfg)
                    model.fit(inputs.iloc[train][columns], targets[train])
                    prediction = model.predict(inputs.iloc[test][columns])
                    joblib.dump({"model": model, "columns": columns}, out / "models" / f"{date}_{name}.joblib", compress=3)
                result = cases.iloc[test].copy()
                result["model"] = name
                result["target"] = targets[test]
                result["prediction"] = prediction
                result["day7"] = inputs.iloc[test].normalized_ns_d7.to_numpy()
                result["day5"] = inputs.iloc[test].normalized_ns_d5.to_numpy()
                outputs.append(result)
            print(json.dumps({"completed_date": date, "train_cases": len(train), "test_cases": len(test), "seconds": round(time.monotonic() - fold_start, 2)}), flush=True)
    predictions = pd.concat(outputs, ignore_index=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    write_json(out / "folds.json", folds)
    write_json(out / "run_manifest.json", {"code_hashes": code_hashes(), "seconds": time.monotonic() - started, "predictions_sha256": data.digest(out / "predictions.csv"), "reserve_outcomes_scored": False})
    summarize(out)


def summarize(out: Path) -> None:
    cfg = data.protocol()
    p = pd.read_csv(out / "predictions.csv", dtype={"identity": str})
    p["absolute_error"] = abs(p.prediction - p.target)
    p["squared_error"] = (p.prediction - p.target) ** 2
    chemical = p.groupby(["model", "date", "identity"], as_index=False).absolute_error.mean()
    dates = chemical.groupby(["model", "date"], as_index=False).absolute_error.mean()
    chemical.to_csv(out / "chemical_metrics.csv", index=False)
    dates.to_csv(out / "date_metrics.csv", index=False)
    large = abs(p.target) >= 1
    late = large & (abs(p.day7) < 0.5)
    metrics = []
    for model, frame in p.groupby("model"):
        a = frame.loc[large.loc[frame.index]]
        b = frame.loc[late.loc[frame.index]]
        metrics.append({
            "model": model,
            "primary_date_macro_mae": float(dates.loc[dates.model == model, "absolute_error"].mean()),
            "case_mae": float(frame.absolute_error.mean()),
            "case_rmse": float(np.sqrt(frame.squared_error.mean())),
            "cases": len(frame),
            "large_change_cases": len(a),
            "large_change_miss_rate": float((abs(a.prediction) < 1).mean()) if len(a) else None,
            "early_quiet_late_change_cases": len(b),
            "early_quiet_late_change_mae": float(b.absolute_error.mean()) if len(b) else None,
            "early_quiet_late_change_miss_rate": float((abs(b.prediction) < 1).mean()) if len(b) else None,
        })
    table = pd.DataFrame(metrics).sort_values("primary_date_macro_mae")
    table.to_csv(out / "metrics.csv", index=False)
    baseline = table.loc[table.model != cfg["primary_model"]].iloc[0].model
    wide = dates.pivot(index="date", columns="model", values="absolute_error")
    primary = wide[cfg["primary_model"]].to_numpy()
    reference = wide[baseline].to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    indices = rng.integers(len(wide), size=(10000, len(wide)))
    boot = 1 - primary[indices].mean(axis=1) / reference[indices].mean(axis=1)
    summary = {
        "protocol": cfg["version"],
        "primary_model": cfg["primary_model"],
        "strongest_baseline": baseline,
        "relative_mae_reduction": float(1 - primary.mean() / reference.mean()),
        "paired_date_descriptive_interval_95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "dates_improved": int((primary < reference).sum()),
        "date_groups": len(wide),
        "development_substances": int(p.identity.nunique()),
        "development_cases": int(p.case_id.nunique()),
        "early_quiet_late_change_cases": int(p.loc[late, "case_id"].nunique()),
        "metrics": metrics,
        "reserve_outcomes_scored": False,
        "interpretation": "Exploratory project-selection results on public rat cortical cultures; not prospective, clinical, or living organ-chip validation.",
    }
    write_json(out / "summary.json", summary)
    print(table.to_string(index=False), flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"metrics", "interpretation"}}, indent=2), flush=True)


def audit(out: Path, raw_path: Path) -> None:
    cfg = data.protocol()
    inputs, targets, cases = load_prepared(out)
    raw, _ = data.load_raw(raw_path)
    past = raw.loc[raw.DIV <= max(cfg["cutoff_days"])].copy()
    input_full = data.build_inputs(raw, cfg)
    input_past = data.build_inputs(past, cfg)
    pd.testing.assert_frame_equal(input_full, input_past, check_exact=True)
    changed = raw.copy()
    future = changed.DIV > max(cfg["cutoff_days"])
    # Perturb every future functional value, including all future controls.
    for column in cfg["feature_readouts"]:
        changed.loc[future, column] = 123456.0
    pd.testing.assert_frame_equal(input_full, data.build_inputs(changed, cfg), check_exact=True)
    reserved = data.reserve_ids(raw, cfg)
    assert not set(cases.identity) & reserved
    assert not set(cases.date) & set(cfg["reserved_dates"])
    p = pd.read_csv(out / "predictions.csv", dtype={"identity": str})
    folds = json.loads((out / "folds.json").read_text(encoding="utf-8"))
    replays, max_difference = 0, 0.0
    for date, train, test in data.split_indices(inputs):
        fold = next(f for f in folds if f["test_date"] == date)
        assert set(fold["train_case_ids"]) == set(cases.iloc[train].case_id)
        assert set(fold["test_case_ids"]) == set(cases.iloc[test].case_id)
        assert not set(cases.iloc[train].identity) & set(cases.iloc[test].identity)
        assert date not in set(cases.iloc[train].date)
        for model in cfg["models"]:
            recorded = p.loc[(p.date == date) & (p.model == model)].set_index("case_id").reindex(cases.iloc[test].case_id)
            assert len(recorded) == len(test) and recorded.prediction.notna().all()
            np.testing.assert_allclose(recorded.target, targets[test], atol=1e-12)
            if model in {"control_reference", "persistence_day7", "linear_trend"}:
                continue
            bundle = joblib.load(out / "models" / f"{date}_{model}.joblib")
            predicted = bundle["model"].predict(inputs.iloc[test][bundle["columns"]])
            difference = float(np.max(abs(predicted - recorded.prediction.to_numpy())))
            max_difference = max(max_difference, difference)
            np.testing.assert_allclose(predicted, recorded.prediction, atol=1e-10, rtol=0)
            replays += 1
    result = {"passed": True, "future_removal_input_invariance": True, "future_value_and_control_mutation_invariance": True, "reserved_date_and_identity_disjointness": True, "fold_date_and_identity_disjointness": True, "saved_model_replays": replays, "maximum_replay_difference": max_difference, "case_count": len(inputs), "reserve_outcomes_scored": False}
    write_json(out / "audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "evaluate", "summarize", "audit"], required=True)
    parser.add_argument("--out", type=Path, default=data.DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=data.DEFAULT_RAW)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.raw, args.out)
    elif args.stage == "evaluate":
        evaluate(args.out)
    elif args.stage == "audit":
        audit(args.out, args.raw)
    else:
        summarize(args.out)
