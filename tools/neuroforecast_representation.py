"""One documented control-relative representation comparison on development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import neuroforecast_benchmark as baseline
import neuroforecast_data as data

PROTOCOL = data.ROOT / "experiments/neuroforecast_representation_protocol.json"


def representation(inputs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = {name: inputs[name] for name in ["normalized_ns_d5", "normalized_ns_d7", "log10_dose"]}
    for day in cfg["cutoff_days"]:
        for field in cfg["feature_readouts"]:
            raw = np.sinh(inputs[f"raw__{field}__d{day}"].to_numpy())
            control = np.sinh(inputs[f"control__{field}__d{day}"].to_numpy())
            denominator = abs(raw) + abs(control)
            contrast = np.divide(raw - control, denominator, out=np.zeros(len(inputs)), where=denominator != 0)
            cols[f"contrast__{field}__d{day}"] = contrast
            cols[f"missing__{field}__d{day}"] = inputs[f"missing__{field}__d{day}"]
    return pd.DataFrame(cols, index=inputs.index)


def selected_columns(model: str, x: pd.DataFrame) -> list[str]:
    if model.endswith("activity_coordination"):
        fields = ["meanfiringrate", "r", "nAE", "nABE", "ns.n"]
        return ["normalized_ns_d5", "normalized_ns_d7", "log10_dose"] + [f"contrast__{f}__d{d}" for d in [5, 7] for f in fields]
    if model.endswith("day7"):
        return [c for c in x if not c.endswith("d5")]
    return list(x.columns)


def run(source: Path) -> None:
    start = time.monotonic()
    cfg = data.protocol()
    plan = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    out = source / "representation_followup"
    out.mkdir(exist_ok=True)
    if (out / "predictions.csv").exists():
        raise ValueError("Preserve the existing follow-up predictions")
    inputs, targets, cases = baseline.load_prepared(source)
    transformed = representation(inputs, cfg)
    # A change of units for a measured rate and its controls must not change its contrast.
    scaled = inputs.copy()
    for day in cfg["cutoff_days"]:
        for prefix in ["raw", "control"]:
            key = f"{prefix}__meanfiringrate__d{day}"
            scaled[key] = np.arcsinh(60 * np.sinh(scaled[key]))
    np.testing.assert_allclose(transformed, representation(scaled, cfg), equal_nan=True, atol=1e-12, rtol=1e-12)
    outputs, replays = [], []
    (out / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for date, train, test in data.split_indices(inputs):
            assert not set(cases.iloc[train].identity) & set(cases.iloc[test].identity)
            for name in plan["models"]:
                columns = selected_columns(name, transformed)
                model = baseline.fitted_model(name, cfg)
                model.fit(transformed.iloc[train][columns], targets[train])
                prediction = model.predict(transformed.iloc[test][columns])
                model_path = out / "models" / f"{date}_{name}.joblib"
                joblib.dump({"model": model, "columns": columns}, model_path, compress=3)
                loaded = joblib.load(model_path)
                restored = loaded["model"].predict(transformed.iloc[test][columns])
                np.testing.assert_allclose(prediction, restored, atol=1e-12, rtol=0)
                replays.append(float(np.max(abs(prediction - restored))))
                result = cases.iloc[test].copy()
                result["model"] = name
                result["prediction"] = prediction
                result["target"] = targets[test]
                result["day7"] = inputs.iloc[test].normalized_ns_d7.to_numpy()
                result["day5"] = inputs.iloc[test].normalized_ns_d5.to_numpy()
                outputs.append(result)
            print(json.dumps({"relative_representation_date_complete": date}), flush=True)
    p = pd.concat(outputs, ignore_index=True)
    p.to_csv(out / "predictions.csv", index=False)
    original = pd.read_csv(source / "predictions.csv", dtype={"identity": str})
    all_results = pd.concat([original, p], ignore_index=True)
    all_results["absolute_error"] = abs(all_results.prediction - all_results.target)
    chemical = all_results.groupby(["model", "date", "identity"], as_index=False).absolute_error.mean()
    dates = chemical.groupby(["model", "date"], as_index=False).absolute_error.mean()
    dates.to_csv(out / "date_metrics.csv", index=False)
    metrics = dates.groupby("model").absolute_error.mean().sort_values()
    metrics.rename("primary_date_macro_mae").to_csv(out / "metrics.csv")
    wide = dates.pivot(index="date", columns="model", values="absolute_error")
    comparator = metrics.drop(plan["primary_model"]).index[0]
    a, b = wide[plan["primary_model"]].to_numpy(), wide[comparator].to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    index = rng.integers(len(wide), size=(10000, len(wide)))
    gain = 1 - a[index].mean(axis=1) / b[index].mean(axis=1)
    summary = {
        "status": plan["status"],
        "primary_model": plan["primary_model"],
        "strongest_comparator": comparator,
        "relative_mae_reduction": float(1 - a.mean() / b.mean()),
        "descriptive_paired_date_interval_95": np.quantile(gain, [0.025, 0.975]).tolist(),
        "date_groups_improved": int((a < b).sum()),
        "date_groups": len(wide),
        "metrics": {str(k): float(v) for k, v in metrics.items()},
        "seconds": time.monotonic() - start,
        "rate_unit_invariance_check": True,
        "saved_model_replays": len(replays),
        "max_replay_difference": max(replays),
        "reserve_outcomes_scored": False,
        "source_prepared_manifest_sha256": data.digest(source / "prepared_manifest.json"),
        "source_predictions_sha256": data.digest(source / "predictions.csv"),
        "code_sha256": data.digest(Path(__file__)),
        "protocol_sha256": data.digest(PROTOCOL),
    }
    baseline.write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=data.DEFAULT_OUT)
    args = parser.parse_args()
    run(args.source)
