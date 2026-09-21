"""Run the selected prototype from day-7 observations and their controls only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import neuroforecast_data as data

MODEL = "extra_trees_relative_day7"


def inputs_from_day7(observed: pd.DataFrame) -> pd.DataFrame:
    if set(observed.DIV) != {7}:
        raise ValueError("This entry point accepts day-7 observations only")
    if observed.duplicated(data.PHYSICAL).any():
        raise ValueError("Duplicate physical observations must be audited first")
    if set(observed.units) != {"uM"}:
        raise ValueError("Expected concentrations in uM")
    if observed["eligible_identity"].dtype != bool:
        raise ValueError("eligible_identity must contain audited Boolean values")
    cfg = data.protocol()
    day = data._observed_day(observed, 7, cfg)
    columns = {
        "normalized_ns_d7": day.normalized_ns,
        "log10_dose": np.log10(day.index.get_level_values("dose").to_numpy(dtype=float)),
    }
    for field in cfg["feature_readouts"]:
        raw = day[field].to_numpy()
        control = day["control__" + field].to_numpy()
        denominator = abs(raw) + abs(control)
        columns[f"contrast__{field}__d7"] = np.divide(raw - control, denominator, out=np.zeros(len(day)), where=denominator != 0)
        columns[f"missing__{field}__d7"] = day["missing__" + field]
    return pd.DataFrame(columns, index=day.index)


def predict(observed: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    inputs = inputs_from_day7(observed)
    bundle = joblib.load(model_path)
    if set(bundle["columns"]) != set(inputs.columns):
        raise ValueError("Model does not have the frozen day-7 feature schema")
    result = inputs.index.to_frame(index=False)
    result["case_id"] = [f"{d}|{i}|{c:.12g}" for d, i, c in inputs.index]
    result["forecast_log2_relative_ns"] = bundle["model"].predict(inputs[bundle["columns"]])
    result["forecast_relative_ns"] = 2 ** result.forecast_log2_relative_ns
    result["large_change_alert"] = abs(result.forecast_log2_relative_ns) >= 1
    return result


def export_examples(source: Path, raw_path: Path) -> None:
    cfg = data.protocol()
    raw, _ = data.load_raw(raw_path)
    predictions = pd.read_csv(source / "representation_followup/predictions.csv", dtype={"identity": str})
    predictions = predictions.loc[predictions.model == MODEL]
    dates = pd.read_csv(source / "representation_followup/date_metrics.csv")
    losses = dates.loc[dates.model == MODEL].set_index("date").absolute_error.sort_values()
    selected = {"representative": int(losses.index[len(losses) // 2]), "largest_error": int(losses.index[-1])}
    out = source / "demo"
    out.mkdir(exist_ok=True)
    results = []
    for label, date in selected.items():
        expected = predictions.loc[predictions.date == date].set_index("case_id")
        observed = raw.loc[(raw.date == date) & (raw.DIV == 7)].copy()
        ids = [f"{d}|{i}|{c:.12g}" for d, i, c in observed[data.KEY].itertuples(index=False, name=None)]
        observed = observed.loc[(observed.dose == 0) | pd.Series(ids, index=observed.index).isin(expected.index)]
        input_path = out / f"{label}_day7.csv"
        observed.to_csv(input_path, index=False)
        model_path = source / "representation_followup/models" / f"{date}_{MODEL}.joblib"
        actual = predict(pd.read_csv(input_path, dtype={"identity": str}), model_path).set_index("case_id")
        actual = actual.reindex(expected.index)
        np.testing.assert_allclose(actual.forecast_log2_relative_ns, expected.prediction, atol=1e-10, rtol=0)
        actual.reset_index().to_csv(out / f"{label}_forecast.csv", index=False)
        invalid = observed.copy()
        invalid.loc[invalid.index[0], "DIV"] = 12
        try:
            inputs_from_day7(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Future observations were accepted by the inference entry point")
        results.append({"example": label, "held_out_date": date, "cases": len(actual), "input": input_path.name, "model": str(model_path), "input_sha256": data.digest(input_path), "model_sha256": data.digest(model_path), "maximum_difference_from_cv": float(np.max(abs(actual.forecast_log2_relative_ns - expected.prediction)))})
    manifest = {"purpose": "Executable retrospective examples, with no future observations in the input", "selection_rule": "Upper median and largest per-date development MAE of the selected model", "examples": results, "future_input_rejection_checked": True, "point_predictions_are_not_calibrated_probabilities": True, "reserve_outcomes_scored": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-examples", action="store_true")
    parser.add_argument("--source", type=Path, default=data.DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=data.DEFAULT_RAW)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.export_examples:
        export_examples(args.source, args.raw)
    else:
        if not all([args.input, args.model, args.output]):
            parser.error("Supply --input, --model, --output, or use --export-examples")
        result = predict(pd.read_csv(args.input, dtype={"identity": str}), args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(json.dumps({"output": str(args.output), "cases": len(result), "input_day": 7, "forecast_day": 12}), flush=True)
