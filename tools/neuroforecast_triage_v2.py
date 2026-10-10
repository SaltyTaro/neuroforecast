"""NeuroForecast v2 day-7 triage: batch quality, a day-12 forecast beside persistence, and a trust verdict.

Input: well-level day-5 and day-7 MEA readouts with same-plate zero-dose controls (days 9 and 12 are rejected).
Output, per condition: the v2 forecast of the day-12 control-relative network-spike readout (log2), an 80%
interval, the carry-day-7-forward value, a difficulty score and a trust verdict; per batch: a quality verdict.

Three later steps close the loop. --day12 scores the day-7 report against the batch's day-12 recordings once they
exist (the sealed test's own target definition), and --record writes an admission record: per condition, whether the
forecast may stand in provisionally or day 12 must be measured, the reasons, and the model and input hashes.
--history takes the lab's own earlier checked records and refits the interval scale from them (split-conformal
quantile of |measured - forecast| / difficulty); forecasts and verdicts do not change.

Evidence (two sealed tests, each scored once):
  * v1 reserve, 3 later batches: persistence 0.6088 beat the v1 model 0.7716.
  * External, 24 earlier batches, 105 unseen chemicals: this v2 model 0.735 vs persistence 0.980,
    better on 19 of 24 batches. Declining the least predictable 15% cut kept-condition error by 27%.
    Intervals covered 70% at a nominal 80%; the quiet-condition early-warning claim did not replicate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models/v2"
LOCK = ROOT / "evaluation/v2_development/development_lock.json"
REQUIRED = ["date", "Plate.SN", "well", "DIV", "trt", "dose", "units"]
LEVEL = "0.8"

EVIDENCE = {
    "v1_reserve": {"batches": 3, "conditions": 224, "model_mae": 0.7716, "persistence_mae": 0.6088, "note": "v1 lost to persistence"},
    "external": {"batches": 24, "conditions": 927, "unseen_chemicals": 105, "model_mae": 0.735, "persistence_mae": 0.980,
                 "batches_better_than_persistence": 19, "kept_error_reduction": 0.27, "interval_coverage_at_nominal_0.8": 0.702},
}
LIMITS = [
    "Validated on EPA rat cortical cultures on 48-well microelectrode arrays; not on organ chips, human cells, or clinical toxicity.",
    "Intervals covered 70% of external outcomes at a nominal 80% level; read them as approximate.",
    "The batch rule marks a degenerate day-7 reference, not a failed forecast: it fired on 3 of 41 batches, and on 2 of them the forecast still beat carrying day 7 forward.",
    "Every externally declined condition was already strongly suppressed at day 7; a decline means recovery or progression is undetermined, not that there is no effect.",
    "Forecast magnitudes are not probabilities. Units are log2 of the same-plate control-relative network-spike count.",
]


def normalize(observed: pd.DataFrame, readouts: list[str]) -> pd.DataFrame:
    missing = [c for c in REQUIRED + readouts if c not in observed.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame = observed.copy()
    frame["date"] = frame["date"].astype(int)
    frame["DIV"] = frame["DIV"].astype(int)
    days = set(frame.DIV)
    if not days <= {5, 7}:
        raise ValueError(f"This tool decides at day 7; it rejects observations from days {sorted(days - {5, 7})}")
    if days != {5, 7}:
        raise ValueError("Both day-5 and day-7 observations are required by the v2 model")
    if set(frame.units) != {"uM"}:
        raise ValueError("Expected concentrations in uM")
    if frame.duplicated(data.PHYSICAL).any():
        raise ValueError("Duplicate physical observations must be audited before triage")
    if "identity" not in frame:
        frame["identity"] = frame.trt.astype(str)
    if "eligible_identity" not in frame:
        frame["eligible_identity"] = frame.identity.notna()
    frame["eligible_identity"] = frame.eligible_identity.astype(bool)
    return frame


def load_bundle(bundle: Path = BUNDLE, lock_path: Path = LOCK) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for name, digest in lock["models"].items():
        if data.digest(bundle / name) != digest:
            raise ValueError(f"{name} differs from the locked model")
    locked = lock["locked"]
    if locked["product"] != "small_refit" or locked["trust"] != "sigma_v2":
        raise ValueError("This tool implements the locked small_refit product with the sigma_v2 trust score")
    return {"product": joblib.load(bundle / "final_small_refit.joblib"), "sigma": joblib.load(bundle / "final_sigma_v2.joblib"),
            "threshold": locked["trust_threshold"], "quantile": locked["conformal_quantiles"][LEVEL]}


def triage(observed: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, dict]:
    cfg = v2.v2_config()
    raw = normalize(observed, cfg["feature_readouts"])
    features = v2.features_from(raw, cfg)
    if features.empty:
        raise ValueError(f"No treated condition has at least {cfg['min_replicates_per_case_day']} replicate wells on both day 5 and day 7; "
                         "replicates may sit on different plates of the same batch, each with its own zero-dose controls")
    with threadpool_limits(limits=1):
        forecast = bundle["product"]["model"].predict(features[bundle["product"]["columns"]])
        sigma = np.maximum(bundle["sigma"]["model"].predict(features[bundle["sigma"]["columns"]]), gate.SIGMA_FLOOR)
    half = bundle["quantile"] * sigma
    out = features.index.to_frame(index=False)
    out["case_id"] = gate.case_ids(features.index)
    names = raw.loc[raw.identity.notna()].groupby("identity").trt.agg(lambda x: sorted(set(map(str, x)))[0])
    out.insert(2, "name", out.identity.map(names).fillna(out.identity))
    out["replicates"] = features["rel__rep_count_d7"].to_numpy()
    out["day7_observed_log2"] = features.normalized_ns_d7.to_numpy()
    out["forecast_log2"] = forecast
    out["forecast_low"] = forecast - half
    out["forecast_high"] = forecast + half
    out["persistence_log2"] = features.normalized_ns_d7.to_numpy()
    out["difficulty"] = sigma
    out["forecast_trusted"] = sigma <= bundle["threshold"]
    out["verdict"] = np.where(out.forecast_trusted, "forecast usable", "declined: measure day 12 directly")
    out["quality_low_reference_activity"] = np.sinh(features["rel__plate_ctrl_median_ns_d7"].to_numpy()) < 1
    out["quality_few_controls"] = features["rel__plate_ctrl_count_d7"].to_numpy() < 4
    out["fold_change_estimate"] = 2 ** out.forecast_log2
    # Display rule: when half or more of a batch's conditions have a silent day-7 reference, none is shown as usable.
    # forecast_trusted keeps the validated difficulty verdict; only the printed verdict is overridden.
    low_batch = out.groupby("date").quality_low_reference_activity.transform("mean") >= 0.5
    out.loc[low_batch, "verdict"] = "batch reference unusable: measure day 12 directly"
    batches = []
    for date, group in out.groupby("date"):
        low = float(group.quality_low_reference_activity.mean())
        batches.append({
            "batch": int(date), "conditions": int(len(group)),
            "day7_reference_activity_network_spikes": float(np.sinh(features.loc[features.index.get_level_values("date") == date, "rel__plate_ctrl_median_ns_d7"]).median()),
            "forecasts_declined": int((~group.forecast_trusted).sum()),
            "low_reference_activity_fraction": low,
            "batch_verdict": ("day-7 reference activity is too low to normalize against; this batch's day-7 values and forecasts are unverified, so measure day 12 directly"
                              if low >= 0.5 else "day-7 measurement quality is usable"),
        })
    summary = {"decision_day": 7, "conditions": int(len(out)), "batches": batches, "interval_nominal_level": float(LEVEL),
               "evidence": EVIDENCE, "limits": LIMITS}
    return out, summary


MIN_HISTORY_BATCHES = 2


def conformal_quantile(scores, level: float) -> float:
    """The ceil((n+1) level)-th smallest score: the split-conformal quantile with the finite-sample correction."""
    s = np.sort(np.asarray(scores, dtype=float))
    rank = min(len(s), int(np.ceil((len(s) + 1) * level)))
    return float(s[rank - 1])


def lab_calibration(records: list[tuple[str, bytes]], exclude_batches: set[int], level: float = float(LEVEL)) -> dict:
    """Refit the interval scale from a lab's own checked admission records (name, bytes).

    Every scored condition (day-12 reference usable) of every record contributes |measured - forecast| / difficulty.
    Records must come from these locked models and carry a day-12 check; records of the batch being analysed are
    refused, so a batch never calibrates its own intervals."""
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    models = {"final_small_refit.joblib": lock["models"]["final_small_refit.joblib"],
              "final_sigma_v2.joblib": lock["models"]["final_sigma_v2.joblib"], "development_lock.json": data.digest(LOCK)}
    scores, batches, sources = [], set(), []
    for name, raw in records:
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{name}: not a JSON admission record ({error})") from None
        if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
            raise ValueError(f"{name}: not a {RECORD_SCHEMA} record")
        if record.get("locked_models") != models:
            raise ValueError(f"{name}: made with different models; only records from these locked models can recalibrate them")
        if not record.get("day12_check"):
            raise ValueError(f"{name}: has no day-12 check; run it with --day12 first")
        own = sorted({int(c["batch"]) for c in record["conditions"]} & exclude_batches)
        if own:
            raise ValueError(f"{name}: is a record of batch {own[0]}, the batch being analysed; a batch cannot calibrate itself")
        used = 0
        for c in record["conditions"]:
            check = c.get("day12") or {}
            measured, forecast, difficulty = check.get("measured_log2"), c.get("forecast_log2"), c.get("difficulty")
            if check.get("reference_usable") and measured is not None and forecast is not None and difficulty:
                scores.append(abs(measured - forecast) / difficulty)
                batches.add(int(c["batch"]))
                used += 1
        sources.append({"file": name, "sha256": hashlib.sha256(raw).hexdigest(), "conditions_used": used})
    if len(batches) < MIN_HISTORY_BATCHES:
        raise ValueError(f"Recalibration needs checked records from at least {MIN_HISTORY_BATCHES} batches; these cover {len(batches)}")
    shipped = json.loads(LOCK.read_text(encoding="utf-8"))["locked"]["conformal_quantiles"][LEVEL]
    scores = np.asarray(scores)
    return {"method": "split-conformal quantile of |measured - forecast| / difficulty over the history's scored conditions",
            "level": level, "shipped_scale": float(shipped), "scale": conformal_quantile(scores, level),
            "history_batches": sorted(batches), "history_conditions": int(len(scores)),
            "history_coverage_at_shipped_scale": float((scores <= shipped).mean()), "records": sources}


def recalibrate(table: pd.DataFrame, calibration: dict) -> pd.DataFrame:
    """Intervals at the lab's scale; forecasts, difficulty and verdicts are unchanged."""
    out = table.copy()
    half = calibration["scale"] * out.difficulty
    out["forecast_low"] = out.forecast_log2 - half
    out["forecast_high"] = out.forecast_log2 + half
    return out


def render_calibration(calibration: dict) -> str:
    return (f"Intervals recalibrated on this lab's own day-12 checks: {calibration['history_conditions']} conditions from "
            f"{len(calibration['history_batches'])} earlier batches.\n  80% scale {calibration['shipped_scale']:.3f} (shipped) -> "
            f"{calibration['scale']:.3f}; at the shipped scale those outcomes were covered "
            f"{calibration['history_coverage_at_shipped_scale']:.0%} of the time. Forecasts and verdicts are unchanged.")


def normalize_day12(observed: pd.DataFrame, readouts: list[str]) -> pd.DataFrame:
    """Day-12 recordings for the check. They are read only after the day-7 report exists and never reach a model."""
    missing = [c for c in REQUIRED + readouts if c not in observed.columns]
    if missing:
        raise ValueError(f"Missing columns in the day-12 file: {missing}")
    frame = observed.copy()
    frame["date"] = frame["date"].astype(int)
    frame["DIV"] = frame["DIV"].astype(int)
    frame = frame.loc[frame.DIV == 12]
    if frame.empty:
        raise ValueError("The day-12 file holds no day-12 recordings")
    if set(frame.units) != {"uM"}:
        raise ValueError("Expected concentrations in uM")
    if frame.duplicated(data.PHYSICAL).any():
        raise ValueError("Duplicate physical observations must be audited before the check")
    if "identity" not in frame:
        frame["identity"] = frame.trt.astype(str)
    if "eligible_identity" not in frame:
        frame["eligible_identity"] = frame.identity.notna()
    frame["eligible_identity"] = frame.eligible_identity.astype(bool)
    return frame


def _mean(values: pd.Series) -> float | None:
    return float(values.mean()) if len(values) else None


def day12_check(table: pd.DataFrame, observed12: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Score a day-7 report against day 12, with the sealed test's own target and reference definitions."""
    cfg = v2.v2_config()
    raw = normalize_day12(observed12, cfg["feature_readouts"])
    target = data.build_targets(raw, cfg)
    reference = gate.outcome_quality(raw, cfg).reindex(target.index)
    measured = target.index.to_frame(index=False)
    measured["day12_measured_log2"] = target.to_numpy()
    measured["day12_reference_network_spikes"] = reference.to_numpy()
    out = table.merge(measured, on=data.KEY, how="left", validate="one_to_one")
    out["day12_reference_usable"] = out.day12_reference_network_spikes >= v2.USABLE_MIN
    out["scored"] = out.day12_measured_log2.notna() & out.day12_reference_usable
    out["forecast_abs_error"] = (out.forecast_log2 - out.day12_measured_log2).abs()
    out["carry_forward_abs_error"] = (out.persistence_log2 - out.day12_measured_log2).abs()
    out["inside_interval80"] = (out.day12_measured_log2 >= out.forecast_low) & (out.day12_measured_log2 <= out.forecast_high)
    batches = []
    for date, group in out.groupby("date"):
        s = group.loc[group.scored]
        chemical = lambda column: s[column].groupby(s.identity).mean().mean() if len(s) else None
        provisional = s.loc[s.verdict == "forecast usable"]
        declined = s.loc[s.verdict != "forecast usable"]
        batches.append({
            "batch": int(date), "conditions_in_report": int(len(group)),
            "conditions_measured": int(group.day12_measured_log2.notna().sum()),
            "conditions_scored": int(len(s)),
            "reference_unusable": int((group.day12_measured_log2.notna() & ~group.day12_reference_usable).sum()),
            "forecast_error": chemical("forecast_abs_error"), "carry_forward_error": chemical("carry_forward_abs_error"),
            "forecast_closer": int((s.forecast_abs_error < s.carry_forward_abs_error).sum()),
            "interval80_coverage": _mean(s.inside_interval80.astype(float)),
            "provisional": {"conditions": int(len(provisional)), "forecast_error": _mean(provisional.forecast_abs_error),
                            "carry_forward_error": _mean(provisional.carry_forward_abs_error)},
            "declined": {"conditions": int(len(declined)), "forecast_error": _mean(declined.forecast_abs_error),
                         "carry_forward_error": _mean(declined.carry_forward_abs_error)},
        })
    check = {
        "definition": ("Day-12 truth is the sealed test's target: log2 of the replicate mean of (network spikes + 1) / "
                       "(same-plate day-12 control median + 1). A condition is scored when its day-12 control median is at "
                       f"least {v2.USABLE_MIN:g} network spikes. Batch errors are mean absolute errors averaged within each "
                       "chemical, then over chemicals; provisional and declined errors are plain means over conditions."),
        "batches": batches,
    }
    return out, check


def render_check(check: dict) -> str:
    lines = ["", "Day-12 check (scored after day 12 was recorded; nothing here changes the day-7 report):"]
    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
    for b in check["batches"]:
        lines += [f"  Batch {b['batch']}: {b['conditions_scored']} of {b['conditions_in_report']} conditions scored "
                  f"({b['reference_unusable']} with an unusable day-12 reference)",
                  f"    error: forecast {fmt(b['forecast_error'])} vs carry day 7 forward {fmt(b['carry_forward_error'])}; "
                  f"forecast closer on {b['forecast_closer']} of {b['conditions_scored']}",
                  f"    80% intervals held {fmt(b['interval80_coverage'])} of outcomes",
                  f"    provisional ({b['provisional']['conditions']}): forecast {fmt(b['provisional']['forecast_error'])} vs carry-forward {fmt(b['provisional']['carry_forward_error'])}; "
                  f"declined ({b['declined']['conditions']}): forecast {fmt(b['declined']['forecast_error'])} vs carry-forward {fmt(b['declined']['carry_forward_error'])}"]
    return "\n".join(lines)


RECORD_SCHEMA = "neuroforecast-admission-record/1"


def _finite(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def condition_decision(row) -> tuple[str, list[str]]:
    reasons = []
    if row.verdict.startswith("batch reference unusable"):
        reasons.append("batch_reference_unusable")
    if not row.forecast_trusted:
        reasons.append("difficulty_above_threshold")
    if row.quality_low_reference_activity:
        reasons.append("low_reference_activity")
    if row.quality_few_controls:
        reasons.append("few_controls")
    return ("provisional forecast" if row.verdict == "forecast usable" else "measure day 12"), reasons


def admission_record(table: pd.DataFrame, summary: dict, source: str, input_bytes: bytes,
                     check: dict | None = None, check_source: str | None = None, check_bytes: bytes | None = None,
                     calibration: dict | None = None) -> dict:
    """What a lab files with the data: which forecasts may stand in provisionally, which must be measured, and why."""
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    conditions = []
    for row in table.sort_values(["date", "name", "dose"]).itertuples(index=False):
        decision, reasons = condition_decision(row)
        entry = {"batch": int(row.date), "chemical": str(row.name), "identity": str(row.identity), "dose_uM": float(row.dose),
                 "day7_log2": _finite(row.day7_observed_log2), "forecast_log2": _finite(row.forecast_log2),
                 "interval80_log2": [_finite(row.forecast_low), _finite(row.forecast_high)],
                 "carry_forward_log2": _finite(row.persistence_log2), "difficulty": _finite(row.difficulty),
                 "decision": decision, "reasons": reasons}
        if check is not None:
            entry["day12"] = {"measured_log2": _finite(row.day12_measured_log2),
                              "reference_usable": bool(row.day12_reference_usable),
                              "forecast_abs_error": _finite(row.forecast_abs_error),
                              "carry_forward_abs_error": _finite(row.carry_forward_abs_error),
                              "inside_interval80": bool(row.inside_interval80) if pd.notna(row.day12_measured_log2) else None}
        conditions.append(entry)
    batches = []
    for b in summary["batches"]:
        group = table.loc[table.date == b["batch"]]
        usable = b["low_reference_activity_fraction"] < 0.5
        batches.append({"batch": b["batch"], "verdict": "usable" if usable else "reference unusable",
                        "reasons": [] if usable else ["low_reference_activity"],
                        "conditions": b["conditions"], "chemicals": int(group.identity.nunique()),
                        "declined": b["forecasts_declined"],
                        "provisional": int((group.verdict == "forecast usable").sum()),
                        "day7_reference_network_spikes": _finite(b["day7_reference_activity_network_spikes"]),
                        "low_reference_fraction": _finite(b["low_reference_activity_fraction"])})
    record = {
        "schema": RECORD_SCHEMA,
        "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": {"name": "NeuroForecast", "model": "v2", "implementation": "python"},
        "locked_models": {"final_small_refit.joblib": lock["models"]["final_small_refit.joblib"],
                          "final_sigma_v2.joblib": lock["models"]["final_sigma_v2.joblib"],
                          "development_lock.json": data.digest(LOCK)},
        "input": {"file": source, "sha256": hashlib.sha256(input_bytes).hexdigest(), "days": [5, 7]},
        "decision_rule": {"trust_threshold": float(bundle_threshold()), "interval_level": float(LEVEL),
                          "interval_scale": float(calibration["scale"] if calibration else
                                                  json.loads(LOCK.read_text(encoding="utf-8"))["locked"]["conformal_quantiles"][LEVEL]),
                          "batch_rule": "reference unusable when at least half the conditions sit on plates whose day-7 control median is below one network spike"},
        "interval_calibration": calibration,
        "batches": batches,
        "conditions": conditions,
        "day12_check": None,
        "evidence": EVIDENCE,
        "limits": LIMITS,
    }
    if check is not None:
        record["day12_check"] = {"input": {"file": check_source, "sha256": hashlib.sha256(check_bytes).hexdigest(), "days": [12]},
                                 "definition": check["definition"],
                                 "batches": [{k: _clean(v) for k, v in b.items()} for b in check["batches"]]}
    return record


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return _finite(value) if value is not None else None


def bundle_threshold() -> float:
    return json.loads(LOCK.read_text(encoding="utf-8"))["locked"]["trust_threshold"]


def render(out: pd.DataFrame, summary: dict) -> str:
    lines = [f"NeuroForecast v2 triage - decision at day 7, {summary['conditions']} conditions"]
    for batch in summary["batches"]:
        group = out.loc[out.date == batch["batch"]]
        lines += ["", f"Batch {batch['batch']}: {batch['conditions']} conditions, day-7 reference activity {batch['day7_reference_activity_network_spikes']:.1f} network spikes",
                  f"  measurement quality: {batch['batch_verdict']}",
                  f"  {batch['forecasts_declined']} of {batch['conditions']} forecasts declined as unpredictable", ""]
        lines.append(f"    {'chemical':<26} {'uM':>8}  {'day 7':>7} {'forecast':>9} {'80% interval':>18}  verdict")
        for _, row in group.sort_values(["name", "dose"]).iterrows():
            interval = f"[{row.forecast_low:+.2f}, {row.forecast_high:+.2f}]"
            lines.append(f"    {str(row['name'])[:26]:<26} {row.dose:>8.3g}  {row.day7_observed_log2:>+7.2f} {row.forecast_log2:>+9.2f} {interval:>18}  {row.verdict}")
    e = summary["evidence"]
    lines += ["", "Held-out evidence (two sealed tests, each scored once):",
              f"  3 later batches (v1 model): persistence {e['v1_reserve']['persistence_mae']:.4f} beat the model {e['v1_reserve']['model_mae']:.4f}",
              f"  24 earlier batches, 105 unseen chemicals (this model): {e['external']['model_mae']:.3f} versus persistence {e['external']['persistence_mae']:.3f}, better on {e['external']['batches_better_than_persistence']} of 24",
              f"  intervals covered {e['external']['interval_coverage_at_nominal_0.8']:.0%} at a nominal 80% level"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Well-level CSV of day-5 and day-7 observations and zero-dose controls")
    parser.add_argument("--output", type=Path, help="Per-condition table (CSV)")
    parser.add_argument("--summary", type=Path, help="Batch summary (JSON)")
    parser.add_argument("--day12", type=Path, help="Later: well-level day-12 CSV of the same batch, to check the day-7 report")
    parser.add_argument("--record", type=Path, help="Admission record (JSON): decisions, reasons, model and input hashes")
    parser.add_argument("--history", type=Path, nargs="+", metavar="RECORD",
                        help="This lab's earlier admission records with a day-12 check: refit the interval scale from them")
    args = parser.parse_args(argv)
    # The frozen feature code builds wide frames column by column; pandas warns about speed, not correctness.
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    try:
        table, report = triage(pd.read_csv(args.input, dtype={"identity": str}), load_bundle())
        calibration = None
        if args.history:
            calibration = lab_calibration([(p.name, p.read_bytes()) for p in args.history], set(int(d) for d in table.date))
            table = recalibrate(table, calibration)
        print(render(table, report))
        if calibration:
            print("\n" + render_calibration(calibration))
        check = None
        if args.day12:
            table, check = day12_check(table, pd.read_csv(args.day12, dtype={"identity": str}))
            print(render_check(check))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        # Input problems (missing file, wrong columns or days, duplicates, too few replicates) are reported, not traced.
        parser.exit(2, f"error: {error}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.record:
        record = admission_record(table, report, args.input.name, args.input.read_bytes(), check,
                                  args.day12.name if args.day12 else None, args.day12.read_bytes() if args.day12 else None,
                                  calibration)
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
