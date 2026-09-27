"""NeuroForecast v2 day-7 triage: batch quality, a day-12 forecast beside persistence, and a trust verdict.

Input: well-level day-5 and day-7 MEA readouts with same-plate zero-dose controls (days 9 and 12 are rejected).
Output, per condition: the v2 forecast of the day-12 control-relative network-spike readout (log2), an 80%
interval, the carry-day-7-forward value, a difficulty score and a trust verdict; per batch: a quality verdict.

Evidence (two sealed tests, each scored once):
  * v1 reserve, 3 later batches: persistence 0.6088 beat the v1 model 0.7716.
  * External, 24 earlier batches, 105 unseen chemicals: this v2 model 0.735 vs persistence 0.980,
    better on 19 of 24 batches. Declining the least predictable 15% cut kept-condition error by 27%.
    Intervals covered 70% at a nominal 80%; the quiet-condition early-warning claim did not replicate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Well-level CSV of day-5 and day-7 observations and zero-dose controls")
    parser.add_argument("--output", type=Path, help="Per-condition table (CSV)")
    parser.add_argument("--summary", type=Path, help="Batch summary (JSON)")
    args = parser.parse_args()
    table, report = triage(pd.read_csv(args.input, dtype={"identity": str}), load_bundle())
    print(render(table, report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
