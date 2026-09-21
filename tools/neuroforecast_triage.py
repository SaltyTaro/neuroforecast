"""NeuroForecast trust triage: from day-7 MEA wells, report measurement quality, a day-12 forecast beside its comparators, whether that forecast can be trusted, and a ranked review list.

The validated claim is triage, not forecasting accuracy. On three held-out experiment dates with
32 unseen chemicals, carrying the day-7 readout forward was more accurate than the learned model
(0.6088 versus 0.7716 date-macro MAE). What generalized is the reliability layer, so this tool
always prints the simple comparator beside the model and marks conditions it cannot forecast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import neuroforecast_data as data
import neuroforecast_gate as gate

def _defaults() -> tuple[Path, Path, Path]:
    """Prefer a model bundle shipped beside the code, so a fresh clone runs without the experiment directory."""
    local = data.ROOT / "models"
    root = local if (local / "final_sigma.joblib").exists() else gate.DEFAULT_OUT
    bundle = local if root is local else root / "models"
    return bundle, root / "development_lock.json", root / "reserve_summary.json"


DEFAULT_BUNDLE, DEFAULT_LOCK, DEFAULT_RESERVE = _defaults()
PRODUCT = "extra_trees_relative_day7"
COMPARATOR = "extra_trees_activity_coordination"
ALLOWED_DAYS = {5, 7}
DECISION_DAY = 7
QUIET_LIMIT = 0.5  # the predefined "has barely moved by day 7" boundary, in log2 relative units


def contrasts(day: pd.DataFrame, cfg: dict, suffix: int) -> dict:
    columns = {}
    for field in cfg["feature_readouts"]:
        raw = day[field].to_numpy(dtype=float)
        control = day["control__" + field].to_numpy(dtype=float)
        denominator = abs(raw) + abs(control)
        columns[f"contrast__{field}__d{suffix}"] = np.divide(raw - control, denominator, out=np.zeros(len(day)), where=denominator != 0)
        columns[f"missing__{field}__d{suffix}"] = day["missing__" + field].to_numpy(dtype=float)
    return columns


def build_features(observed: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, bool]:
    """Day-7 features, plus day-5 features when they are supplied. Nothing after day 7 is accepted."""
    days = set(observed.DIV)
    if not days <= ALLOWED_DAYS:
        raise ValueError(f"This tool decides at day {DECISION_DAY}; it rejects observations from days {sorted(days - ALLOWED_DAYS)}")
    if DECISION_DAY not in days:
        raise ValueError("Day-7 observations are required")
    if observed.duplicated(data.PHYSICAL).any():
        raise ValueError("Duplicate physical observations must be audited before triage")
    if set(observed.units) != {"uM"}:
        raise ValueError("Expected concentrations in uM")
    if observed["eligible_identity"].dtype != bool:
        raise ValueError("eligible_identity must contain audited Boolean values")
    day7 = data._observed_day(observed.loc[observed.DIV == 7], 7, cfg)
    frame = pd.DataFrame({"normalized_ns_d7": day7.normalized_ns, "log10_dose": np.log10(day7.index.get_level_values("dose").to_numpy(dtype=float))}, index=day7.index)
    for name, values in contrasts(day7, cfg, 7).items():
        frame[name] = values
    has_day5 = 5 in days
    if has_day5:
        day5 = data._observed_day(observed.loc[observed.DIV == 5], 5, cfg)
        early = pd.DataFrame({"normalized_ns_d5": day5.normalized_ns}, index=day5.index)
        for name, values in contrasts(day5, cfg, 5).items():
            early[name] = values
        frame = frame.join(early, how="left")
        has_day5 = frame.normalized_ns_d5.notna().any()
    rel = gate.reliability_indicators(observed.loc[observed.DIV == 7], cfg).reindex(frame.index)
    rel.pop("rel__plates_spanned")
    frame = frame.join(rel[gate.RELIABILITY])
    return frame.replace([np.inf, -np.inf], np.nan), bool(has_day5)


def load_bundle(bundle: Path, lock_path: Path, reserve_path: Path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    reserve = json.loads(reserve_path.read_text(encoding="utf-8"))
    models = {name: joblib.load(bundle / f"final_{name}.joblib") for name in [PRODUCT, COMPARATOR]}
    sigma = joblib.load(bundle / "final_sigma.joblib")
    if sigma["floor"] != gate.SIGMA_FLOOR:
        raise ValueError("Difficulty model floor does not match the locked gate configuration")
    level = lock["primary_level"]
    return {
        "models": models,
        "sigma": sigma,
        "sigma_threshold": lock["locked_thresholds"]["abstention_sigma_p80"],
        "flag_thresholds": lock["locked_thresholds"],
        "level": level,
        "scaled_quantile": reserve["conformal_quantiles"][f"scaled_q_{level}"],
        "budget": lock["primary_budget"],
        "evidence": {
            "reserve_dates": reserve["reserve_dates"],
            "reserve_cases": reserve["reserve_cases"],
            "product_mean_mae": reserve["date_macro_mae"][PRODUCT],
            "persistence_mean_mae": reserve["date_macro_mae"]["persistence_day7"],
            "comparator_mean_mae": reserve["date_macro_mae"][COMPARATOR],
            "observed_coverage_at_nominal": reserve["coverage"][f"scaled_{level}"]["pooled_observed_coverage"],
            "retained_case_mae": reserve["abstention"]["retained_case_mae"],
            "all_case_mae": reserve["abstention"]["all_case_mae"],
        },
    }


def quality_flags(frame: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "low_reference_activity": np.sinh(frame["rel__plate_ctrl_median_ns_d7"]) < 1,
        "few_controls": frame["rel__plate_ctrl_count_d7"] < 4,
        "replicates_disagree": frame["rel__rep_sd_log2_ns_d7"] > thresholds["rep_sd_log2_ns_d7_p90"],
        "controls_disagree": frame["rel__plate_ctrl_iqr_ratio_d7"] > thresholds["plate_ctrl_iqr_ratio_d7_p90"],
    }, index=frame.index)


def rank_within(out: pd.DataFrame, prefix: str, subset: pd.Index, budget: float = 0.2) -> pd.DataFrame:
    """Rank by forecast magnitude inside each batch, over the given conditions, and flag the top budget share."""
    rank = pd.Series(pd.NA, index=out.index, dtype="Int64")
    flagged = pd.Series(False, index=out.index)
    for _, group in out.loc[subset].groupby("date"):
        order = abs(group.forecast_log2).rank(ascending=False, method="first").astype(int)
        rank.loc[group.index] = order
        flagged.loc[group.index] = order <= int(np.ceil(budget * len(group)))
    out[f"{prefix}_rank"] = rank
    out[f"{prefix}_flagged"] = flagged
    return out


def triage(observed: pd.DataFrame, bundle: dict, budget: float | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = data.protocol()
    frame, has_day5 = build_features(observed, cfg)
    budget = bundle["budget"] if budget is None else budget
    product = bundle["models"][PRODUCT]
    forecast = product["model"].predict(frame[product["columns"]])
    sigma = np.maximum(bundle["sigma"]["model"].predict(frame[bundle["sigma"]["columns"]]), gate.SIGMA_FLOOR)
    half_width = bundle["scaled_quantile"] * sigma
    persistence = frame.normalized_ns_d7.to_numpy()

    out = frame.index.to_frame(index=False)
    out["case_id"] = gate.case_ids(frame.index)
    names = observed.loc[observed.identity.notna()].groupby("identity").trt.agg(lambda x: sorted(set(x))[0])
    out.insert(2, "name", out.identity.map(names).fillna(out.identity))
    out["replicates"] = frame["rel__rep_count_d7"].to_numpy()
    out["day7_observed_log2"] = persistence
    out["forecast_log2"] = forecast
    out["forecast_low"] = forecast - half_width
    out["forecast_high"] = forecast + half_width
    out["persistence_log2"] = persistence
    if has_day5:
        comparator = bundle["models"][COMPARATOR]
        out["comparator_log2"] = comparator["model"].predict(frame[comparator["columns"]])
    else:
        out["comparator_log2"] = np.nan
    out["difficulty"] = sigma
    out["forecast_trusted"] = sigma <= bundle["sigma_threshold"]
    out["verdict"] = np.where(out.forecast_trusted, "forecast usable", "declined: measure day 12 directly")

    flags = quality_flags(frame, bundle["flag_thresholds"]).reset_index(drop=True)
    for column in flags.columns:
        out[f"quality_{column}"] = flags[column].to_numpy()
    out["quality_any_flag"] = flags.any(axis=1).to_numpy()

    # Two ranked lists, per batch, exactly as validated on the reserve: over every condition,
    # and within the conditions that have barely moved by day 7, where the naive rule is blind.
    out["already_changed_by_day7"] = abs(out.day7_observed_log2) >= QUIET_LIMIT
    out = rank_within(out, "review", out.index, budget=budget)
    out = rank_within(out, "early_warning", out.index[~out.already_changed_by_day7], budget=budget)
    out["fold_change_estimate"] = 2 ** out.forecast_log2

    batches = []
    for date, group in out.groupby("date"):
        flagged = group.quality_low_reference_activity.mean()
        batches.append({
            "batch": int(date),
            "conditions": int(len(group)),
            "median_replicates": float(group.replicates.median()),
            "day7_reference_activity_network_spikes": float(np.sinh(frame.loc[frame.index.get_level_values("date") == date, "rel__plate_ctrl_median_ns_d7"]).median()),
            "conditions_with_any_quality_flag": int(group.quality_any_flag.sum()),
            "low_reference_activity_fraction": float(flagged),
            "forecasts_declined": int((~group.forecast_trusted).sum()),
            "review_list_size": int(group.review_flagged.sum()),
            "quiet_at_day7": int((~group.already_changed_by_day7).sum()),
            "early_warning_list_size": int(group.early_warning_flagged.sum()),
            "batch_verdict": ("day-7 reference activity is too low for this batch; forecasts here are unreliable and a day-12 measurement is recommended" if flagged >= 0.5 else "day-7 measurement quality is usable"),
        })
    summary = {
        "decision_day": DECISION_DAY,
        "conditions": int(len(out)),
        "batches": batches,
        "review_budget": budget,
        "interval_nominal_level": bundle["level"],
        "day5_supplied_so_small_model_comparator_available": has_day5,
        "validated_evidence": bundle["evidence"],
        "honest_limits": [
            "On three held-out experiment dates with 32 unseen chemicals, carrying the day-7 readout forward was more accurate than this model (0.6088 versus 0.7716). Read the forecast beside the persistence column, not instead of it.",
            "Prediction intervals observed 71.4% coverage at a nominal 80% level on those dates; they are approximate under batch shift.",
            "Declining a forecast halved the error of the conditions kept (0.446 versus 0.784 for all cases) and also improves persistence, so the trust verdict applies to either predictor.",
            "Forecast magnitudes are not calibrated probabilities. Units are log2 of the same-plate control-relative network-spike readout.",
            "Validated on public EPA rat cortical cultures on microelectrode arrays. Not validated on organ chips, human cells, or clinical toxicity.",
        ],
    }
    return out, summary


def render(out: pd.DataFrame, summary: dict) -> str:
    lines = [f"NeuroForecast trust triage - decision at day {summary['decision_day']}, {summary['conditions']} conditions"]
    for batch in summary["batches"]:
        lines.append("")
        lines.append(f"Batch {batch['batch']}: {batch['conditions']} conditions, day-7 reference activity {batch['day7_reference_activity_network_spikes']:.1f} network spikes")
        lines.append(f"  measurement quality: {batch['batch_verdict']}")
        lines.append(f"  {batch['conditions_with_any_quality_flag']} conditions carry a quality flag; {batch['forecasts_declined']} forecasts declined; {batch['quiet_at_day7']} still quiet at day 7")
        group = out.loc[out.date == batch["batch"]]
        for title, rank_column, flag_column, note in [
            ("Early warning: quiet at day 7, forecast to change", "early_warning_rank", "early_warning_flagged", "this is where the day-7 readout carries no information and the forecast added the most on held-out batches"),
            ("Largest forecast change overall", "review_rank", "review_flagged", "mostly conditions that have already changed by day 7, which the day-7 column shows directly"),
        ]:
            head = group.loc[group[flag_column]].sort_values(rank_column)
            lines.append("")
            lines.append(f"  {title} ({len(head)} of {int(group[rank_column].notna().sum())} conditions) - {note}")
            if not len(head):
                continue
            lines.append(f"    {'#':>2}  {'chemical':<26} {'uM':>8}  {'day 7':>7} {'forecast':>9} {'80% interval':>18}  {'persistence':>11}  verdict")
            for _, row in head.iterrows():
                interval = f"[{row.forecast_low:+.2f}, {row.forecast_high:+.2f}]"
                lines.append(f"    {row[rank_column]:>2}  {str(row['name'])[:26]:<26} {row.dose:>8.3g}  {row.day7_observed_log2:>+7.2f} {row.forecast_log2:>+9.2f} {interval:>18}  {row.persistence_log2:>+11.2f}  {row.verdict}")
    lines.append("")
    evidence = summary["validated_evidence"]
    lines.append(f"Held-out evidence ({len(evidence['reserve_dates'])} experiment dates, {evidence['reserve_cases']} conditions, chemicals unseen in training):")
    lines.append(f"  forecast error {evidence['product_mean_mae']:.3f} versus persistence {evidence['persistence_mean_mae']:.3f} - the simple rule was more accurate")
    lines.append(f"  keeping only trusted forecasts: {evidence['retained_case_mae']:.3f} versus {evidence['all_case_mae']:.3f} for all conditions")
    lines.append(f"  intervals observed {evidence['observed_coverage_at_nominal']:.1%} coverage at a nominal {summary['interval_nominal_level']:.0%} level")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triage day-7 MEA observations: measurement quality, forecast with comparators, trust verdict, review list.")
    parser.add_argument("--input", type=Path, required=True, help="Audited well-level CSV of day-7 (optionally also day-5) observations and their zero-dose controls")
    parser.add_argument("--output", type=Path, help="Where to write the per-condition triage table")
    parser.add_argument("--summary", type=Path, help="Where to write the batch-level JSON summary")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--reserve", type=Path, default=DEFAULT_RESERVE)
    parser.add_argument("--budget", type=float, help="Review budget per batch; the validated operating point is used when omitted")
    args = parser.parse_args()
    bundle = load_bundle(args.bundle, args.lock, args.reserve)
    table, report = triage(pd.read_csv(args.input, dtype={"identity": str}), bundle, args.budget)
    print(render(table, report), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
