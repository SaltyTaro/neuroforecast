"""Post-hoc sensitivity: how the locked v2 tool degrades with fewer untreated control wells per plate.

A neural-chip run has far fewer vehicle-treated chips than a 48-well plate has control wells. This keeps k
randomly chosen zero-dose wells per plate at days 5 and 7 (k = 1, 2, 3, 4, all), rebuilds the day-7 inputs, and
applies the locked v2 models. Targets keep their full day-12 controls, so only the day-7 information changes.

It runs on the external dates, which were already scored once; this is a descriptive robustness analysis of the
locked tool, not a validation, and it selects nothing. Needs the EPA refinement files (fetch_epa_refinement.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import external_nfa_intake as intake  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_triage_v2 as t2  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]


def subsample(raw: pd.DataFrame, k: int | None, seed: int) -> pd.DataFrame:
    if k is None:
        return raw
    # The same physical wells are kept at day 5 and day 7: a well is date + plate + well.
    controls = raw.loc[raw.dose == 0, ["date", "Plate.SN", "well"]].drop_duplicates()
    shuffled = controls.sample(frac=1.0, random_state=seed)
    keep = shuffled.loc[shuffled.groupby(["date", "Plate.SN"]).cumcount() < k]
    kept = set(map(tuple, keep.to_numpy()))
    is_control = raw.dose == 0
    mask = ~is_control | pd.Series([t in kept for t in map(tuple, raw[["date", "Plate.SN", "well"]].to_numpy())], index=raw.index)
    return raw.loc[mask]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/v2_posthoc/control_subsampling.json")
    args = parser.parse_args()
    intake.RAW_DIR = args.raw
    raw = v2.release_frame()
    external = raw.loc[~raw.date.isin(v2.DEV_DATES) & raw.DIV.isin([5, 7])]
    scored = pd.read_csv(ROOT / "evaluation/external/arm_b_cases.csv", dtype={"identity": str})
    scored = scored.loc[scored.primary.astype(bool) & scored.usable.astype(bool), ["case_id", "date", "identity", "dose", "target"]]
    bundle = t2.load_bundle()
    rows = []
    for k in [1, 2, 3, 4, None]:
        for seed in (SEEDS if k is not None else [0]):
            table, summary = t2.triage(subsample(external, k, seed), bundle)
            frame = scored.merge(table[["case_id", "forecast_log2", "persistence_log2", "forecast_trusted", "quality_low_reference_activity"]], on="case_id", how="inner")
            product = gate.macro_mae(frame.assign(prediction=frame.forecast_log2))
            persistence = gate.macro_mae(frame.assign(prediction=frame.persistence_log2))
            error = abs(frame.forecast_log2 - frame.target)
            keep = frame.forecast_trusted.astype(bool)
            rows.append({"controls_per_plate": "all" if k is None else k, "seed": seed, "conditions_scored": int(len(frame)),
                         "product_date_macro": product, "persistence_date_macro": persistence,
                         "declined_share": float((~keep).mean()), "kept_error_reduction": float(1 - error[keep].mean() / error.mean()),
                         "low_reference_batches": int(sum(b["low_reference_activity_fraction"] >= 0.5 for b in summary["batches"]))})
    detail = pd.DataFrame(rows)
    summary = detail.groupby("controls_per_plate", sort=False).agg(
        conditions_scored=("conditions_scored", "mean"), product_date_macro=("product_date_macro", "mean"),
        persistence_date_macro=("persistence_date_macro", "mean"), declined_share=("declined_share", "mean"),
        kept_error_reduction=("kept_error_reduction", "mean"), low_reference_batches=("low_reference_batches", "mean")).reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"note": __doc__.strip().splitlines()[0], "seeds": SEEDS, "summary": summary.to_dict("records"),
                                       "detail": detail.to_dict("records")}, indent=2) + "\n", encoding="utf-8")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
