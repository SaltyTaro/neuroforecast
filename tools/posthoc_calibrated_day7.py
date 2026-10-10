"""Post hoc baseline: a calibrated day-7 curve (isotonic regression of day 12 on day 7 alone).

Carrying day 7 forward ignores the drift every culture shows between days 7 and 12; silent wells, for example, stay
silent while controls keep maturing. A one-variable monotone fit of day 12 on day 7 learns that drift without any
other readout, so it measures how much of v2's gain needs more than the day-7 value itself.

Both settings reuse saved evidence only; nothing sealed is re-scored and no locked criterion changes.
  * 41 batches, the v3 study's outer folds: the curve is fit on each fold's training batches, with the test fold's
    chemicals purged, exactly as every v3 candidate was.
  * 24 external batches: the curve is fit on the 17-batch v2 development set and applied once.

Two related checks of where v2's gain comes from:
  * kept conditions only: v2 against the curve on the external conditions v2 did not decline, and the share of v2's
    pooled gain over carry-forward that sits in the declined conditions and in strongly lowered day-7 values;
  * batch difficulty: whether the shipped per-condition difficulty score, averaged over a batch, ranks batches by
    their realised forecast error (Spearman correlation of batch means against date-macro MAE).

    python -X utf8 -B tools/posthoc_calibrated_day7.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/posthoc_baselines/calibrated_day7.json"


def date_losses(frame: pd.DataFrame, column: str) -> pd.Series:
    error = (frame[column] - frame.target).abs()
    return error.groupby([frame.date, frame.identity]).mean().groupby(level=0).mean()


def paired(a: pd.Series, b: pd.Series, seed: int) -> list[float]:
    """Date-paired bootstrap of mean(a) - mean(b), 10,000 resamples."""
    index = np.random.default_rng(seed).integers(len(a), size=(10000, len(a)))
    diff = a.to_numpy()[index].mean(axis=1) - b.to_numpy()[index].mean(axis=1)
    return np.quantile(diff, [0.025, 0.975]).tolist()


def curve(x: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(x, y)


def study_cases() -> pd.DataFrame:
    """41-batch outer-fold predictions of the v3 study, with the calibrated curve added fold by fold."""
    table = pd.read_csv(ROOT / "evaluation/v3/analysis/outer_case_table.csv", dtype={"identity": str})
    usable = table.usable.astype(bool)
    table["calibrated_day7"] = np.nan
    for fold in sorted(table.outer_fold.unique()):
        test = table.outer_fold == fold
        # Purge every chemical tested on the test batches, usable target or not, exactly as the v3 study did.
        train = ~test & ~table.identity.isin(table.loc[test, "identity"]) & usable
        fitted = curve(table.loc[train, "day7"].to_numpy(), table.loc[train, "target"].to_numpy())
        table.loc[test, "calibrated_day7"] = fitted.predict(table.loc[test, "day7"].to_numpy())
    return table.loc[usable].copy()


def external_cases() -> pd.DataFrame:
    dev = pd.read_csv(ROOT / "evaluation/v2_development/dev_oof_predictions.csv", dtype={"identity": str})
    dev = dev.loc[dev.usable.astype(bool)]
    cases = pd.read_csv(ROOT / "evaluation/external/arm_b_cases.csv", dtype={"identity": str})
    cases = cases.loc[cases.primary.astype(bool) & cases.usable.astype(bool)].copy()
    cases["calibrated_day7"] = curve(dev.day7.to_numpy(), dev.target.to_numpy()).predict(cases.day7.to_numpy())
    return cases


def summary(cases: pd.DataFrame, v2: str, carry: str, seed: int) -> dict:
    losses = {name: date_losses(cases, column) for name, column in
              [("v2", v2), ("carry_forward", carry), ("calibrated_day7", "calibrated_day7")]}
    gain = losses["carry_forward"].mean() - losses["v2"].mean()
    return {
        "batches": int(len(losses["v2"])), "conditions": int(len(cases)),
        "date_macro_mae": {name: float(l.mean()) for name, l in losses.items()},
        "v2_minus_calibrated_interval": paired(losses["v2"], losses["calibrated_day7"], seed),
        "batches_v2_better_than_calibrated": int((losses["v2"] < losses["calibrated_day7"]).sum()),
        "calibrated_minus_carry_forward_interval": paired(losses["calibrated_day7"], losses["carry_forward"], seed),
        "share_of_v2_gain_from_calibration": float((losses["carry_forward"].mean() - losses["calibrated_day7"].mean()) / gain),
        "v2_reduction_vs_calibrated": float(1 - losses["v2"].mean() / losses["calibrated_day7"].mean()),
    }


def kept_external(cases: pd.DataFrame, seed: int = 20260927) -> dict:
    """v2 against the calibrated curve on the conditions v2 kept, and where v2's pooled gain over carry-forward sits."""
    declined = cases.declined.astype(bool)
    kept = cases.loc[~declined]
    v2, curve_ = date_losses(kept, "prediction"), date_losses(kept, "calibrated_day7")
    gain = (cases.day7 - cases.target).abs() - (cases.prediction - cases.target).abs()
    return {
        "conditions": int(len(kept)), "batches": int(len(v2)),
        "date_macro_mae": {"v2": float(v2.mean()), "calibrated_day7": float(curve_.mean())},
        "v2_minus_calibrated_interval": paired(v2, curve_, seed),
        "batches_v2_better_than_calibrated": int((v2 < curve_).sum()),
        "share_of_pooled_gain_from_declined": float(gain[declined].sum() / gain.sum()),
        "share_of_pooled_gain_from_day7_at_most_minus_1": float(gain[cases.day7 <= -1].sum() / gain.sum()),
    }


def batch_difficulty(cases: pd.DataFrame, sigma: str, prediction: str) -> dict:
    """Rank correlation, over batches, of the batch-mean difficulty score with the batch's date-macro MAE."""
    mean_sigma = cases.groupby("date")[sigma].mean()
    error = date_losses(cases, prediction).reindex(mean_sigma.index)
    return {"batches": int(len(mean_sigma)), "spearman": float(mean_sigma.rank().corr(error.rank()))}


def run() -> dict:
    study, external = study_cases(), external_cases()
    return {
        "status": "post hoc, descriptive; both sealed sets were already spent",
        "study_41_batches": summary(study, "pred__v2_incumbent", "pred__persistence_day7", seed=20261007),
        "external_24_batches": summary(external, "prediction", "day7", seed=20260927),
        "external_kept_conditions": kept_external(external),
        "batch_difficulty": {"study_41_batches": batch_difficulty(study, "sigma__incumbent", "pred__v2_incumbent"),
                             "external_24_batches": batch_difficulty(external, "trust_score", "prediction")},
    }


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
