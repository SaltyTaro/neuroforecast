"""NeuroForecast validation harness: a public benchmark for day-7 -> day-12 forecasts of neuronal MEA assays,
with fixed splits under four designs and one scoring function.

The benchmark is every eligible condition of the 41 public EPA network-formation-assay batches (2,285 conditions,
2,243 with a usable day-12 target, 243 chemicals). Features (days 5 and 7 only) and targets are the v3 study's
prepared tables in evaluation/v3/prepared. Both sealed tests are spent, so this is a development benchmark, not a
sealed test.

Split designs, from weakest to the one that matches deployment (a new culture batch testing new chemicals):
  random_condition  conditions assigned to folds at random; batches and chemicals are shared across folds
  chemical          chemicals held out; the same culture batches appear in training
  batch             culture batches held out; the same chemicals may appear in training
  batch_chemical    batches held out and every test chemical purged from training (the design of both sealed
                    tests and the v3 study)

The reference split for scoring is batch_chemical with the v3 study's outer folds (9 folds, dates interleaved), so
v2, carry-forward and the calibrated day-7 curve have reference predictions on exactly the same cases.

    import neuroforecast_harness as h
    bench = h.load_benchmark()
    for k, train, test in h.splits(bench.cases, "batch_chemical"):
        ...fit on bench.features.iloc[train] where bench.cases.usable, predict bench.features.iloc[test]...
    result = h.score(bench, predictions)          # predictions: DataFrame with case_id, prediction[, low, high, decline]
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "evaluation/v3/prepared"
REFERENCE = ROOT / "evaluation/v3/analysis/outer_case_table.csv"
DESIGNS = ("random_condition", "chemical", "batch", "batch_chemical")
FOLDS = 9
INNER_FOLDS = 4
SEED = 20261008
RESAMPLES = 10000
# The shipped v2 recipe's 13 inputs: activity and coordination readouts at days 5 and 7, relative to same-plate
# controls, plus log concentration.
SMALL_COLUMNS = ["normalized_ns_d5", "normalized_ns_d7", "log10_dose",
                 "contrast__meanfiringrate__d5", "contrast__r__d5", "contrast__nAE__d5", "contrast__nABE__d5", "contrast__ns.n__d5",
                 "contrast__meanfiringrate__d7", "contrast__r__d7", "contrast__nAE__d7", "contrast__nABE__d7", "contrast__ns.n__d7"]


@dataclass
class Benchmark:
    cases: pd.DataFrame      # one row per condition: case_id, date, identity, dose, target, day7, usable, outer_fold, ...
    features: pd.DataFrame   # same rows: day-5 and day-7 inputs only


def load_benchmark(prepared: Path = PREPARED) -> Benchmark:
    cases = pd.read_csv(prepared / "cases.csv", dtype={"identity": str})
    features = pd.read_csv(prepared / "features.csv")
    if len(cases) != len(features) or not np.allclose(cases.day7, features.normalized_ns_d7, equal_nan=True):
        raise ValueError("cases.csv and features.csv are not row-aligned")
    cases["usable"] = cases.usable.astype(bool)
    return Benchmark(cases, features)


# --------------------------------------------------------------------------- splits

def fold_assignment(cases: pd.DataFrame, design: str, repetition: int | None = None, k: int = FOLDS,
                    seed: int = SEED, salt: tuple = ()) -> np.ndarray:
    """Fold of every row. repetition=None gives the published reference split: for batch_chemical the v3 outer
    folds, for the other designs repetition 0."""
    if design not in DESIGNS:
        raise ValueError(f"unknown design {design!r}; choose from {DESIGNS}")
    if design == "batch_chemical" and repetition is None and not salt:
        return cases.outer_fold.to_numpy(dtype=int)
    rng = np.random.default_rng([seed, 0 if repetition is None else repetition, DESIGNS.index(design), *salt])
    if design == "random_condition":
        fold = np.empty(len(cases), dtype=int)
        fold[rng.permutation(len(cases))] = np.arange(len(cases)) % k
        return fold
    key = cases.identity if design == "chemical" else cases.date
    groups = rng.permutation(np.array(sorted(key.unique())))
    return key.map({g: i % k for i, g in enumerate(groups)}).to_numpy(dtype=int)


def split(cases: pd.DataFrame, design: str, fold: np.ndarray, k: int, pool: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Train and test positions for fold k within pool. batch_chemical also purges every test chemical from training."""
    pool = np.arange(len(cases)) if pool is None else pool
    test = pool[fold[pool] == k]
    train = pool[fold[pool] != k]
    if design == "batch_chemical":
        train = train[~cases.identity.iloc[train].isin(set(cases.identity.iloc[test])).to_numpy()]
    return train, test


def splits(cases: pd.DataFrame, design: str, repetition: int | None = None):
    """Yield (fold, train positions, test positions) for every outer fold. Fit only on usable training rows."""
    fold = fold_assignment(cases, design, repetition)
    for k in range(int(fold.max()) + 1):
        train, test = split(cases, design, fold, k)
        yield k, train, test


def inner_splits(cases: pd.DataFrame, design: str, pool: np.ndarray, repetition: int, outer: int):
    """The same design applied inside an outer training set, for calibration residuals."""
    fold = np.full(len(cases), -1)
    fold[pool] = fold_assignment(cases.iloc[pool], design, repetition, k=INNER_FOLDS, salt=(outer + 1,))
    for j in range(INNER_FOLDS):
        yield j, *split(cases, design, fold, j, pool)


# --------------------------------------------------------------------------- baselines

def calibrated_day7(cases: pd.DataFrame, design: str, repetition: int | None = None) -> np.ndarray:
    """Isotonic fit of day 12 on day 7 alone, fitted fold by fold under the given design."""
    out = np.full(len(cases), np.nan)
    usable = cases.usable.to_numpy(dtype=bool)
    for _, train, test in splits(cases, design, repetition):
        fit = train[usable[train]]
        curve = IsotonicRegression(out_of_bounds="clip").fit(cases.day7.to_numpy()[fit], cases.target.to_numpy()[fit])
        out[test] = curve.predict(cases.day7.to_numpy()[test])
    return out


def reference_predictions(bench: Benchmark) -> pd.DataFrame:
    """Reference forecasts on the reference split (batch_chemical, v3 outer folds)."""
    table = pd.read_csv(REFERENCE, dtype={"identity": str}).set_index("case_id").reindex(bench.cases.case_id)
    return pd.DataFrame({
        "case_id": bench.cases.case_id.to_numpy(),
        "carry_forward": bench.cases.day7.to_numpy(),
        "calibrated_day7": calibrated_day7(bench.cases, "batch_chemical"),
        "v2": table.pred__v2_incumbent.to_numpy(),
        "v2_plus_transformer": table.pred__ens_v2_pct.to_numpy(),
    })


# --------------------------------------------------------------------------- metrics

def date_losses(frame: pd.DataFrame, column: str) -> pd.Series:
    """Per-batch loss: absolute error averaged over doses within a chemical, then over chemicals within the batch."""
    error = (frame[column] - frame.target).abs()
    return error.groupby([frame.date, frame.identity]).mean().groupby(level=0).mean()


def paired(a: pd.Series, b: pd.Series, seed: int = SEED, n: int = RESAMPLES) -> dict:
    """Batch-paired bootstrap of mean(a) - mean(b): 95% interval and how many batches a is better on."""
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy()
    boot = d[np.random.default_rng(seed).integers(len(d), size=(n, len(d)))].mean(axis=1)
    return {"difference": float(d.mean()), "low": float(np.quantile(boot, 0.025)), "high": float(np.quantile(boot, 0.975)),
            "batches_better": int((d < 0).sum()), "batches": int(len(d))}


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Split-conformal quantile with the finite-sample correction: the ceil((n+1) level)-th smallest score."""
    s = np.sort(np.asarray(scores, dtype=float))
    rank = min(len(s), math.ceil((len(s) + 1) * level))
    return float(s[rank - 1])


def worst_fifth(per_batch: pd.Series) -> float:
    return float(per_batch.sort_values().head(max(1, len(per_batch) // 5)).mean())


def coverage(frame: pd.DataFrame, level: float) -> dict:
    """Coverage of [low, high]: pooled, mean over batches, the worst fifth of batches, share of batches within 0.1."""
    hit = (frame.target >= frame.low) & (frame.target <= frame.high)
    per_batch = hit.groupby(frame.date).mean()
    return {"nominal": level, "pooled": float(hit.mean()), "batch_mean": float(per_batch.mean()),
            "worst_fifth": worst_fifth(per_batch), "share_of_batches_within_0.1": float((per_batch >= level - 0.1).mean()),
            "mean_width": float((frame.high - frame.low).mean())}


def score(bench: Benchmark, predictions: pd.DataFrame, design: str = "batch_chemical", level: float = 0.8,
          baselines: pd.DataFrame | None = None) -> dict:
    """Score predictions for every usable condition. Columns: case_id, prediction, optional low/high, optional decline.

    On the reference split the comparison includes v2 and the v2 + transformer average. For another design the
    baselines are carry-forward and the calibrated curve fitted under that design's own folds, so the comparison is
    fair, but the scores are not comparable with the reference split's (see the split-optimism study)."""
    cases = bench.cases
    if predictions.case_id.duplicated().any():
        raise ValueError("predictions repeat a case_id")
    merged = cases.loc[cases.usable, ["case_id", "date", "identity", "dose", "target", "day7"]].merge(
        predictions, on="case_id", how="left", validate="one_to_one")
    missing = int(merged.prediction.isna().sum())
    if missing:
        raise ValueError(f"{missing} of {len(merged)} usable conditions have no finite prediction")
    if baselines is None:
        if design == "batch_chemical":
            baselines = reference_predictions(bench)
        else:
            baselines = pd.DataFrame({"case_id": cases.case_id, "carry_forward": cases.day7,
                                      "calibrated_day7": calibrated_day7(cases, design, None)})
    merged = merged.merge(baselines, on="case_id", how="left")
    losses = date_losses(merged, "prediction")
    result = {"design": design, "conditions": int(len(merged)), "batches": int(len(losses)), "chemicals": int(merged.identity.nunique()),
              "date_macro_mae": float(losses.mean()), "pooled_mae": float((merged.prediction - merged.target).abs().mean()),
              "worst_batch_mae": float(losses.max()), "against": {}}
    for name in [c for c in baselines.columns if c != "case_id"]:
        base = date_losses(merged, name)
        result["against"][name] = {"date_macro_mae": float(base.mean()), **paired(losses, base)}
    if {"low", "high"} <= set(merged.columns):
        if merged[["low", "high"]].isna().any().any():
            raise ValueError("low/high must be finite for every usable condition")
        result["intervals"] = coverage(merged, level)
    if "decline" in merged.columns:
        declined = merged.decline.fillna(False).astype(bool)
        error = (merged.prediction - merged.target).abs()
        result["refusal"] = {"declined": int(declined.sum()), "share_kept": float(1 - declined.mean()),
                             "kept_pooled_mae": float(error[~declined].mean()),
                             "declined_pooled_mae": float(error[declined].mean()) if declined.any() else None,
                             "kept_date_macro_mae": float(date_losses(merged.loc[~declined], "prediction").mean()),
                             "error_reduction_from_declining": float(1 - error[~declined].mean() / error.mean())}
    result["per_batch"] = {int(d): float(v) for d, v in losses.items()}
    return result
