"""Post hoc: is the harness's reference split itself optimistic, because its folds interleave dates?

The reference split holds out culture batches and their chemicals, but each fold mixes batches from 2014 to 2017, so a
model is trained on batches recorded after the one it is tested on. A lab only ever has earlier batches. On the 24
external batches the reference split scores the v2 recipe at 0.666, while the sealed external test (trained on the 17
later batches only) scored 0.735 on the same 927 conditions. This script separates time order from training size.

Rules, fixed before this script was first run:
  * models: carry-forward, the calibrated day-7 curve (isotonic) and the shipped v2 recipe (Extra Trees, 13 inputs),
    refitted here with the split study's own model factory and seed, on usable conditions only;
  * every training set drops all chemicals tested in the test batch (as in the reference split);
  * forward: each batch with at least 10 earlier batches is predicted from all earlier batches (31 batches);
  * period: each of the 24 batches from 2014-16 is predicted from the 17 batches from 2016-17 only (the sealed design);
  * interleaved: the reference split (v3 outer folds), scored on the same batches;
  * size-matched: each test batch predicted from as many batches as the forward (or period) design had, drawn at random
    from all other batches regardless of date; 5 draws, per-batch loss averaged over draws;
  * metric: date-macro MAE; paired batch bootstrap of the difference (seed 20261008).

Descriptive only, and outside the pre-registered split study: the 41 batches were already used by that study and by the
model study, and nothing here changes a locked result or the shipped model.

    python -X utf8 -B tools/posthoc_temporal_split.py             # refit (about 2 minutes) and analyse
    python -X utf8 -B tools/posthoc_temporal_split.py --analyze   # analyse the shipped predictions only
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_harness as h  # noqa: E402
import neuroforecast_split_study as study  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/posthoc_baselines/temporal_split.json"
PREDICTIONS = ROOT / "evaluation/posthoc_baselines/temporal_split_predictions.csv.gz"
MODELS = ("carry_forward", "calibrated_day7", "extra_trees_v2")
MIN_HISTORY = 10
DRAWS = 5
PERIOD_SPLIT = 20160701  # the external batches end in June 2016; the 17 development batches start in July 2016


def fit_rows(cases: pd.DataFrame, usable: np.ndarray, train_dates, test: np.ndarray) -> np.ndarray:
    """Usable rows of the training batches, minus every chemical tested in the test rows."""
    chemicals = set(cases.identity.iloc[test])
    keep = cases.date.isin(set(train_dates)).to_numpy() & usable & ~cases.identity.isin(chemicals).to_numpy()
    return np.flatnonzero(keep)


def predictions() -> pd.DataFrame:
    """Every design's forecasts for every usable condition (NaN where a design does not test that batch)."""
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    bench = h.load_benchmark()
    cases = bench.cases
    x = bench.features[h.SMALL_COLUMNS].to_numpy(dtype=float)
    usable = cases.usable.to_numpy(dtype=bool)
    day7, y = cases.day7.to_numpy(), cases.target.to_numpy()
    dates = sorted(int(d) for d in cases.date.unique())
    columns: dict[str, np.ndarray] = {}

    def fill(design: str, test: np.ndarray, fit: np.ndarray) -> None:
        for m in MODELS:
            columns.setdefault(f"{design}__{m}", np.full(len(cases), np.nan))[test] = study.fit_predict(m, x, day7, y, fit, test)

    for _, train, test in h.splits(cases, "batch_chemical"):
        fill("interleaved", test, train[usable[train]])
    early = [d for d in dates if d < PERIOD_SPLIT]
    late = [d for d in dates if d >= PERIOD_SPLIT]
    for design, tests, train_for, n_for, salt in [
            ("forward", dates[MIN_HISTORY:], lambda date: [b for b in dates if b < date], lambda date: dates.index(date), 1),
            ("period", early, lambda date: late, lambda date: len(late), 2)]:
        for date in tests:
            test = np.flatnonzero(cases.date.to_numpy() == date)
            fill(design, test, fit_rows(cases, usable, train_for(date), test))
            others = [b for b in dates if b != date]
            for d in range(DRAWS):
                drawn = np.random.default_rng([h.SEED, salt, d, int(date)]).choice(others, size=n_for(date), replace=False)
                fill(f"{design}_size_matched_{d}", test, fit_rows(cases, usable, list(drawn), test))
    table = cases.loc[:, ["case_id", "date", "identity", "dose", "target", "day7", "usable"]].assign(**columns)
    return table.loc[usable].reset_index(drop=True)


# --------------------------------------------------------------------------- analysis (fast, from the predictions)

def losses(table: pd.DataFrame, design: str, dates: list[int]) -> dict[str, pd.Series]:
    """Per-batch losses of each model under a design on the given batches; size-matched designs average their draws."""
    frame = table.loc[table.date.isin(dates)]
    if design.endswith("_size_matched"):
        return {m: pd.concat([h.date_losses(frame, f"{design}_{d}__{m}") for d in range(DRAWS)], axis=1).mean(axis=1)
                for m in MODELS}
    return {m: h.date_losses(frame, f"{design}__{m}") for m in MODELS}


def compare(a: dict[str, pd.Series], b: dict[str, pd.Series]) -> dict:
    out = {}
    for m in MODELS:
        p = h.paired(a[m], b[m])
        out[m] = {"a": float(a[m].mean()), "b": float(b[m].mean()), "relative": float(a[m].mean() / b[m].mean() - 1),
                  "difference": p["difference"], "low": p["low"], "high": p["high"],
                  "batches_better": p["batches_better"], "batches": p["batches"]}
    return out


def summary(per: dict[str, pd.Series]) -> dict:
    mae = {m: float(per[m].mean()) for m in MODELS}
    return {"date_macro_mae": mae, "edge_over_calibrated": 1 - mae["extra_trees_v2"] / mae["calibrated_day7"],
            "edge_over_carry_forward": 1 - mae["extra_trees_v2"] / mae["carry_forward"]}


def analyze(table: pd.DataFrame) -> dict:
    result = {"status": "post hoc, descriptive; outside the pre-registered split study; nothing locked is changed",
              "rules": {"min_history_batches": MIN_HISTORY, "size_matched_draws": DRAWS, "period_split": PERIOD_SPLIT,
                        "purge": "every chemical tested in the test batch is dropped from training"}}
    for design in ("forward", "period"):
        dates = sorted(int(d) for d in table.loc[table[f"{design}__extra_trees_v2"].notna()].date.unique())
        inter, own, size = (losses(table, k, dates) for k in ("interleaved", design, f"{design}_size_matched"))
        result[design] = {
            "test_batches": len(dates), "first_test_batch": dates[0], "conditions": int(table.date.isin(dates).sum()),
            "interleaved": summary(inter), design: summary(own), "size_matched": summary(size),
            f"interleaved_vs_{design}": compare(inter, own), f"size_matched_vs_{design}": compare(size, own),
            "interleaved_vs_size_matched": compare(inter, size)}
    result["all_41_interleaved"] = summary(losses(table, "interleaved", sorted(int(d) for d in table.date.unique())))
    return result


if __name__ == "__main__":
    if "--analyze" in sys.argv[1:]:
        table = pd.read_csv(PREDICTIONS, dtype={"identity": str})
    else:
        table = predictions()
        PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(PREDICTIONS, index=False, lineterminator="\n", compression={"method": "gzip", "mtime": 0})
    result = analyze(table)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    et = "extra_trees_v2"
    for design in ("forward", "period"):
        r = result[design]
        print(f"{design}: {r['test_batches']} test batches from {r['first_test_batch']}, {r['conditions']} conditions")
        for name in ["interleaved", design, "size_matched"]:
            s = r[name]
            print(f"  {name:<13} v2 {s['date_macro_mae'][et]:.3f}  calibrated {s['date_macro_mae']['calibrated_day7']:.3f}  "
                  f"carry-forward {s['date_macro_mae']['carry_forward']:.3f}  edge over calibrated {s['edge_over_calibrated']:.0%}")
        for key in [k for k in r if "_vs_" in k]:
            c = r[key][et]
            print(f"  {key:<28} v2 {c['relative']:+.1%}  diff {c['difference']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}]  "
                  f"better on {c['batches_better']}/{c['batches']}")
