"""Post hoc: does recalibrating the intervals from a lab's own day-12 checks restore coverage? (24 external batches)

The tool's 80% interval is forecast +/- scale x difficulty, with the scale (1.506) fitted on the 17 development
batches. On the external test the intervals held 70% of outcomes, not 80%. A lab running the tool collects one
checked admission record per batch. `neuroforecast_triage_v2.py --history` refits the scale from those records: the
split-conformal quantile of |measured - forecast| / difficulty over every scored condition of the lab's earlier
batches. This script replays that rule in date order over the 24 external batches, using only saved predictions.

Rule, fixed before this script was first run:
  * history = every scored condition (usable day-12 target) of the batches recorded before this one;
  * recalibrate once the history holds at least 2 batches, otherwise keep the shipped scale;
  * primary: expanding history (all earlier batches); also shown: the 6 most recent batches only.
Metrics are over the batches that could be recalibrated (the 3rd to the 24th), for the shipped and the lab scale.
Added after the first run, at a reviewer's request: two oracle constants chosen with hindsight on those batches (one
widening of the shipped intervals to the lab rule's mean width, and the widening that reaches nominal pooled coverage).
A lab cannot know either in advance; they show what a single correct inflation factor would have achieved.

Descriptive only: the external set was already scored once, and nothing here changes a locked result.

    python -X utf8 -B tools/posthoc_lab_recalibration.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_harness as h  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/posthoc_baselines/lab_recalibration.json"
MIN_BATCHES = 2
WINDOW = 6
LEVELS = (0.8, 0.9)


def external() -> pd.DataFrame:
    cases = pd.read_csv(ROOT / "evaluation/external/arm_b_cases.csv", dtype={"identity": str})
    return cases.loc[cases.primary.astype(bool) & cases.usable.astype(bool)].sort_values(["date", "identity", "dose"]).reset_index(drop=True)


def shipped_scale(cases: pd.DataFrame, level: float) -> float:
    tag = f"{level:.1f}"
    return float(((cases[f"high_{tag}"] - cases[f"low_{tag}"]) / 2 / cases.trust_score).median())


def replay(cases: pd.DataFrame, level: float, window: int | None = None) -> pd.DataFrame:
    """Per condition: the scale in force when its batch was analysed, under the shipped and the lab rule."""
    dates = sorted(cases.date.unique())
    score = (cases.target - cases.prediction).abs() / cases.trust_score
    rows = []
    for i, date in enumerate(dates):
        earlier = dates[:i] if window is None else dates[max(0, i - window):i]
        lab = h.conformal_quantile(score[cases.date.isin(earlier)].to_numpy(), level) if len(earlier) >= MIN_BATCHES else np.nan
        rows.append({"date": date, "order": i + 1, "history_batches": len(earlier), "lab_scale": lab})
    return cases.merge(pd.DataFrame(rows), on="date")


def coverage(frame: pd.DataFrame, scale: pd.Series | float, level: float) -> dict:
    half = scale * frame.trust_score
    return h.coverage(frame.assign(low=frame.prediction - half, high=frame.prediction + half), level)


def run() -> dict:
    cases = external()
    result = {"status": "post hoc, descriptive; the external set was already scored once",
              "rule": {"min_history_batches": MIN_BATCHES, "history": "expanding (primary); last 6 batches (secondary)",
                       "scale": "split-conformal quantile of |measured - forecast| / difficulty over the history's scored conditions"},
              "levels": {}}
    for level in LEVELS:
        shipped = shipped_scale(cases, level)
        entry = {"shipped_scale": shipped}
        for name, window in [("expanding", None), ("last_6_batches", WINDOW)]:
            frame = replay(cases, level, window)
            evaluated = frame.loc[frame.lab_scale.notna()]
            per_batch = evaluated.groupby("date").lab_scale.first()
            lab = coverage(evaluated, evaluated.lab_scale, level)
            # Oracle comparisons, chosen with hindsight on the evaluated batches (a lab cannot know these factors in advance):
            # one constant widening of the shipped intervals with the same mean width as the lab rule.
            same_width = lab["mean_width"] / coverage(evaluated, shipped, level)["mean_width"]
            entry[name] = {"batches_evaluated": int(evaluated.date.nunique()), "conditions_evaluated": int(len(evaluated)),
                           "shipped": coverage(evaluated, shipped, level), "lab": lab,
                           "oracle_constant_same_width": {"factor": float(same_width),
                                                          **coverage(evaluated, shipped * same_width, level)},
                           "lab_scale_first": float(per_batch.iloc[0]), "lab_scale_last": float(per_batch.iloc[-1]),
                           "lab_scale_range": [float(per_batch.min()), float(per_batch.max())]}
        evaluated = replay(cases, level).pipe(lambda f: f.loc[f.lab_scale.notna()])
        nominal = h.conformal_quantile(((evaluated.target - evaluated.prediction).abs() / evaluated.trust_score).to_numpy(), level)
        entry["oracle_constant_nominal"] = {"factor": float(nominal / shipped), **coverage(evaluated, nominal, level)}
        result["levels"][f"{level:.1f}"] = entry
    return result


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    for level, e in result["levels"].items():
        for name in ["expanding", "last_6_batches"]:
            s, l = e[name]["shipped"], e[name]["lab"]
            print(f"{level} {name:<15} {e[name]['batches_evaluated']} batches | shipped scale {e['shipped_scale']:.3f}: pooled {s['pooled']:.3f}, "
                  f"batch mean {s['batch_mean']:.3f}, worst fifth {s['worst_fifth']:.3f}, width {s['mean_width']:.2f} | lab scale "
                  f"{e[name]['lab_scale_first']:.2f}->{e[name]['lab_scale_last']:.2f}: pooled {l['pooled']:.3f}, batch mean {l['batch_mean']:.3f}, "
                  f"worst fifth {l['worst_fifth']:.3f}, within 0.1 {l['share_of_batches_within_0.1']:.0%}, width {l['mean_width']:.2f}")
