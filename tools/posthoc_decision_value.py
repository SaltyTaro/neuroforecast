"""Post hoc: where does the learned forecast beat simple rules, and does it change a 2-fold-change call?

Two questions reviewers asked, answered from saved predictions only (no model is fitted here):

1. Where does v2 beat the calibrated day-7 curve? On the harness's reference split (41 batches, batches and chemicals
   held out; development data already used by the model and split studies), errors of carry-forward, the calibrated
   curve and v2 by day-7 band, and on the conditions a three-line rule would keep: decline when day 7 is at or below
   -1.3 log2 (the most extreme day-7 value the tool kept on the external test), else use the calibrated curve.
2. Does the tool change a binary call? On the external test (already scored once; descriptive), the call "this
   condition will not change 2-fold by day 12" (|day 12| < 1 log2) made by the tool (kept and |forecast| < 1) and by
   reading day 7 (|day 7| < 1), with what the tool did with the large changes.

    python -X utf8 -B tools/posthoc_decision_value.py
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
OUT = ROOT / "evaluation/posthoc_baselines/decision_value.json"
REFUSE_AT = -1.3
LARGE = 1.0
MODELS = ("carry_forward", "calibrated_day7", "v2")
BANDS = {
    "collapsed (day 7 <= -1.3)": lambda d7: d7 <= REFUSE_AT,
    "lowered (-1.3 < day 7 < -0.5)": lambda d7: (d7 > REFUSE_AT) & (d7 < -0.5),
    "near control (|day 7| <= 0.5)": lambda d7: d7.abs() <= 0.5,
    "raised (day 7 > 0.5)": lambda d7: d7 > 0.5,
}


def compare(frame: pd.DataFrame) -> dict:
    losses = {m: h.date_losses(frame, m) for m in MODELS}
    p = h.paired(losses["v2"], losses["calibrated_day7"])
    return {"conditions": int(len(frame)), "batches": int(frame.date.nunique()),
            "date_macro_mae": {m: float(s.mean()) for m, s in losses.items()},
            "v2_minus_calibrated": {k: p[k] for k in ("difference", "low", "high", "batches_better", "batches")}}


def where_ml_helps() -> dict:
    bench = h.load_benchmark()
    ref = h.reference_predictions(bench)
    frame = bench.cases.assign(**{m: ref[m].to_numpy() for m in MODELS})
    frame = frame.loc[frame.usable]
    out = {"split": "reference split (batch_chemical, v3 outer folds), 41 batches", "all": compare(frame), "bands": {}}
    for name, rule in BANDS.items():
        out["bands"][name] = compare(frame.loc[rule(frame.day7)])
    out["three_line_rule_keeps"] = compare(frame.loc[frame.day7 > REFUSE_AT])
    return out


def binary_call() -> dict:
    cases = pd.read_csv(ROOT / "evaluation/external/arm_b_cases.csv", dtype={"identity": str})
    cases = cases.loc[cases.primary.astype(bool) & cases.usable.astype(bool)]
    large = cases.target.abs() >= LARGE
    kept = ~cases.declined.astype(bool)
    confident = kept & ((cases["low_0.8"] > 0) | (cases["high_0.8"] < 0))
    tool_says_small = kept & (cases.prediction.abs() < LARGE)
    day7_says_small = cases.day7.abs() < LARGE

    def call(says_small: pd.Series) -> dict:
        return {"called_small": int(says_small.sum()), "npv": float((~large[says_small]).mean()),
                "large_changes_missed": int((large & says_small).sum())}

    return {
        "cohort": "external test, primary usable conditions (already scored once; descriptive)",
        "conditions": int(len(cases)), "large_changes": int(large.sum()),
        "large_change_definition": "|day-12 log2 change| >= 1 (2-fold)",
        "large_already_2_fold_at_day7": float((large & (cases.day7.abs() >= LARGE)).sum() / large.sum()),
        "kept": int(kept.sum()), "kept_forecasts_within_0.5": float((cases.prediction[kept].abs() <= 0.5).mean()),
        "kept_intervals_excluding_zero": int(confident.sum()),
        "confident_calls_large_and_right_sign": int((confident & large & (np.sign(cases.prediction) == np.sign(cases.target))).sum()),
        "large_changes": {"anticipated_confidently": int((large & confident).sum()), "declined": int((large & ~kept).sum()),
                          "kept_not_confident": int((large & kept & ~confident).sum())},
        "declined_with_large_change": float(large[~kept].mean()),
        "declined_max_day7": float(cases.day7[~kept].max()),
        "tool_no_large_change_call": call(tool_says_small),
        "day7_no_large_change_call": call(day7_says_small),
    }


def run() -> dict:
    return {"status": "post hoc, descriptive; nothing is fitted and nothing locked changes",
            "refuse_at_day7": REFUSE_AT, "where_ml_helps": where_ml_helps(), "binary_call": binary_call()}


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    w = result["where_ml_helps"]
    for name, r in [("all", w["all"]), *w["bands"].items(), ("three-line rule keeps", w["three_line_rule_keeps"])]:
        m, d = r["date_macro_mae"], r["v2_minus_calibrated"]
        print(f"{name:<32} {r['conditions']:>5} cond {r['batches']:>2} batches | carry-forward {m['carry_forward']:.3f} "
              f"calibrated {m['calibrated_day7']:.3f} v2 {m['v2']:.3f} | v2-calibrated {d['difference']:+.3f} "
              f"[{d['low']:+.3f}, {d['high']:+.3f}] better on {d['batches_better']}/{d['batches']}")
    b = result["binary_call"]
    print(json.dumps({k: v for k, v in b.items() if k not in ("cohort",)}, indent=1))
