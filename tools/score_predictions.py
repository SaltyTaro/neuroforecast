"""Score any day-12 forecast on the NeuroForecast validation harness (41 public EPA batches, 2,243 conditions).

A predictions CSV needs `case_id` and `prediction` (log2 day-12 network spikes relative to same-plate controls) for
every usable condition. Optional columns: `low` and `high` (an interval at --level), and `decline` (1 where your
method refuses to forecast). Case IDs and fold assignments are in evaluation/harness/folds.csv; the inputs (days 5 and
7 only) are in evaluation/v3/prepared/features.csv, row-aligned with cases.csv.

Train under the design you score. The reference design is batch_chemical, the one a lab meets in use: a new culture
batch testing new chemicals. Other designs are scored against baselines fitted under the same design, but their
errors are optimistic (see docs/validation_harness.md).

    python -X utf8 -B tools/score_predictions.py --predictions my_forecasts.csv
    python -X utf8 -B tools/score_predictions.py --example          # the shipped v2 forecasts, with their intervals
    python -X utf8 -B tools/score_predictions.py --export           # rewrite evaluation/harness/ from the prepared tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_harness as h  # noqa: E402

OUT = h.ROOT / "evaluation/harness"
EXAMPLE = h.ROOT / "examples/harness/v2_reference_predictions.csv"


def export() -> None:
    """Folds for every design (the published reference split) and reference forecasts on the reference split."""
    bench = h.load_benchmark()
    cases = bench.cases
    folds = cases[["case_id", "date", "identity", "dose", "usable"]].copy()
    for design in h.DESIGNS:
        folds[f"fold__{design}"] = h.fold_assignment(cases, design)
    OUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT / "folds.csv", index=False, lineterminator="\n")
    h.reference_predictions(bench).to_csv(OUT / "reference_predictions.csv", index=False, lineterminator="\n")
    table = pd.read_csv(h.REFERENCE, dtype={"identity": str}).set_index("case_id").reindex(cases.case_id)
    example = pd.DataFrame({"case_id": cases.case_id.to_numpy(), "prediction": table.pred__v2_incumbent.to_numpy(),
                            "low": table["lo__incumbent__date_scaled_0.8"].to_numpy(),
                            "high": table["hi__incumbent__date_scaled_0.8"].to_numpy()})
    EXAMPLE.parent.mkdir(parents=True, exist_ok=True)
    example.loc[cases.usable.to_numpy()].to_csv(EXAMPLE, index=False, lineterminator="\n")
    print(f"wrote {OUT / 'folds.csv'}, {OUT / 'reference_predictions.csv'} and {EXAMPLE}")


def render(result: dict) -> str:
    reference = result["design"] == "batch_chemical"
    lines = [f"NeuroForecast harness: {result['conditions']} conditions, {result['batches']} batches, {result['chemicals']} chemicals; "
             f"design {result['design']}{' (reference split)' if reference else ''}",
             f"  your forecast: date-macro MAE {result['date_macro_mae']:.3f} (pooled {result['pooled_mae']:.3f}; worst batch {result['worst_batch_mae']:.2f})",
             "  against, on the same conditions (MAE; your difference with its paired 95% interval over batches; batches you win):"]
    for name, row in result["against"].items():
        lines.append(f"    {name:<20} {row['date_macro_mae']:.3f}   {row['difference']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]   "
                     f"{row['batches_better']}/{row['batches']}")
    if "intervals" in result:
        c = result["intervals"]
        lines.append(f"  intervals at nominal {c['nominal']:.2f}: pooled {c['pooled']:.3f}, batch mean {c['batch_mean']:.3f}, "
                     f"worst fifth of batches {c['worst_fifth']:.3f}, batches covering at least {c['nominal'] - 0.1:.0%}: {c['share_of_batches_within_0.1']:.0%}, "
                     f"mean width {c['mean_width']:.2f}")
    if "refusal" in result:
        r = result["refusal"]
        lines.append(f"  refusal: {r['declined']} declined, {r['share_kept']:.0%} kept; kept MAE {r['kept_pooled_mae']:.3f}, "
                     f"declining cut pooled error by {r['error_reduction_from_declining']:.0%}")
    if not reference:
        lines.append("  note: this design shares culture batches or chemicals between training and test, so its errors are "
                     "optimistic; report the batch_chemical score as your estimate for new batches.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, help="CSV with case_id, prediction, optional low, high, decline")
    parser.add_argument("--design", choices=h.DESIGNS, default="batch_chemical", help="The split your predictions were made under")
    parser.add_argument("--level", type=float, default=0.8, help="Nominal level of low/high (default 0.8)")
    parser.add_argument("--json", type=Path, help="Write the full result, including per-batch errors")
    parser.add_argument("--example", action="store_true", help=f"Score {EXAMPLE.relative_to(h.ROOT).as_posix()}")
    parser.add_argument("--export", action="store_true", help="Rewrite the published folds and reference predictions")
    args = parser.parse_args(argv)
    if args.export:
        export()
        return
    path = EXAMPLE if args.example else args.predictions
    if path is None:
        parser.error("give --predictions, --example or --export")
    try:
        predictions = pd.read_csv(path)
        if not {"case_id", "prediction"} <= set(predictions.columns):
            raise ValueError("the predictions file needs case_id and prediction columns")
        result = h.score(h.load_benchmark(), predictions, args.design, args.level)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(render(result))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
