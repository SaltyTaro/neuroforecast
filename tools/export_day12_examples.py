"""Write held-out example batches for the day-12 check: day-5/day-7 inputs and the day-12 recordings.

The external test is spent (scored once on September 27, 2026), so showing its day-12 recordings changes no result.
Batches are chosen by fixed rules over the published per-batch errors, not by eye:

  * best:   the batch where v2's error was lowest (20160120, the existing example);
  * median: the batch whose v2 error is closest to the median over the 24 batches;
  * lost:   the batch where v2 lost to carrying day 7 forward by the largest relative margin.

Each day-12 file must reproduce the sealed test's per-condition targets (to 1e-12) and usable flags, or the
export stops.
The release itself (All_DIV_Data.Rdata) is read from the author's data drive; the written CSVs ship in the repository.

    python -X utf8 -B tools/export_day12_examples.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evaluation/external/arm_b_cases.csv"
TARGETS = [ROOT / "examples/v2", ROOT / "app/examples/v2"]
COLUMNS = ["date", "Plate.SN", "well", "DIV", "trt", "dose", "units", "identity", "eligible_identity", "spid"]


def per_batch(cases: pd.DataFrame) -> pd.DataFrame:
    """Published per-batch errors on the primary cohort (date-macro: chemicals averaged within a batch)."""
    primary = cases.loc[cases.primary.astype(bool) & cases.usable.astype(bool)]
    losses = {}
    for name, column in [("v2", "prediction"), ("persistence", "day7")]:
        error = (primary[column] - primary.target).abs()
        losses[name] = error.groupby([primary.date, primary.identity]).mean().groupby(level=0).mean()
    table = pd.DataFrame(losses)
    table["relative_gain"] = 1 - table.v2 / table.persistence
    table["rank_v2"] = table.v2.rank().astype(int)
    return table


def choose(table: pd.DataFrame) -> dict[str, int]:
    median = table.v2.median()
    return {
        "best": int(table.v2.idxmin()),
        "median": int((table.v2 - median).abs().idxmin()),
        "lost": int(table.relative_gain.idxmin()),
    }


def frame_for(raw: pd.DataFrame, date: int, days: list[int], readouts: list[str]) -> pd.DataFrame:
    rows = raw.loc[(raw.date == date) & raw.DIV.isin(days), COLUMNS + readouts]
    return rows.sort_values(["Plate.SN", "DIV", "well"], key=lambda s: s.map(str) if s.name != "DIV" else s, kind="stable")


def day12_truth(rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The sealed test's own definitions: data.build_targets and gate.outcome_quality on day-12 rows only."""
    target = data.build_targets(rows, cfg)
    quality = gate.outcome_quality(rows, cfg).reindex(target.index)
    out = target.index.to_frame(index=False)
    out["target"] = target.to_numpy()
    out["day12_ctrl_median_ns"] = quality.to_numpy()
    out["usable"] = out.day12_ctrl_median_ns >= v2.USABLE_MIN
    return out


def main() -> None:
    cases = pd.read_csv(CASES, dtype={"identity": str})
    table = per_batch(cases)
    chosen = choose(table)
    cfg = v2.v2_config()
    readouts = cfg["feature_readouts"]
    raw = v2.release_frame()
    raw = raw.loc[raw.date.isin(chosen.values())].copy()

    # Git stores the examples with LF; a Windows checkout may hold CRLF, so compare line content.
    existing = (TARGETS[0] / "external_20160120_day5_day7.csv").read_bytes().replace(b"\r\n", b"\n")
    regenerated = frame_for(raw, 20160120, [5, 7], readouts).to_csv(index=False, lineterminator="\n").encode("utf-8")
    if regenerated != existing:
        raise ValueError("The recipe does not regenerate the shipped 20160120 example byte for byte")

    written = {}
    for role, date in chosen.items():
        early = frame_for(raw, date, [5, 7], readouts)
        late = frame_for(raw, date, [12], readouts)
        truth = day12_truth(late, cfg)
        expected = cases.loc[cases.date == date].set_index(data.KEY)
        joined = truth.set_index(data.KEY).join(expected[["target", "day12_ctrl_median_ns", "usable"]], rsuffix="_sealed", how="outer")
        if joined.target_sealed.isna().any() or joined.target.isna().any():
            raise ValueError(f"{date}: day-12 conditions differ from the sealed test's")
        gap = float((joined.target - joined.target_sealed).abs().max())
        if gap > 1e-12:   # floating-point summation order only
            raise ValueError(f"{date}: day-12 targets differ from the sealed test's by {gap}")
        if not (joined.usable == joined.usable_sealed.astype(bool)).all():
            raise ValueError(f"{date}: usable flags differ from the sealed test's")
        for target in TARGETS:
            target.mkdir(parents=True, exist_ok=True)
            if role != "best":
                (target / f"external_{date}_day5_day7.csv").write_bytes(early.to_csv(index=False, lineterminator="\n").encode("utf-8"))
            (target / f"external_{date}_day12.csv").write_bytes(late.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        row = table.loc[date]
        written[role] = {"date": date, "conditions": int(len(truth)), "v2": round(float(row.v2), 6),
                         "persistence": round(float(row.persistence), 6), "rank_of_24": int(row.rank_v2),
                         "max_target_difference": gap}
    print(json.dumps({"median_v2_over_batches": round(float(table.v2.median()), 6), "chosen": written}, indent=2))


if __name__ == "__main__":
    main()
