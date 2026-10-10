"""NeuroForecast v3 data: pool all 45 EPA NFA dates into one development benchmark.

Both earlier sealed tests (the 3-date reserve and the 28-date external set) are spent, so v3 is a
development study: every model is compared by the same nested, chemical-purged, batch-grouped
cross-validation (experiments/neuroforecast_v3_protocol.json). Nothing here is a new sealed test.

Outputs (E:/kaggle/AI4S/experiments/neuroforecast_v3/prepared):
  cases.csv      one row per condition (date, identity, dose) with the v2 target, usability and fold
  features.csv   the v2 day-5/day-7 feature table plus batch-context descriptors (days 5 and 7 only)
  twin.csv       later-day targets for the multi-endpoint "functional profile" task (days 9 and 12)
  wells.csv.gz   one row per physical well with readouts at days 5, 7, 9 and 12 (for the plate-context model)
  manifest.json  counts, fold assignment and SHA-256 of every file and of the code that wrote them
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_nfa_intake as intake  # noqa: E402
import neuroforecast_data as data  # noqa: E402  (frozen core, imported unchanged)
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402

OUT = Path("E:/kaggle/AI4S/experiments/neuroforecast_v3")
PROTOCOL = data.ROOT / "experiments/neuroforecast_v3_protocol.json"
READOUTS = intake.READOUTS_IN_RELEASE
DAYS = [5, 7, 9, 12]
OUTER_FOLDS = 9
CONTEXT = ["ctx_ctrl_growth_ns", "ctx_ctrl_growth_mfr", "ctx_ctrl_growth_nae", "ctx_slope_d5_d7", "ctx_corr_d5_d7",
           "ctx_changed_frac_d5", "ctx_plate_ctrl_growth_ns", "ctx_ctrl_median_mfr_d7", "ctx_ctrl_median_nae_d7"]


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def outer_fold_of(dates: list[int], k: int = OUTER_FOLDS) -> dict[int, int]:
    """Interleave the sorted dates into k folds, so every fold spans the 2014-2017 range."""
    return {date: i % k for i, date in enumerate(sorted(dates))}


def control_medians(raw: pd.DataFrame, day: int, field: str) -> pd.Series:
    rows = raw.loc[(raw.DIV == day) & (raw.dose == 0)]
    return rows.groupby(["date", "Plate.SN"])[field].median()


def growth(raw: pd.DataFrame, field: str) -> pd.Series:
    """Per-plate log2 growth of the control median from day 5 to day 7."""
    return np.log2((control_medians(raw, 7, field) + 1) / (control_medians(raw, 5, field) + 1))


def batch_context(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """In-context descriptors of each culture batch from days 5 and 7 only.

    They describe how this batch's controls are maturing and how persistent chemical effects already
    are between days 5 and 7, the batch property that decided both earlier sealed tests.
    """
    early = raw.loc[raw.DIV.isin([5, 7])]
    dates = features.index.get_level_values("date")
    d5, d7 = features.normalized_ns_d5, features.normalized_ns_d7
    table = pd.DataFrame(index=sorted(set(dates)))
    for name, field in [("ns", "ns.n"), ("mfr", "meanfiringrate"), ("nae", "nAE")]:
        table[f"ctx_ctrl_growth_{name}"] = growth(early, field).groupby(level=0).median()
    slope, corr = {}, {}
    for date, group in pd.DataFrame({"d5": d5.to_numpy(), "d7": d7.to_numpy()}, index=dates).groupby(level=0):
        x, y = group.d5.to_numpy(), group.d7.to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        slope[date] = float(np.polyfit(x, y, 1)[0]) if len(x) > 2 and np.ptp(x) > 0 else np.nan
        corr[date] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan
    table["ctx_slope_d5_d7"] = pd.Series(slope)
    table["ctx_corr_d5_d7"] = pd.Series(corr)
    table["ctx_changed_frac_d5"] = (d5.abs() >= 0.5).groupby(dates).mean()
    table["ctx_ctrl_median_mfr_d7"] = np.arcsinh(control_medians(early, 7, "meanfiringrate")).groupby(level=0).median()
    table["ctx_ctrl_median_nae_d7"] = np.arcsinh(control_medians(early, 7, "nAE")).groupby(level=0).median()
    out = table.reindex(dates).set_axis(features.index)
    # Plate-level control growth for the plates a condition's wells sit on (mean over those plates).
    treated = early.loc[(early.DIV == 7) & (early.dose > 0) & early.eligible_identity, data.KEY + ["Plate.SN"]].drop_duplicates()
    plate_growth = growth(early, "ns.n").rename("g").reset_index()
    per_case = treated.merge(plate_growth, on=["date", "Plate.SN"], how="left").groupby(data.KEY).g.mean()
    out["ctx_plate_ctrl_growth_ns"] = per_case.reindex(features.index).to_numpy()
    return out[CONTEXT]


def twin_targets(raw: pd.DataFrame, cfg: dict, index: pd.Index) -> pd.DataFrame:
    """Condition-level later-day profile: relative network spikes and the control contrast of every readout."""
    cols = {}
    for day in [9, 12]:
        observed = data._observed_day(raw.loc[raw.DIV == day], day, cfg)
        cols[f"normalized_ns_d{day}"] = observed.normalized_ns
        for field in READOUTS:
            value, control = observed[field], observed["control__" + field]
            denominator = value.abs() + control.abs()
            cols[f"contrast__{field}__d{day}"] = ((value - control) / denominator).where(denominator != 0, 0.0)
    return pd.DataFrame(cols).reindex(index)


def well_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per physical well, readouts at each day side by side."""
    keys = ["date", "Plate.SN", "well"]
    static = raw.sort_values("DIV").groupby(keys).agg(dose=("dose", "first"), identity=("identity", "first"),
                                                       eligible=("eligible_identity", "first"), trt=("trt", "first"),
                                                       spid=("spid", "first"))
    wide = raw.pivot_table(index=keys, columns="DIV", values=READOUTS, aggfunc="first", dropna=False)
    wide.columns = [f"{field}__d{day}" for field, day in wide.columns]
    for field in READOUTS:
        for day in DAYS:
            if f"{field}__d{day}" not in wide:
                wide[f"{field}__d{day}"] = np.nan
    wide = wide[[f"{field}__d{day}" for day in DAYS for field in READOUTS]]
    return static.join(wide).reset_index()


def prepare(out: Path) -> None:
    target = out / "prepared"
    if (target / "manifest.json").exists():
        raise ValueError("Prepared data exist; preserve them")
    target.mkdir(parents=True, exist_ok=True)
    cfg = v2.v2_config()
    raw = v2.release_frame()
    features = v2.features_from(raw.loc[raw.DIV.isin([5, 7])], cfg)
    features = features.join(batch_context(raw, features))
    targets = data.build_targets(raw, cfg)
    common = features.index.intersection(targets.index, sort=False)
    features = features.loc[common]
    quality = v2.outcome_quality(raw, cfg).reindex(common)
    cases = common.to_frame(index=False)
    cases["case_id"] = gate.case_ids(common)
    cases["target"] = targets.loc[common].to_numpy()
    cases["day7"] = features.normalized_ns_d7.to_numpy()
    cases["day12_ctrl_median_ns"] = quality.to_numpy()
    cases["usable"] = cases.day12_ctrl_median_ns >= v2.USABLE_MIN
    cases["spid"] = v2.sample_ids(raw).reindex(common).to_numpy()
    source = raw.loc[raw.dose > 0].groupby("date").source.agg(lambda s: sorted(set(s))[0])
    cases["source"] = cases.date.map(source).to_numpy()
    cases["cohort"] = np.where(cases.date.isin(v2.DEV_DATES), "archive17", "external28")
    folds = outer_fold_of(sorted(cases.date.unique()))
    cases["outer_fold"] = cases.date.map(folds).to_numpy()
    twin = twin_targets(raw, cfg, common)
    wells = well_table(raw)
    wells["outer_fold"] = wells.date.map(folds)

    features.reset_index(drop=True).to_csv(target / "features.csv", index=False)
    cases.to_csv(target / "cases.csv", index=False)
    twin.reset_index(drop=True).to_csv(target / "twin.csv", index=False)
    wells.to_csv(target / "wells.csv.gz", index=False)
    manifest = {
        "cases": len(cases), "usable_cases": int(cases.usable.sum()), "dates": int(cases.date.nunique()),
        "identities": int(cases.identity.nunique()), "plates": int(wells[["date", "Plate.SN"]].drop_duplicates().shape[0]),
        "wells": len(wells), "cases_by_cohort": cases.cohort.value_counts().to_dict(),
        "usable_by_cohort": cases.loc[cases.usable].cohort.value_counts().to_dict(),
        "outer_folds": {str(k): sorted(int(d) for d, f in folds.items() if f == k) for k in range(OUTER_FOLDS)},
        "feature_readouts": cfg["feature_readouts"], "context_columns": CONTEXT,
        "files": {p.name: data.digest(p) for p in sorted(target.glob("*")) if p.name != "manifest.json"},
        "code": {p.name: data.digest(p) for p in [Path(__file__), Path(v2.__file__), Path(intake.__file__),
                                                   Path(data.__file__), Path(gate.__file__)]},
    }
    intake.write_json(target / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("files", "code")}, indent=2))


def interleave(dates, k: int) -> list[list[int]]:
    ordered = sorted(int(d) for d in set(dates))
    return [ordered[i::k] for i in range(k)]


def purged_split(cases: pd.DataFrame, pool: np.ndarray, test_dates) -> tuple[np.ndarray, np.ndarray]:
    """Train/test positions within `pool`: test = conditions on `test_dates`; train = other dates minus
    every condition of any chemical tested on `test_dates` (the chemical purge used since v1)."""
    in_pool = cases.iloc[pool]
    is_test = in_pool.date.isin(list(test_dates)).to_numpy()
    chemicals = set(in_pool.identity[is_test])
    train = pool[~is_test & ~in_pool.identity.isin(chemicals).to_numpy()]
    return train, pool[is_test]


def outer_splits(cases: pd.DataFrame):
    pool = np.arange(len(cases))
    for k in range(OUTER_FOLDS):
        dates = sorted(cases.loc[cases.outer_fold == k, "date"].unique())
        train, test = purged_split(cases, pool, dates)
        yield k, dates, train, test


def inner_splits(cases: pd.DataFrame, train: np.ndarray, k: int = 4):
    for j, dates in enumerate(interleave(cases.iloc[train].date, k)):
        inner_train, inner_test = purged_split(cases, train, dates)
        yield j, dates, inner_train, inner_test


def date_losses(cases: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray) -> pd.Series:
    """Per-date loss: mean absolute error averaged within chemical, then over chemicals (as since v1)."""
    frame = cases.loc[mask, ["date", "identity", "target"]].assign(prediction=np.asarray(prediction)[mask])
    return gate.date_losses(frame)


def macro(cases: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray) -> float:
    return float(date_losses(cases, prediction, mask).mean())


def load(out: Path = OUT) -> dict:
    target = out / "prepared"
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if data.digest(target / name) != digest:
            raise ValueError(f"Prepared file changed: {name}")
    return {
        "manifest": manifest,
        "features": pd.read_csv(target / "features.csv"),
        "cases": pd.read_csv(target / "cases.csv", dtype={"identity": str, "spid": str}),
        "twin": pd.read_csv(target / "twin.csv"),
    }


def load_wells(out: Path = OUT) -> pd.DataFrame:
    return pd.read_csv(out / "prepared/wells.csv.gz", dtype={"identity": str, "spid": str, "Plate.SN": str, "well": str})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    prepare(args.out)


if __name__ == "__main__":
    main()
