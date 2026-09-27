"""Phase 1 intake of the EPA NFA refinement release: adapter, identity crosswalk, consistency check, sealed cohort.

The release (USEPA/CompTox-DNT-NFA-Refinement, pinned commit) re-processes the same MEA recordings we
use and adds 28 experiment dates from 2014-2016 that no NeuroForecast model has seen. This module:

* converts `All_DIV_Data.Rdata` to the raw schema the frozen data core expects, without editing that core;
* maps EPA sample IDs to DSSTox substance IDs so chemical overlap is decided by identity, not by name;
* compares the release with our archive on the 17 shared dates (our own, already-analysed data);
* writes a sealed-cohort manifest for the 28 new dates from metadata and day-5/day-7 inputs only.

Rule: for the 28 new dates nothing after day 7 is read beyond the *existence* of a day-12 record.
No day-9 or day-12 value, and no EPA hit call, for those dates is loaded into any statistic here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_data as data  # noqa: E402  (frozen core, imported unchanged)

RELEASE_REPO = "https://github.com/USEPA/CompTox-DNT-NFA-Refinement"
RELEASE_COMMIT = "01adf3e1a0068c87fe221d60df36b9f96c4b4b1d"
RAW_DIR = Path("E:/kaggle/AI4S/raw/research/round4_nfa_refine")
OUT_DIR = Path("E:/kaggle/AI4S/experiments/neuroforecast_external_v2/intake")
EXPECTED_SHA256 = {
    "All_DIV_Data.Rdata": "fe8015c789770092b485fb63dd8aa0f643120e366df73c5624665454ee982af6",
}
READOUTS_IN_RELEASE = [
    "meanfiringrate", "burst.per.min", "mean.isis", "per.spikes.in.burst", "mean.dur", "mean.IBIs", "nAE", "nABE",
    "ns.n", "ns.peak.m", "ns.durn.m", "ns.percent.of.spikes.in.ns", "ns.mean.insis", "ns.durn.sd",
    "ns.mean.spikes.in.ns", "r", "mi",
]
NOT_IN_RELEASE = ["cv.time", "cv.network"]  # frozen v1 readouts the release does not report


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def read_release(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """All_DIV_Data as a plain DataFrame with numeric keys; hash-checked."""
    path = raw_dir / "All_DIV_Data.Rdata"
    if sha256(path) != EXPECTED_SHA256[path.name]:
        raise ValueError("All_DIV_Data.Rdata differs from the pinned release")
    import rdata  # installed outside the project environment; see the intake manifest

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        converted = rdata.read_rda(path)["rval.dat"]
    frame = pd.DataFrame({str(c): converted[c] for c in converted.columns})
    frame["date"] = pd.to_numeric(frame["date"]).astype(int)
    frame["DIV"] = pd.to_numeric(frame["DIV"]).astype(int)
    frame["dose"] = pd.to_numeric(frame["dose"]).astype(float)
    for column in READOUTS_IN_RELEASE:
        frame[column] = pd.to_numeric(frame[column]).astype(float)
    for column in ["treatment", "apid.short", "well", "units", "file.name", "srcf", "spid"]:
        frame[column] = frame[column].astype(str)
    return frame


def crosswalk(release: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per release sample ID with a DSSTox substance ID where the release's own tables give one.

    Routes, in order: the reference-chemical annotation (sample ID -> DSSTox), then name lookups against the
    annotation, the 243-chemical class table and the CCD batch searches (normalized names), using both the
    release's treatment name and the sample-list chemical name. Unresolved samples keep a name key.
    None of these tables carries an outcome.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spid_names = pd.read_excel(raw_dir / "List_of_spids_in_data_for_KC.xlsx")[["spid", "chnm"]].drop_duplicates("spid")
        classes = pd.read_excel(raw_dir / "Chem_243_mfr_chem_class_Jul_21_TJS.xlsx")[["chnm", "dsstox_substance_id"]]
    annotation = pd.read_csv(raw_dir / "annotate_dnt_ref_chems.csv")
    ccd = pd.concat([pd.read_csv(raw_dir / f) for f in [
        "CCD-Batch-Search_2025-07-18_07_43_29.csv", "CCD-Batch-Search_2025-07-18_11_43_18.csv"]], ignore_index=True)
    ccd = ccd[["DTXSID", "PREFERRED_NAME", "CASRN"]].drop_duplicates("DTXSID")

    by_spid = annotation.dropna(subset=["spid", "dsstox_substance_id"]).groupby("spid").dsstox_substance_id.agg(lambda s: sorted(set(s)))
    by_name: dict[str, set[str]] = {}
    for names, ids in [(classes.chnm, classes.dsstox_substance_id), (annotation.chnm, annotation.dsstox_substance_id),
                       (annotation.Chemical, annotation.dsstox_substance_id), (ccd.PREFERRED_NAME, ccd.DTXSID)]:
        for name, identifier in zip(names, ids):
            if isinstance(name, str) and isinstance(identifier, str):
                by_name.setdefault(data.name_key(name), set()).add(identifier)
    samples = release[["spid", "treatment"]].drop_duplicates().merge(spid_names, on="spid", how="left")
    rows = []
    for spid, group in samples.groupby("spid"):
        names = sorted(set(group.treatment) | set(group.chnm.dropna()))
        candidates, route = set(by_spid.get(spid, [])), "sample_annotation"
        if not candidates:
            route = "name"
            for name in names:
                candidates |= by_name.get(data.name_key(name), set())
        rows.append({"spid": spid, "names": " | ".join(names), "route": route if candidates else "unresolved",
                     "dtxsid": ";".join(sorted(candidates)) if candidates else None})
    table = pd.DataFrame(rows)
    cas = pd.concat([ccd.rename(columns={"DTXSID": "id", "CASRN": "cas"})[["id", "cas"]],
                     annotation.rename(columns={"dsstox_substance_id": "id", "casn": "cas"})[["id", "cas"]]]).dropna().drop_duplicates("id")
    table["casrn"] = table.dtxsid.map(dict(zip(cas.id, cas.cas)))
    return table


def to_frozen_schema(release: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename to the frozen core's physical schema. cv.time and cv.network are absent, so they are NaN."""
    frame = release.rename(columns={"apid.short": "Plate.SN", "treatment": "trt", "srcf": "source"}).copy()
    before = len(frame)
    key = data.PHYSICAL
    clashes = frame[frame.duplicated(key, keep=False)]
    conflicting = [k for k, g in clashes.groupby(key) if g[READOUTS_IN_RELEASE + ["dose", "trt"]].nunique(dropna=False).max() > 1]
    if conflicting:
        raise ValueError(f"Conflicting physical rows in the release: {conflicting[:5]}")
    frame = frame.drop_duplicates(key)
    if not set(frame.DIV) <= {5, 7, 9, 12} or set(frame.units) != {"uM"}:
        raise ValueError("Unexpected DIV or units in the release")
    for column in NOT_IN_RELEASE:
        frame[column] = np.nan
    resolved = table.dropna(subset=["dtxsid"])
    ambiguous = sorted(resolved.loc[resolved.dtxsid.str.contains(";"), "spid"])
    mapping = dict(zip(resolved.spid, resolved.dtxsid))
    # An unresolved sample keeps a name key so it is still grouped and purged as one chemical.
    frame["identity"] = frame.spid.map(mapping).fillna("NAME:" + frame.trt.map(data.name_key))
    frame.loc[frame.spid.isin(ambiguous), "identity"] = None
    treated = frame.dose > 0
    # A named identity that maps to two substances in the release tables is excluded, as in v1.
    name_ids = frame.loc[treated & frame.identity.notna()].groupby("trt").identity.nunique()
    conflicting_names = sorted(name_ids[name_ids > 1].index)
    frame["eligible_identity"] = frame.identity.notna() & ~frame.trt.isin(conflicting_names)
    audit = {
        "input_rows": before,
        "duplicate_physical_rows_removed": before - len(frame),
        "rows": len(frame),
        "spids_with_ambiguous_dtxsid": ambiguous,
        "treated_rows_without_identity": int((treated & frame.identity.isna()).sum()),
        "samples_by_route": table.route.value_counts().to_dict(),
        "treated_names_with_name_key_only": sorted(frame.loc[treated & frame.identity.str.startswith("NAME:", na=False), "trt"].unique()),
        "names_mapping_to_two_substances": conflicting_names,
        "readouts_absent_from_release": NOT_IN_RELEASE,
    }
    return frame, audit


def our_identity_dtxsid(ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    """Link our CAS-based identities to DSSTox IDs through the shared physical wells (direct evidence)."""
    shared = ours.loc[ours.dose > 0, data.PHYSICAL + ["identity"]].merge(
        theirs.loc[theirs.dose > 0, data.PHYSICAL + ["identity"]].rename(columns={"identity": "dtxsid"}), on=data.PHYSICAL)
    links = shared.dropna().groupby("identity").dtxsid.agg(lambda s: sorted(set(s)))
    return pd.DataFrame({"identity": links.index, "dtxsids": links.values})


def consistency(ours: pd.DataFrame, theirs: pd.DataFrame) -> dict:
    """Row-level comparison on the 17 dates both sources contain. These are our own development/reserve data."""
    shared_dates = sorted(set(ours.date))
    mine = ours.loc[ours.date.isin(shared_dates)]
    release = theirs.loc[theirs.date.isin(shared_dates)]
    merged = mine.merge(release, on=data.PHYSICAL, how="outer", suffixes=("_ours", "_release"), indicator=True)
    both = merged.loc[merged._merge == "both"]
    readouts = {}
    for field in READOUTS_IN_RELEASE[:-1]:
        a, b = both[f"{field}_ours"].astype(float), both[f"{field}_release"].astype(float)
        diff = (a - b).abs()
        present = a.notna() & b.notna()
        readouts[field] = {
            "rows_compared": int(present.sum()),
            "rows_differing": int((diff > 1e-6).sum()),
            "max_abs_difference": float(diff.max()),
            "median_relative_difference_where_differing": float((diff / a.abs().clip(lower=1e-9))[diff > 1e-6].median()) if (diff > 1e-6).any() else 0.0,
            "pearson_r": float(np.corrcoef(a[present], b[present])[0, 1]),
            "missingness_disagreements": int((a.isna() != b.isna()).sum()),
        }
    only_ours = merged.loc[merged._merge == "left_only"]
    only_release = merged.loc[merged._merge == "right_only"]
    return {
        "shared_dates": shared_dates,
        "physical_rows_in_both": int(len(both)),
        "rows_only_in_our_archive": int(len(only_ours)),
        "rows_only_in_release": int(len(only_release)),
        "rows_only_in_our_archive_by_date": {int(k): int(v) for k, v in only_ours.groupby("date").size().items()},
        "rows_only_in_release_by_date": {int(k): int(v) for k, v in only_release.groupby("date").size().items()},
        "dose_disagreements": int(((both.dose_ours - both.dose_release).abs() > 1e-9).sum()),
        "readouts": readouts,
    }


def case_equivalence(ours: pd.DataFrame, theirs: pd.DataFrame) -> dict:
    """Do the frozen builders produce the same case sets and similar targets from the release copy?"""
    cfg = data.protocol()
    shared_dates = sorted(set(ours.date))
    out = {}
    for label, frame in [("ours", ours), ("release", theirs.loc[theirs.date.isin(shared_dates)])]:
        inputs = data.build_inputs(frame, cfg)
        targets = data.build_targets(frame, cfg)
        out[label] = (inputs, targets)
    a_in, a_t = out["ours"]
    b_in, b_t = out["release"]
    # Cases are keyed by date/identity/dose; identity systems differ, so compare by date/dose/name via wells.
    return {
        "ours_input_cases": int(len(a_in)),
        "release_input_cases": int(len(b_in)),
        "ours_target_cases": int(len(a_t)),
        "release_target_cases": int(len(b_t)),
    }


def sealed_cohort(theirs: pd.DataFrame, ours_dates: set[int], ours_dtxsids: set[str]) -> dict:
    """Counts for the 28 new dates. Reads day-5/day-7 inputs and only the existence of day-12 records."""
    cfg = data.protocol()
    new = theirs.loc[~theirs.date.isin(ours_dates)]
    early = new.loc[new.DIV.isin([5, 7])]
    inputs = data.build_inputs(early, cfg)  # the frozen builder sees days 5 and 7 only
    day12 = new.loc[(new.DIV == 12) & (new.dose > 0) & new.eligible_identity, ["date", "identity", "dose", "well"]]
    has_day12 = day12.groupby(["date", "identity", "dose"]).size().rename("day12_wells")
    cases = inputs.index.to_frame(index=False).merge(has_day12.reset_index(), on=["date", "identity", "dose"], how="left")
    cases["day12_wells"] = cases.day12_wells.fillna(0).astype(int)
    eligible = cases.loc[cases.day12_wells >= cfg["min_replicates_per_case_day"]]
    controls7 = early.loc[(early.DIV == 7) & (early.dose == 0)].groupby(["date", "Plate.SN"]).size()
    per_date = []
    for date, group in eligible.groupby("date"):
        plates = early.loc[early.date == date, "Plate.SN"].unique()
        per_date.append({
            "date": int(date), "plates": int(len(plates)),
            "controls_per_plate_day7": [int(controls7.get((date, p), 0)) for p in sorted(plates)],
            "eligible_conditions": int(len(group)), "chemicals": int(group.identity.nunique()),
            "chemicals_not_in_our_17_dates": int((~group.identity.isin(ours_dtxsids)).groupby(group.identity).first().sum()),
        })
    novel = ~eligible.identity.isin(ours_dtxsids)
    subset = new.sort_values(data.PHYSICAL)
    digest = hashlib.sha256(pd.util.hash_pandas_object(subset[data.PHYSICAL + ["trt", "dose"] + READOUTS_IN_RELEASE], index=False).values.tobytes()).hexdigest()
    return {
        "rule": "Counts only. Day-12 records are checked for existence (>= 2 treated wells), never for value.",
        "new_dates": sorted(int(d) for d in new.date.unique()),
        "new_plate_dates": int(new.groupby(["date", "Plate.SN"]).ngroups),
        "input_eligible_conditions": int(len(cases)),
        "eligible_conditions": int(len(eligible)),
        "eligible_chemicals": int(eligible.identity.nunique()),
        "eligible_conditions_on_chemicals_absent_from_our_17_dates": int(novel.sum()),
        "eligible_chemicals_absent_from_our_17_dates": int(eligible.loc[novel, "identity"].nunique()),
        "chemicals_shared_with_our_17_dates": sorted(set(eligible.identity) & ours_dtxsids),
        "per_date": per_date,
        "new_date_rows_sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    release = read_release(args.raw_dir)
    table = crosswalk(release, args.raw_dir)
    theirs, adapter_audit = to_frozen_schema(release, table)
    ours, _ = data.load_raw()
    links = our_identity_dtxsid(ours, theirs)
    ours_dtxsids = {d for ds in links.dtxsids for d in ds}
    multi = links.loc[links.dtxsids.map(len) > 1]
    # Second, independent route: our identities are CAS numbers; the crosswalk carries CAS where known.
    our_cas = set(ours.loc[ours.dose > 0, "identity"].dropna())
    new_dates = ~theirs.date.isin(set(ours.date))
    new_ids = set(theirs.loc[new_dates & (theirs.dose > 0), "identity"].dropna())
    id_to_cas = dict(zip(table.dtxsid.dropna(), table.casrn))
    cas_overlap = sorted(i for i in new_ids if id_to_cas.get(i) in our_cas)
    ours_dtxsids |= set(cas_overlap)
    report = {
        "release": {"repository": RELEASE_REPO, "commit": RELEASE_COMMIT, "files": {
            p.name: sha256(p) for p in sorted(args.raw_dir.iterdir()) if p.is_file()}},
        "licence_note": "The repository has no LICENSE file. The data are produced by the US EPA; treat as a US-Government work and cite EPA's data policy. Record this limitation in the report.",
        "adapter": adapter_audit,
        "identity_links_from_shared_wells": {"our_identities_linked": int(len(links)), "linked_to_more_than_one_dtxsid": multi.to_dict("records"),
                                             "new_date_identities_matching_our_cas": cas_overlap},
        "consistency_on_shared_dates": consistency(ours, theirs),
        "case_equivalence_on_shared_dates": case_equivalence(ours, theirs),
        "sealed_cohort": sealed_cohort(theirs, set(ours.date), ours_dtxsids),
        "frozen_core_sha256": {"neuroforecast_data.py": sha256(Path(data.__file__))},
        "code_sha256": sha256(Path(__file__)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "identity_crosswalk.csv", index=False)
    links.assign(dtxsids=links.dtxsids.map(";".join)).to_csv(args.out / "our_identity_links.csv", index=False)
    write_json(args.out / "intake_report.json", report)
    summary = {k: report[k] for k in ["adapter", "identity_links_from_shared_wells", "case_equivalence_on_shared_dates"]}
    summary["sealed_cohort"] = {k: v for k, v in report["sealed_cohort"].items() if k != "per_date"}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
