"""Audited public EPA data and temporally separated NeuroForecast inputs/targets."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments/neuroforecast_selection_protocol.json"
DEFAULT_RAW = ROOT / "raw/research/round3/epa_nfa_raw.zip"
DEFAULT_OUT = Path("E:/kaggle/AI4S/experiments/neuroforecast_selection_v1")
KEY = ["date", "identity", "dose"]
PHYSICAL = ["date", "Plate.SN", "well", "DIV"]
ALIASES = {
    "Disulfiram - ToxCast G-8": "Disulfiram",
    "Rotenone - ToxCast G-8": "Rotenone",
    "Valinomycin - NTP": "Valinomycin",
}


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def name_key(value: str) -> str:
    return re.sub("[^a-z0-9]", "", str(value).lower())


def load_raw(path: Path = DEFAULT_RAW) -> tuple[pd.DataFrame, dict]:
    cfg = protocol()
    if digest(path) != cfg["archive_sha256"]:
        raise ValueError("The archive differs from the frozen public source")
    mapping, crosswalk = {}, []
    members = {}
    with zipfile.ZipFile(path) as archive:
        for member, column in [
            ("New NTP/sourceData/NTP Experimental Summary of Data for Analysis_Complete.csv", "preferred name"),
            ("New TC/sourceData/ToxCast Experimental Summary of Data for Analysis_edit.csv", "preferred_name"),
        ]:
            payload = archive.read(member)
            members[member] = hashlib.sha256(payload).hexdigest()
            metadata = pd.read_csv(io.BytesIO(payload), dtype=str)
            for _, row in metadata.iterrows():
                identity = row["casrn"].strip("'")
                key = name_key(row[column])
                if key in mapping and mapping[key] != identity:
                    raise ValueError("Conflicting experimental identities")
                mapping[key] = identity
                crosswalk.append({"name": row[column], "identity": identity, "member": member})
        for alias, source_name in ALIASES.items():
            mapping[name_key(alias)] = mapping[name_key(source_name)]
        frames = []
        for source, member in zip(["NTP", "TC"], cfg["raw_members"], strict=True):
            payload = archive.read(member)
            members[member] = hashlib.sha256(payload).hexdigest()
            frame = pd.read_csv(io.BytesIO(payload))
            frame["source"] = source
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["identity"] = raw["trt"].map(lambda value: mapping.get(name_key(value)))
    unmapped = sorted(raw.loc[raw.identity.isna(), "trt"].unique())
    if unmapped != sorted(cfg["unmapped_exclusions"]):
        raise ValueError(f"Unexpected unidentifiable substances: {unmapped}")
    collisions = raw[raw.duplicated(PHYSICAL, keep=False)]
    value_columns = [c for c in raw.columns if c not in {"source", "trt"}]
    for key, records in collisions.groupby(PHYSICAL):
        disagreements = [c for c in value_columns if records[c].nunique(dropna=False) > 1]
        if disagreements:
            raise ValueError(f"Conflicting physical measurements {key}: {disagreements}")
    before = len(raw)
    raw = raw.drop_duplicates(PHYSICAL).copy()
    if not set(raw.DIV).issubset({5, 7, 9, 12}) or set(raw.units) != {"uM"}:
        raise ValueError("Unexpected timing or dose units")
    raw["eligible_identity"] = raw.identity.notna() & ~raw.identity.isin(cfg["identity_conflict_exclusions"])
    # Keep zero-dose wells for plate-specific controls, including those assigned
    # to an excluded chemical. They received no chemical at any measured time.
    audit = {
        "archive_sha256": cfg["archive_sha256"],
        "member_sha256": members,
        "input_rows": before,
        "duplicate_physical_rows_removed": before - len(raw),
        "physical_recordings": len(raw),
        "physical_wells": raw.groupby(PHYSICAL[:3]).ngroups,
        "culture_date_groups": int(raw.date.nunique()),
        "named_treatments": int(raw.trt.nunique()),
        "identified_substances": int(raw.identity.nunique()),
        "eligible_substances": int(raw.loc[raw.eligible_identity, "identity"].nunique()),
        "excluded_identity_ids": cfg["identity_conflict_exclusions"],
        "unmapped_names": unmapped,
        "silent_recordings": int((raw.nAE == 0).sum()),
        "aliases": ALIASES,
        "crosswalk": crosswalk,
    }
    return raw, audit


def _observed_day(raw: pd.DataFrame, day: int, cfg: dict) -> pd.DataFrame:
    """Produce case summaries from one day only; never fills from another day."""
    current = raw.loc[raw.DIV == day].copy()
    fields = cfg["feature_readouts"]
    controls = current.loc[current.dose == 0].groupby(["date", "Plate.SN"])[fields].median()
    controls = controls.add_prefix("control__")
    current = current.join(controls, on=["date", "Plate.SN"], validate="many_to_one")
    current = current.loc[current.eligible_identity & (current.dose > 0)].copy()
    pseudo = cfg["target_pseudocount"]
    current["relative_ns"] = (current["ns.n"] + pseudo) / (current["control__ns.n"] + pseudo)
    by_case = current.groupby(KEY, sort=True)
    summaries = by_case[fields + ["control__" + x for x in fields]].mean()
    counts = by_case.size().rename("replicates")
    # Control absence is an unresolved input, not a fabricated zero.
    relative = by_case.relative_ns.mean()
    summaries["normalized_ns"] = np.log2(relative)
    summaries["replicates"] = counts
    for field in fields:
        summaries["missing__" + field] = by_case[field].agg(lambda x: x.isna().mean())
    enough_replicates = summaries.replicates >= cfg["min_replicates_per_case_day"]
    enough_reference_values = by_case.relative_ns.count() >= cfg["min_replicates_per_case_day"]
    return summaries.loc[enough_replicates & enough_reference_values]


def build_inputs(raw: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = protocol() if cfg is None else cfg
    pieces = []
    for day in cfg["cutoff_days"]:
        # Pass only the currently permitted measurements to the summarizer.
        observed = _observed_day(raw.loc[raw.DIV == day], day, cfg)
        features = pd.DataFrame(index=observed.index)
        features[f"normalized_ns_d{day}"] = observed.normalized_ns
        for field in cfg["feature_readouts"]:
            features[f"raw__{field}__d{day}"] = np.arcsinh(observed[field])
            features[f"control__{field}__d{day}"] = np.arcsinh(observed["control__" + field])
            features[f"missing__{field}__d{day}"] = observed["missing__" + field]
        pieces.append(features)
    inputs = pieces[0].join(pieces[1], how="inner", validate="one_to_one")
    inputs["log10_dose"] = np.log10(inputs.index.get_level_values("dose").to_numpy(dtype=float))
    finite_reference = np.isfinite(inputs[[f"normalized_ns_d{d}" for d in cfg["cutoff_days"]]]).all(axis=1)
    return inputs.loc[finite_reference].replace([np.inf, -np.inf], np.nan)


def build_targets(raw: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    cfg = protocol() if cfg is None else cfg
    late = _observed_day(raw.loc[raw.DIV == cfg["target_day"]], cfg["target_day"], cfg)
    return late.normalized_ns.rename("target").loc[lambda x: np.isfinite(x)]


def reserve_ids(raw: pd.DataFrame, cfg: dict) -> set[str]:
    return set(raw.loc[raw.date.isin(cfg["reserved_dates"]), "identity"].dropna())


def development(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    reserved = reserve_ids(raw, cfg)
    # No reserve day-12 readout is passed into either builder during selection.
    dev_raw = raw.loc[~raw.date.isin(cfg["reserved_dates"])].copy()
    inputs = build_inputs(dev_raw, cfg)
    allowed = ~inputs.index.get_level_values("identity").isin(reserved)
    inputs = inputs.loc[allowed]
    target = build_targets(dev_raw, cfg)
    common = inputs.index.intersection(target.index, sort=False)
    manifest = {
        "reserved_dates": cfg["reserved_dates"],
        "reserved_identities": sorted(reserved),
        "reserve_outcomes_scored": False,
        "eligible_early_cases": len(inputs),
        "cases_missing_eligible_target": len(inputs) - len(common),
    }
    return inputs.loc[common], target.loc[common], manifest


def split_indices(inputs: pd.DataFrame):
    metadata = inputs.index.to_frame(index=False)
    for date in sorted(metadata.date.unique()):
        test = np.flatnonzero(metadata.date.to_numpy() == date)
        chemicals = set(metadata.iloc[test].identity)
        train = np.flatnonzero((metadata.date.to_numpy() != date) & ~metadata.identity.isin(chemicals).to_numpy())
        if not len(train) or not len(test):
            raise ValueError("Empty development fold")
        yield int(date), train, test
