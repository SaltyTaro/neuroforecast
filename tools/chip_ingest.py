"""Chip ingestion (ingestion only, not validation): spike times from a neural chip -> EPA-style well readouts ->
the batch reference check and a domain check against EPA's control wells.

NeuroForecast reads EPA's per-well summaries of a 15-minute, 16-electrode recording. A chip exports spike times.
This module turns spike times into 16 of the 17 readouts with EPA's published criteria (docs/input_schema.md):
active electrodes fire at least 5 spikes/min; bursts by the maximum-interval method (start ISI <= 0.1 s, end ISI
> 0.25 s, merge gaps < 0.8 s, keep bursts of >= 5 spikes and >= 0.05 s); bursting electrodes have >= 0.5 bursts/min;
network spikes from 0.05 s bins in which at least 5 electrodes fire; correlation from 10 ms binary trains of the
active electrodes. These are re-implementations, not EPA's code: network-spike duration is the run length rather
than EPA's half-maximum width, and mutual information (`mi`) is not computed. Use EPA's code before trusting any
forecast; this module shows the data path and what does not match.

Example data: Zenodo 4679064 (CC BY 4.0; Vakilna, Tang, Wheeler and Brewer 2021), spontaneous 5-minute recordings of
ten rat hippocampal cultures on a four-compartment microfluidic MEA chip (EC, DG, CA3, CA1, joined by axon tunnels).
Recordings 2 and 4 are one culture at days 21 and 37, so there are nine independent cultures. Each compartment is
treated as a well. The cultures are untreated and recorded at days 21 to 37, so the tool's forecast (days 5 and 7,
treated and control wells) cannot run on them; only the readouts, the reference check and the domain check do.

The "reference check" here only asks whether the chip's median compartment has network activity: this archive has no
vehicle-treated chips, so on mature untreated cultures it cannot fail. The output also cannot enter the tool yet: it
lacks `mi` and the plate, treatment and concentration columns the tool requires.

    python -X utf8 -B tools/chip_ingest.py --zenodo-4679064 <folder with spike_trains_data/>   # rewrites the shipped evidence
    python -X utf8 -B tools/chip_ingest.py --spikes my_chip_spikes.csv --out <folder>   # columns: recording, well, electrode, time_s, duration_s
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/chip_ingestion"
EPA_SECONDS = 900.0            # EPA recordings are 15 minutes
EPA_ELECTRODES = 16
AE_PER_MIN = 5.0
ABE_BURSTS_PER_MIN = 0.5
BURST = {"beg_isi": 0.1, "end_isi": 0.25, "min_ibi": 0.8, "min_durn": 0.05, "min_spikes": 5}
NS_BIN, NS_MIN_ELECTRODES = 0.05, 5
R_BIN = 0.01
READOUTS = ["meanfiringrate", "burst.per.min", "mean.isis", "per.spikes.in.burst", "mean.dur", "mean.IBIs", "nAE", "nABE",
            "ns.n", "ns.peak.m", "ns.durn.m", "ns.percent.of.spikes.in.ns", "ns.mean.insis", "ns.durn.sd", "ns.mean.spikes.in.ns", "r", "mi"]
# Counts depend on recording length and electrode number; they are rescaled to a 15-minute, 16-electrode well.
PER_RECORDING = {"ns.n"}
PER_ELECTRODES = {"nAE", "nABE"}


def bursts(t: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-interval bursts as (first, last) spike indices."""
    if len(t) < 2:
        return []
    isi = np.diff(t)
    found, i = [], 0
    while i < len(isi):
        if isi[i] <= BURST["beg_isi"]:
            j = i
            while j + 1 < len(isi) and isi[j + 1] <= BURST["end_isi"]:
                j += 1
            found.append([i, j + 1])
            i = j + 1
        else:
            i += 1
    merged = []
    for b in found:
        if merged and t[b[0]] - t[merged[-1][1]] < BURST["min_ibi"]:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return [(a, b) for a, b in merged if b - a + 1 >= BURST["min_spikes"] and t[b] - t[a] >= BURST["min_durn"]]


def well_readouts(spikes: list[np.ndarray], duration: float) -> dict:
    """EPA-style readouts of one well (a list of per-electrode spike-time arrays, seconds) recorded for `duration` s."""
    minutes = duration / 60
    counts = np.array([len(t) for t in spikes])
    active = counts / minutes >= AE_PER_MIN
    out = {k: np.nan for k in READOUTS}
    out["nAE"] = int(active.sum())
    out["meanfiringrate"] = float((counts[active] / duration).mean()) if active.any() else 0.0
    per_electrode = [bursts(np.sort(t)) for t in spikes]
    rates = np.array([len(b) / minutes for b in per_electrode])
    out["burst.per.min"] = float(rates[active].mean()) if active.any() else 0.0
    bursting = rates >= ABE_BURSTS_PER_MIN
    out["nABE"] = int(bursting.sum())
    pct, durs, isis, ibis = [], [], [], []
    for t, b, keep in zip(spikes, per_electrode, bursting):
        if not keep:
            continue
        t = np.sort(t)
        pct.append(100 * sum(e - s + 1 for s, e in b) / len(t))
        durs.append(np.mean([t[e] - t[s] for s, e in b]))
        isis.append(np.mean(np.concatenate([np.diff(t[s:e + 1]) for s, e in b])))
        if len(b) > 1:
            ibis.append(np.mean([t[b[k + 1][0]] - t[b[k][1]] for k in range(len(b) - 1)]))
    out["per.spikes.in.burst"] = float(np.mean(pct)) if pct else 0.0
    out["mean.dur"] = float(np.mean(durs)) if durs else np.nan
    out["mean.isis"] = float(np.mean(isis)) if isis else np.nan
    out["mean.IBIs"] = float(np.mean(ibis)) if ibis else np.nan

    # Network spikes: maximal runs of 0.05 s bins in which at least 5 electrodes fire.
    bins = int(np.ceil(duration / NS_BIN))
    binned = np.zeros((len(spikes), bins), dtype=np.int32)
    for k, t in enumerate(spikes):
        np.add.at(binned[k], np.minimum((np.asarray(t) / NS_BIN).astype(int), bins - 1), 1)
    electrodes = (binned > 0).sum(axis=0)
    qualifying = electrodes >= NS_MIN_ELECTRODES
    edges = np.diff(np.concatenate([[0], qualifying.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    out["ns.n"] = int(len(starts))
    total = binned.sum()
    if len(starts):
        in_ns = np.array([binned[:, s:e].sum() for s, e in zip(starts, ends)])
        peaks = np.array([electrodes[s:e].max() for s, e in zip(starts, ends)])
        peak_times = np.array([(s + np.argmax(electrodes[s:e])) * NS_BIN for s, e in zip(starts, ends)])
        lengths = (ends - starts) * NS_BIN
        out["ns.peak.m"] = float(peaks.mean())
        out["ns.durn.m"] = float(lengths.mean())
        out["ns.mean.spikes.in.ns"] = float(in_ns.mean())
        if len(starts) > 1:
            out["ns.mean.insis"] = float(np.diff(peak_times).mean())
            out["ns.durn.sd"] = float(lengths.std(ddof=1))
    out["ns.percent.of.spikes.in.ns"] = float(100 * (in_ns.sum() if len(starts) else 0) / total) if total else 0.0

    # Correlation: mean pairwise Pearson r of 10 ms binary trains over active electrodes; 0 with at most one.
    idx = np.where(active)[0]
    if len(idx) > 1:
        rb = int(np.ceil(duration / R_BIN))
        trains = np.zeros((len(idx), rb), dtype=np.float32)
        for row, k in enumerate(idx):
            trains[row, np.minimum((np.asarray(spikes[k]) / R_BIN).astype(int), rb - 1)] = 1
        c = np.corrcoef(trains)
        out["r"] = float(np.nanmean(c[np.triu_indices(len(idx), 1)]))
    else:
        out["r"] = 0.0
    return out


def to_epa_scale(row: dict, duration: float, electrodes: int) -> dict:
    """Rescale counts to a 15-minute recording and a 16-electrode well, the scale the tool was trained on."""
    scaled = dict(row)
    for k in PER_RECORDING:
        scaled[k] = row[k] * EPA_SECONDS / duration
    for k in PER_ELECTRODES:
        scaled[k] = row[k] * EPA_ELECTRODES / electrodes
    return scaled


# --------------------------------------------------------------------------- example data

FOLDER = re.compile(r"^(?P<culture>\d+)ECDGCA3CA1 (?P<mea>\d+) (?P<plated>\d{6}) (?P<recorded>\d{6}) d(?P<div>\d+) ")


def load_zenodo_4679064(root: Path) -> list[dict]:
    """One entry per recording and compartment: spike times per electrode (s), duration, metadata."""
    import scipy.io as sio

    wells = []
    for folder in sorted((root / "spike_trains_data").iterdir(), key=lambda p: int(FOLDER.match(p.name)["culture"])):
        meta = FOLDER.match(folder.name).groupdict()
        for compartment in ["EC", "DG", "CA3", "CA1"]:
            mea = sio.loadmat(folder / "spikes" / f"mea_{compartment}.mat")["mea"][0, 0]
            fs = float(mea["par"][0, 0]["fs"].ravel()[0])
            # Some files store the per-electrode trains as a row, others as a column.
            trains = list(mea["spike_data"].ravel())
            author_trains = list(mea["burst_data"].ravel())
            duration = trains[0].shape[0] / fs
            spikes = [np.sort(t.tocoo().row if t.shape[0] > 1 else t.tocoo().col) / fs for t in trains]
            # The authors' own burst trains (1 inside a burst): their burst count per electrode, for a sanity check.
            author = [int(np.sum(np.diff(np.concatenate([[0], (np.asarray(b.toarray() if hasattr(b, "toarray") else b).ravel() != 0).astype(int)])) == 1))
                      for b in author_trains]
            if len(author) != len(spikes):
                raise ValueError(f"{folder.name} {compartment}: {len(spikes)} spike trains but {len(author)} burst trains")
            wells.append({"recording": int(meta["culture"]), "chip": meta["mea"], "plated": "20" + meta["plated"],
                          "recorded": "20" + meta["recorded"], "DIV": int(meta["div"]), "well": compartment,
                          "electrodes": len(spikes), "duration_s": duration, "spikes": spikes, "author_bursts": author})
    return wells


def load_spike_csv(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    need = {"recording", "well", "electrode", "time_s", "duration_s"}
    if not need <= set(frame.columns):
        raise ValueError(f"the spike file needs columns {sorted(need)}")
    wells = []
    for (recording, well), g in frame.groupby(["recording", "well"], sort=True):
        spikes = [np.sort(e.time_s.to_numpy(dtype=float)) for _, e in g.groupby("electrode")]
        wells.append({"recording": recording, "well": well, "DIV": g.DIV.iloc[0] if "DIV" in g else np.nan,
                      "electrodes": len(spikes), "duration_s": float(g.duration_s.iloc[0]), "spikes": spikes})
    return wells


# --------------------------------------------------------------------------- checks

def epa_controls(day: int = 12) -> pd.DataFrame:
    wells = pd.read_csv(ROOT / "evaluation/v3/prepared/wells.csv.gz")
    controls = wells.loc[wells.dose == 0]
    return controls[[f"{k}__d{day}" for k in READOUTS]].rename(columns=lambda c: c.split("__")[0])


def domain_check(table: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    """Percentile of each chip readout among EPA control wells, and whether it sits inside their 1st-99th percentiles."""
    rows = []
    for k in READOUTS:
        ref = controls[k].dropna().to_numpy()
        lo, hi = np.quantile(ref, [0.01, 0.99])
        for _, r in table.iterrows():
            v = r[k]
            rows.append({"recording": r.recording, "well": r.well, "readout": k, "value": v,
                         "epa_percentile": float((ref < v).mean() * 100) if np.isfinite(v) else np.nan,
                         "inside_epa_1_99": bool(lo <= v <= hi) if np.isfinite(v) else None})
    return pd.DataFrame(rows)


def run(wells: list[dict], name: str, out: Path = OUT) -> dict:
    rows, ours, theirs = [], [], []
    for w in wells:
        raw = well_readouts(w["spikes"], w["duration_s"])
        scaled = to_epa_scale(raw, w["duration_s"], w["electrodes"])
        rows.append({k: v for k, v in w.items() if k not in ("spikes", "author_bursts")} | {"spikes_total": int(sum(len(t) for t in w["spikes"]))}
                    | {k: scaled[k] for k in READOUTS})
        if "author_bursts" in w:
            ours += [len(bursts(t)) for t in w["spikes"]]
            theirs += w["author_bursts"]
    table = pd.DataFrame(rows)
    domain = domain_check(table, epa_controls(12))
    reference = table.groupby("recording")["ns.n"].median()
    summary = {
        "status": "ingestion only, not validation: readouts are re-implementations of EPA's criteria, and the tool's forecast "
                  "cannot run on untreated recordings without days 5 and 7",
        "source": name, "recordings": int(table.recording.nunique()), "wells": int(len(table)),
        "readouts_computed": [k for k in READOUTS if table[k].notna().any()],
        "readouts_missing": [k for k in READOUTS if table[k].isna().all()],
        "reference_check": {"rule": "usable when the median well has at least one network spike per 15 minutes",
                            "usable_recordings": int((reference >= 1).sum()), "median_network_spikes_per_15_min": float(reference.median())},
        "domain_check": {"reference": "EPA control wells at day 12 (all 41 batches)",
                         "share_inside_epa_1_99_by_readout": {k: float(g.inside_epa_1_99.dropna().mean())
                                                               for k, g in domain.groupby("readout") if g.inside_epa_1_99.notna().any()},
                         "share_inside_overall": float(domain.inside_epa_1_99.dropna().mean())},
    }
    if ours:
        a, b = pd.Series(ours, dtype=float), pd.Series(theirs, dtype=float)
        summary["burst_sanity_check"] = {
            "against": "the dataset authors' own burst trains, a different burst detector",
            "electrodes": int(len(a)), "spearman_per_electrode_burst_count": float(a.rank().corr(b.rank())),
            "bursts_ours": int(a.sum()), "bursts_authors": int(b.sum())}
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{name}_readouts.csv", index=False, lineterminator="\n")
    domain.to_csv(out / f"{name}_domain.csv", index=False, lineterminator="\n")
    (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zenodo-4679064", type=Path, metavar="FOLDER", help="Folder holding the extracted spike_trains_data/")
    group.add_argument("--spikes", type=Path, help="CSV of spike times: recording, well, electrode, time_s, duration_s")
    parser.add_argument("--out", type=Path, help="Output folder (required with --spikes; the Zenodo example defaults to "
                                                 "evaluation/chip_ingestion/, the shipped evidence)")
    args = parser.parse_args()
    if args.spikes and args.out is None:
        parser.error("--spikes needs --out, so that new data never overwrite the shipped evidence")
    if args.zenodo_4679064:
        summary = run(load_zenodo_4679064(args.zenodo_4679064), "zenodo_4679064", args.out or OUT)
    else:
        summary = run(load_spike_csv(args.spikes), args.spikes.stem, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
