"""Reproduce the small neural-source audits from public round-three downloads."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import scipy.io
from scipy.io.matlab import _mio5

DEFAULT = Path(__file__).resolve().parents[1] / "raw/research/round3"


def audit_deephys(root: Path) -> dict:
    path = root / "deephys_quinpirole.mat"
    with path.open("rb") as source:
        if hashlib.file_digest(source, "md5").hexdigest() != "7d1dedaf44ad18d7bd5612b6f06d26ba":
            raise ValueError("DeePhys public checksum mismatch")
    # This archived release uses MATLAB MCOS objects, not an ordinary numeric MAT.
    # Read its embedded v5 workspace and the scalar UTF-16 string payloads only.
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    stream = io.BytesIO(mat["__function_workspace__"].tobytes())
    reader = _mio5.MatFile5Reader(stream, byte_order="<", squeeze_me=True, struct_as_record=False)
    reader.initialize_read()
    stream.seek(8)
    header, _ = reader.read_var_header()
    wrapper = reader.read_var_array(header, process=True).MCOS["arr"][0]
    strings = []
    for index, value in enumerate(wrapper):
        if isinstance(value, np.ndarray) and value.dtype == np.dtype("uint64") and value.ndim == 1 and 5 <= value.size <= 1000:
            if tuple(value[:4]) == (1, 2, 1, 1) and 0 < int(value[4]) < 2000:
                text = value[5:].tobytes()[:2 * int(value[4])].decode("utf-16-le")
                strings.append({"cell": index, "text": text})
    if len(strings) != 510:
        raise ValueError("Unexpected MCOS metadata layout; do not guess at record grouping")
    records = []
    for offset in range(0, len(strings), 10):
        values = [row["text"] for row in strings[offset:offset + 10]]
        matched = re.search(r"/(M\d+)/well(\d+)/sorted/min_(\d+)", values[1])
        if matched is None or values[2] != "Quinpirole":
            raise ValueError("Unexpected recording metadata")
        device, well, phase = matched.groups()
        if values[4] != f"{device}_{int(well)}":
            raise ValueError("Independent path and ChipID fields disagree")
        records.append({"recording_ordinal": offset // 10, "device": device, "well_group": values[4], "recording_date": values[3], "segment_start_min": int(phase), "cell_type": values[5], "mutation": values[6], "source": values[7], "treatment": values[8], "short_name": values[9]})
    frame = pd.DataFrame(records)
    frame.to_csv(root / "deephys_recording_inventory.csv", index=False)
    result = {"recording_objects": len(frame), "well_groups": int(frame.well_group.nunique()), "devices": int(frame.device.nunique()), "recording_dates": frame.recording_date.unique().tolist(), "segments_min": sorted(frame.segment_start_min.unique().tolist()), "recordings_per_well": frame.groupby("well_group").size().to_dict(), "mutation_counts_by_well": frame.drop_duplicates("well_group").mutation.value_counts().to_dict(), "independence_note": "Distinct device/well recording groups; not established as independent donor lines or independent culture preparations."}
    (root / "deephys_group_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def audit_tuni(root: Path) -> list:
    rows = []
    for species in ["hpsc", "rat"]:
        log = pd.read_csv(root / f"tuni_{species}_pharma_explog.csv")
        eligible = log.loc[log.Treatment != "ExcludedWell"]
        result = {"species": species, "plated_eligible_wells": len(eligible), "treatment_counts": eligible.Treatment.value_counts().to_dict(), "plate_ids": log["Plate.SN"].unique().tolist(), "date": log["Experiment.Date"].unique().tolist(), "features": {}}
        for metric in ["sttc", "bursts_per_min", "meanfiringrate_by_active_electordes", "nae"]:
            before = pd.read_csv(root / f"tuni_{species}_baseline_{metric}.csv")
            after = pd.read_csv(root / f"tuni_{species}_pharma_{metric}.csv")
            result["features"][metric] = {"baseline_rows": len(before), "after_rows": len(after), "after_missing_values": int(after.iloc[:, -1].isna().sum()), "omitted_wells": sorted(set(eligible.Well) - set(after.well))}
        rows.append(result)
    (root / "tuni_pairing_audit.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--dataset", choices=["deephys", "tuni", "both"], default="both")
    args = parser.parse_args()
    if args.dataset in {"deephys", "both"}:
        print(json.dumps({"deephys": audit_deephys(args.root)}, indent=2), flush=True)
    if args.dataset in {"tuni", "both"}:
        print(json.dumps({"tuni": audit_tuni(args.root)}, indent=2), flush=True)
