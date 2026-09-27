"""Download the pinned EPA NFA refinement files used by the external test, and verify each checksum.

Source: https://github.com/USEPA/CompTox-DNT-NFA-Refinement at commit 01adf3e1a0068c87fe221d60df36b9f96c4b4b1d.
The repository has no LICENSE file; the data were produced by the US EPA and are cited to it here.
About 3 MB in total.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request

COMMIT = "01adf3e1a0068c87fe221d60df36b9f96c4b4b1d"
BASE = f"https://raw.githubusercontent.com/USEPA/CompTox-DNT-NFA-Refinement/{COMMIT}"
FILES = {
    "source_files/All_DIV_Data.Rdata": "fe8015c789770092b485fb63dd8aa0f643120e366df73c5624665454ee982af6",
    "source/List_of_spids_in_data_for_KC.xlsx": "68758f57179ee2c6ab462ad08604894e4d6dd5ba6ee9128dd6ee60a01d048504",
    "source_files/Chem_243_mfr_chem_class_Jul_21_TJS.xlsx": "a482a21073f4bed296600d1b9e11defd437451fd424fc31ff6a9fde858ca36de",
    "source_files/CCD-Batch-Search_2025-07-18_07_43_29.csv": "4c7418951bcc35e6f97c71b018ce7d363d14b4dccc99cc637798ae7d00da4a8c",
    "source_files/CCD-Batch-Search_2025-07-18_11_43_18.csv": "5463c0251a6aa3f68c9a7292edd529486f2d9e3c3fee6905513370cb77b74d84",
    "output/annotate_dnt_ref_chems.csv": "664de5d0c26671f1f45bd8bfe3e1ae44c100c5249f6b2dc1aa1a288d7a045875",
    "output/Bioactivity_bin_tbl_comp_methods.csv": "c0ced486478c2d33a0cccb32a3dc1ec81f2752f37bb1b7c7a515185c661fea68",
}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fetch(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for remote, expected in FILES.items():
        target = out / Path(remote).name
        if target.exists() and digest(target) == expected:
            print(f"ok       {target.name}")
            continue
        with urllib.request.urlopen(f"{BASE}/{remote}") as response:
            target.write_bytes(response.read())
        found = digest(target)
        if found != expected:
            raise SystemExit(f"checksum mismatch for {remote}: expected {expected}, found {found}")
        print(f"fetched  {target.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("raw/epa_nfa_refinement"))
    fetch(parser.parse_args().out)
