"""Download the public EPA network formation assay archive and verify its checksum."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = "https://pasteur.epa.gov/uploads/10.23719/1503191/NTP_TC_Analysis.zip"
SHA256 = "fd92c1339bb764ee9b96c935bf31867c12640505e5c3f065ad5956a7eea08cfb"
RECORD = "https://doi.org/10.23719/1503191"


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fetch(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and digest(out) == SHA256:
        print(f"{out} already present and verified")
        return
    print(f"downloading {URL} (about 160 MB)")
    with urllib.request.urlopen(URL) as response, out.open("wb") as target:
        while chunk := response.read(1 << 20):
            target.write(chunk)
    found = digest(out)
    if found != SHA256:
        raise SystemExit(f"checksum mismatch: expected {SHA256}, found {found}. Obtain the archive from {RECORD} and retry.")
    print(f"verified {out} sha256={found}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("raw/epa_nfa_raw.zip"))
    fetch(parser.parse_args().out)
