"""Reproduce the sealed external test from a fresh clone, without editing the locked code.

The locked modules (external_nfa_intake.py, neuroforecast_v2.py, neuroforecast_v2_external.py) hold the
author's storage paths as defaults. This wrapper points them at directories you choose, then runs a stage.
Their bytes, and therefore the hashes recorded in evaluation/v2_development/development_lock.json, are unchanged.

    python -X utf8 -B tools/fetch_epa_refinement.py --out raw/epa_nfa_refinement
    python -X utf8 -B tools/fetch_epa_data.py --out raw/epa_nfa_raw.zip          # needed for intake and equivalence only
    python -X utf8 -B tools/reproduce_external.py intake      --raw raw/epa_nfa_refinement --archive raw/epa_nfa_raw.zip --out runs/external
    python -X utf8 -B tools/reproduce_external.py prepare     --raw raw/epa_nfa_refinement --out runs/external
    python -X utf8 -B tools/reproduce_external.py develop     --raw raw/epa_nfa_refinement --out runs/external
    python -X utf8 -B tools/reproduce_external.py freeze      --raw raw/epa_nfa_refinement --out runs/external
    python -X utf8 -B tools/reproduce_external.py external    --raw raw/epa_nfa_refinement --out runs/external --unseal
    python -X utf8 -B tools/reproduce_external.py audit       --raw raw/epa_nfa_refinement --out runs/external

`develop` is deterministic (single-threaded trees); its outputs can be compared file by file with
evaluation/v2_development. `external` scores once per output directory, as in the original run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_nfa_intake as intake  # noqa: E402
import neuroforecast_data as data  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["intake", "equivalence", "prepare", "develop", "freeze", "external", "audit"])
    parser.add_argument("--raw", type=Path, required=True, help="Directory from fetch_epa_refinement.py")
    parser.add_argument("--archive", type=Path, help="The v1 EPA archive (fetch_epa_data.py); intake and equivalence only")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--unseal", action="store_true")
    args = parser.parse_args()
    intake.RAW_DIR = args.raw
    intake.OUT_DIR = args.out / "intake"
    v2.OUT = args.out
    if args.archive:
        data.DEFAULT_RAW = args.archive
        data.load_raw.__defaults__ = (args.archive,)
    if args.stage == "intake":
        sys.argv = [sys.argv[0], "--raw-dir", str(args.raw), "--out", str(args.out / "intake")]
        intake.main()
    elif args.stage == "equivalence":
        import external_nfa_equivalence as equivalence
        equivalence.AUDITED = ROOT / "evaluation/reserve_intervals.csv"
        equivalence.main()
    elif args.stage in {"prepare", "develop"}:
        v2.prepare(args.out) if args.stage == "prepare" else v2.develop(args.out, args.jobs)
    else:
        import neuroforecast_v2_external as ext
        {"freeze": lambda: ext.freeze(args.out), "external": lambda: ext.external(args.out, args.unseal),
         "audit": lambda: ext.audit(args.out)}[args.stage]()


if __name__ == "__main__":
    main()
