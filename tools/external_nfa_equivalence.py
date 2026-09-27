"""Phase 1 equivalence check: does the frozen v1 tool behave the same on the EPA release copy of data we already know?

Everything here uses the 17 dates already in our archive (development and the spent reserve), so no new
outcome is read. It measures two input changes the external test will carry:

1. the release re-processed the recordings (small value differences), and
2. the release lacks the cv.time and cv.network readouts, which the frozen models use.

It runs the shipped `neuroforecast_triage.triage` function unchanged, and re-scores the already-spent reserve from
the release copy. That re-score validates nothing new; it checks that the adapter reproduces a known result.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_nfa_intake as intake  # noqa: E402
import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_triage as triage_tool  # noqa: E402

RESERVE = [20170920, 20171004, 20171011]
AUDITED = Path("D:/kaggle/AI4S-1/neuroforecast/evaluation/reserve_intervals.csv")


def run_tool(observed: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    table, _ = triage_tool.triage(observed.loc[observed.DIV.isin([5, 7])].copy(), bundle)
    return table.set_index("case_id")


def score(table: pd.DataFrame, targets: pd.Series) -> dict:
    frame = table.join(targets.rename("target"), how="inner")
    frame = frame.reset_index()
    product = frame.assign(prediction=frame.forecast_log2)
    persistence = frame.assign(prediction=frame.persistence_log2)
    retained = frame.forecast_trusted
    error = abs(frame.forecast_log2 - frame.target)
    return {
        "conditions": int(len(frame)),
        "product_date_macro_mae": gate.macro_mae(product),
        "persistence_date_macro_mae": gate.macro_mae(persistence),
        "per_date_product": {int(k): float(v) for k, v in gate.date_losses(product).items()},
        "per_date_persistence": {int(k): float(v) for k, v in gate.date_losses(persistence).items()},
        "declined": int((~retained).sum()),
        "retained_case_mae": float(error[retained].mean()),
        "all_case_mae": float(error.mean()),
        "low_reference_activity_flags": int(frame.quality_low_reference_activity.sum()),
    }


def compare(a: pd.DataFrame, b: pd.DataFrame, column: str) -> dict:
    common = a.index.intersection(b.index)
    diff = abs(a.loc[common, column] - b.loc[common, column])
    return {"common": int(len(common)), "only_first": int(len(a.index.difference(b.index))), "only_second": int(len(b.index.difference(a.index))),
            "median_abs_difference": float(diff.median()), "p95_abs_difference": float(diff.quantile(0.95)), "max_abs_difference": float(diff.max())}


def main() -> None:
    bundle = triage_tool.load_bundle(triage_tool.DEFAULT_BUNDLE, triage_tool.DEFAULT_LOCK, triage_tool.DEFAULT_RESERVE)
    cfg = data.protocol()
    ours, _ = data.load_raw()
    release = intake.read_release()
    theirs, _ = intake.to_frozen_schema(release, intake.crosswalk(release))
    links = intake.our_identity_dtxsid(ours, theirs)
    to_cas = {ids[0]: identity for identity, ids in zip(links.identity, links.dtxsids)}
    shared = theirs.loc[theirs.date.isin(set(ours.date))].copy()
    shared["identity"] = shared.identity.map(to_cas).where(shared.identity.map(to_cas).notna(), shared.identity)
    # Our pipeline excludes three conflicting CAS identities; keep that exclusion on the release copy too.
    shared["eligible_identity"] = shared.eligible_identity & ~shared.identity.isin(cfg["identity_conflict_exclusions"])

    reserve_ours = ours.loc[ours.date.isin(RESERVE)]
    reserve_release = shared.loc[shared.date.isin(RESERVE)]
    no_cv = reserve_ours.copy()
    no_cv[intake.NOT_IN_RELEASE] = np.nan

    audited = pd.read_csv(AUDITED, dtype={"identity": str}).set_index("case_id")
    runs = {"archive": run_tool(reserve_ours, bundle), "archive_without_cv": run_tool(no_cv, bundle), "release": run_tool(reserve_release, bundle)}
    targets = {"archive": data.build_targets(reserve_ours, cfg), "release": data.build_targets(reserve_release, cfg)}
    targets = {k: pd.Series(v.to_numpy(), index=gate.case_ids(v.index)) for k, v in targets.items()}

    audited_cases = audited.index
    report = {
        "purpose": "Adapter equivalence on already-known data; no new outcome is read.",
        "archive_reproduces_audited_forecasts": compare(runs["archive"].loc[runs["archive"].index.intersection(audited_cases)], audited.rename(columns={"prediction": "forecast_log2"}), "forecast_log2"),
        "effect_of_missing_cv_readouts_on_forecasts": compare(runs["archive"], runs["archive_without_cv"], "forecast_log2"),
        "effect_of_missing_cv_readouts_on_difficulty": compare(runs["archive"], runs["archive_without_cv"], "difficulty"),
        "effect_of_release_copy_on_forecasts": compare(runs["archive"], runs["release"], "forecast_log2"),
        "effect_of_release_copy_on_difficulty": compare(runs["archive"], runs["release"], "difficulty"),
        "release_vs_archive_targets": compare(targets["archive"].to_frame("t"), targets["release"].to_frame("t"), "t"),
        "reserve_scored_on_audited_cases": {
            "archive": score(runs["archive"].loc[runs["archive"].index.intersection(audited_cases)], targets["archive"]),
            "archive_without_cv": score(runs["archive_without_cv"].loc[runs["archive_without_cv"].index.intersection(audited_cases)], targets["archive"]),
            "release": score(runs["release"].loc[runs["release"].index.intersection(audited_cases)], targets["release"]),
        },
        "audited_reference": {"product_date_macro_mae": 0.7716, "persistence_date_macro_mae": 0.6088, "declined": 32, "retained_case_mae": 0.446, "low_reference_activity_flags": 84},
        "code_sha256": intake.sha256(Path(__file__)),
        "triage_sha256": intake.sha256(Path(triage_tool.__file__)),
    }
    out = intake.OUT_DIR / "equivalence.json"
    intake.write_json(out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
