"""Post hoc analyses of the locked v3 study. None of these existed in the protocol; all are labelled post hoc.

1. Ablations: each PCT ablation against the full PCT, paired over dates.
2. Twin baseline: one Extra Trees per target on the same outer folds and inputs, predicting the same 36
   later-day targets, so the PCT twin head is compared with a learned model, not only with carry-forward.
3. Cohorts: archive (2016-17) against external (2014-16) dates for v2, PCT and their ensemble.
4. Replay: retrain the PCT on outer fold 0 with the locked code and seeds; the forecasts must be bit-identical.

Writes E:/kaggle/AI4S/experiments/neuroforecast_v3/analysis/posthoc.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_v3_analyze as an  # noqa: E402
import neuroforecast_v3_data as v3d  # noqa: E402
import neuroforecast_v3_run as run  # noqa: E402

OUT = v3d.OUT


def outer(out: Path, name: str, n: int, column: str = "pred") -> np.ndarray:
    frame = pd.read_csv(out / "predictions" / f"{name}.csv")
    part = frame.loc[frame.level == "outer"]
    values = np.full(n, np.nan)
    values[part.position.to_numpy()] = part[column].to_numpy()
    return values


def ablations(out: Path, cases: pd.DataFrame, seed: int) -> dict:
    n, usable = len(cases), cases.usable.to_numpy(dtype=bool)
    full = v3d.date_losses(cases, outer(out, "pct", n), usable)
    result = {}
    for name in ["pct_no_context", "pct_no_relations", "pct_no_aux"]:
        losses = v3d.date_losses(cases, outer(out, name, n), usable)
        result[name] = {"mae": float(losses.mean()), "pct_mae": float(full.mean()),
                        "ablation_minus_pct": an.paired_dates(losses, full, seed)}
    return result


def twin_baseline(out: Path, cases: pd.DataFrame, features: pd.DataFrame, twin: pd.DataFrame, cfg: dict) -> dict:
    """Multi-output Extra Trees (v2 hyperparameters) on the full feature set plus context, outer folds only."""
    x = run.with_chemical_context(features, cases)
    cols = run.column_sets(x)["full_ctx"]
    n, usable = len(cases), cases.usable.to_numpy(dtype=bool)
    targets = twin.columns.tolist()
    pred = np.full((n, len(targets)), np.nan)
    # One single-output model per target, each fitted on every usable training case with that target observed
    # (a joint multi-output fit would keep only cases with all 36 targets, discarding much of the data).
    with threadpool_limits(limits=8):
        for k, dates, train, test in v3d.outer_splits(cases):
            for j, name in enumerate(targets):
                y = twin[name].to_numpy()[train]
                ok = usable[train] & np.isfinite(y)
                model = run.make_model({"kind": "extra_trees"}, cfg)
                model.set_params(extratreesregressor__n_jobs=8)
                model.fit(x.iloc[train[ok]][cols], y[ok])
                pred[test, j] = model.predict(x.iloc[test][cols])
    pct = pd.read_csv(out / "predictions/pct.csv")
    pct = pct.loc[pct.level == "outer"].set_index("position")
    # Per-case evidence for the verifier: truth, PCT twin, per-target Extra Trees and carry-forward for every target.
    per_case = cases[["case_id", "date", "identity", "usable"]].copy()
    result = {}
    for j, name in enumerate(targets):
        truth = twin[name].to_numpy()
        model = np.full(n, np.nan)
        model[pct.index.to_numpy()] = pct[f"twin__{name}"].to_numpy()
        base = features.normalized_ns_d7.to_numpy() if name.startswith("normalized_ns") else features[f"contrast__{name.split('__')[1]}__d7"].to_numpy()
        ok = usable & np.isfinite(truth) & np.isfinite(model) & np.isfinite(pred[:, j]) & np.isfinite(base)
        frame = cases.loc[ok, ["date", "identity"]].assign(target=truth[ok])
        scores = {label: gate.date_losses(frame.assign(prediction=values[ok]))
                  for label, values in [("pct_twin", model), ("extra_trees_per_target", pred[:, j]), ("persistence", base)]}
        result[name] = {label: float(s.mean()) for label, s in scores.items()}
        result[name]["pct_minus_et"] = an.paired_dates(scores["pct_twin"], scores["extra_trees_per_target"], 20261007)
        per_case[f"truth__{name}"], per_case[f"pct__{name}"] = truth, model
        per_case[f"et__{name}"], per_case[f"persistence__{name}"] = pred[:, j], base
    per_case.to_csv(out / "analysis/posthoc_twin_cases.csv", index=False)
    return result


def cohorts(out: Path, cases: pd.DataFrame) -> dict:
    n, usable = len(cases), cases.usable.to_numpy(dtype=bool)
    preds = {name: outer(out, name, n) for name in ["persistence_day7", "v2_incumbent", "pct", "et_small_chem"]}
    preds["ens_v2_pct"] = (preds["v2_incumbent"] + preds["pct"]) / 2
    result = {}
    for cohort in ["archive17", "external28"]:
        mask = usable & (cases.cohort == cohort).to_numpy()
        losses = {k: v3d.date_losses(cases, v, mask) for k, v in preds.items()}
        result[cohort] = {"dates": int(cases.loc[mask].date.nunique()), "cases": int(mask.sum()),
                          "mae": {k: float(v.mean()) for k, v in losses.items()},
                          "pct_minus_v2": an.paired_dates(losses["pct"], losses["v2_incumbent"], 20261007),
                          "ens_minus_v2": an.paired_dates(losses["ens_v2_pct"], losses["v2_incumbent"], 20261007)}
    return result


def replay(out: Path) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import neuroforecast_v3_pct as pct

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    proto = v3d.protocol()
    loaded = v3d.load(out)
    cases, x, twin = loaded["cases"], loaded["features"], loaded["twin"]
    batches = pct.build_batches(v3d.load_wells(out), cases, x, twin)
    k, dates, train, test = next(iter(v3d.outer_splits(cases)))
    train_dates = sorted(set(cases.date.iloc[train]))
    on_train = cases.date.isin(train_dates).to_numpy()
    purge = set(cases.identity[on_train]) - set(cases.identity.iloc[train])
    spec = proto["candidates"]["pct"]
    result = pct.fit_predict([batches[d] for d in train_dates], [batches[d] for d in sorted(set(cases.date.iloc[test]))], purge,
                             pct.Config(**spec["params"]), [proto["seed"] + s for s in range(spec["seeds_outer"])], torch.device("cuda"))
    replayed = np.array([result[int(p)][1] for p in test])
    saved = pd.read_csv(out / "predictions/pct.csv")
    saved = saved.loc[(saved.level == "outer") & (saved.outer == k)].set_index("position").loc[test, "pred"].to_numpy()
    # The saved CSV holds float32 values written at float32 round-trip precision, so compare in float32.
    return {"outer_fold": int(k), "cases": int(len(test)), "max_abs_difference_vs_csv": float(np.abs(replayed - saved).max()),
            "identical_in_float32": bool(np.array_equal(replayed.astype(np.float32), saved.astype(np.float32)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    run.verify_lock(args.out)
    loaded = v3d.load(args.out)
    cases, features, twin = loaded["cases"], loaded["features"], loaded["twin"]
    seed = v3d.protocol()["seed"]
    result = {"label": "post hoc; not part of the locked v3 protocol",
              "ablations_vs_pct": ablations(args.out, cases, seed),
              "cohorts": cohorts(args.out, cases),
              "twin_baseline": twin_baseline(args.out, cases, features, twin, data.protocol())}
    if not args.skip_replay:
        result["replay"] = replay(args.out)
    elif (args.out / "analysis/posthoc.json").exists():
        previous = json.loads((args.out / "analysis/posthoc.json").read_text(encoding="utf-8"))
        if "replay" in previous:
            result["replay"] = previous["replay"]   # keep the earlier replay result rather than drop it
    v3d.intake.write_json(args.out / "analysis/posthoc.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "twin_baseline"}, indent=1, default=str)[:4000])
    tb = result["twin_baseline"]
    wins = sum(v["pct_twin"] < v["extra_trees_per_target"] for v in tb.values())
    print(f"twin: PCT beats per-target Extra Trees on {wins} of {len(tb)} targets")


if __name__ == "__main__":
    main()
