"""NeuroForecast v3 study runner (protocol: experiments/neuroforecast_v3_protocol.json).

Stages
  tabular   CPU candidates (trivial rules, the v2 recipe, Extra Trees, ridge, LightGBM, MLP)
  tabpfn    TabPFN regressors (GPU)
  pct       plate-context transformer configurations and ablations (GPU)

Every candidate is fitted on exactly the same splits: 9 interleaved outer date folds; inside each outer
training set, 4 interleaved inner date folds. Both levels purge the held-out chemicals from training.
Inner out-of-fold predictions feed nested selection and calibration; outer predictions are scored once
in `neuroforecast_v3_analyze.py`. Outputs go to E:/kaggle/AI4S/experiments/neuroforecast_v3/predictions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_representation as rep  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402
import neuroforecast_v3_data as v3d  # noqa: E402

OUT = v3d.OUT
LOCKED_CODE = ["neuroforecast_v3_data.py", "neuroforecast_v3_pct.py", "neuroforecast_v3_run.py", "neuroforecast_v3_analyze.py",
               "neuroforecast_v2.py", "external_nfa_intake.py", "neuroforecast_data.py", "neuroforecast_gate.py",
               "neuroforecast_representation.py", "neuroforecast_benchmark.py"]


def protocol() -> dict:
    return v3d.protocol()


def code_hashes() -> dict:
    tools = Path(__file__).resolve().parent
    return {name: data.digest(tools / name) for name in LOCKED_CODE}


def write_lock(out: Path) -> None:
    from datetime import datetime, timezone

    path = out / "lock_v3.json"
    if path.exists():
        raise ValueError("A v3 lock exists; preserve it")
    existing = sorted((out / "predictions").glob("*.csv")) if (out / "predictions").exists() else []
    if existing:
        raise ValueError(f"Predictions exist before the lock: {[p.name for p in existing]}")
    if "SET AT LOCK" in v3d.PROTOCOL.read_text(encoding="utf-8"):
        raise ValueError("The protocol still has unset fields")
    lock = {"locked_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": data.digest(v3d.PROTOCOL),
            "prepared_manifest_sha256": data.digest(out / "prepared/manifest.json"), "code": code_hashes(),
            "outer_predictions_before_lock": 0}
    v3d.intake.write_json(path, lock)
    print(json.dumps(lock, indent=2))


def verify_lock(out: Path) -> None:
    lock = json.loads((out / "lock_v3.json").read_text(encoding="utf-8"))
    if data.digest(v3d.PROTOCOL) != lock["protocol_sha256"]:
        raise ValueError("The protocol changed after the lock")
    if data.digest(out / "prepared/manifest.json") != lock["prepared_manifest_sha256"]:
        raise ValueError("Prepared data changed after the lock")
    changed = [k for k, v in code_hashes().items() if lock["code"].get(k) != v]
    if changed:
        raise ValueError(f"Locked code changed: {changed}")


CHEMICAL = ["chem_max_abs_d7", "chem_d7_at_top_dose", "chem_slope_d7", "chem_dose_rank", "chem_mean_change_d5_d7",
            "chem_d7_next_dose", "chem_max_abs_nae_contrast_d7"]


def chemical_context(features: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Dose-series descriptors from the same batch and chemical, days 5 and 7 only (no label, no later day)."""
    frame = pd.DataFrame({"date": cases.date.to_numpy(), "identity": cases.identity.to_numpy(),
                          "dose": cases.dose.to_numpy(dtype=float), "d5": features.normalized_ns_d5.to_numpy(),
                          "d7": features.normalized_ns_d7.to_numpy(), "nae": features["contrast__nAE__d7"].to_numpy()})
    out = pd.DataFrame(index=frame.index, columns=CHEMICAL, dtype=float)
    for _, g in frame.groupby(["date", "identity"]):
        g = g.sort_values("dose")
        logd = np.log10(g.dose.to_numpy())
        d7 = g.d7.to_numpy()
        slope = float(np.polyfit(logd, d7, 1)[0]) if len(g) > 1 and np.ptp(logd) > 0 and np.isfinite(d7).all() else np.nan
        out.loc[g.index, "chem_max_abs_d7"] = np.nanmax(np.abs(d7))
        out.loc[g.index, "chem_d7_at_top_dose"] = d7[-1]
        out.loc[g.index, "chem_slope_d7"] = slope
        out.loc[g.index, "chem_dose_rank"] = np.arange(len(g)) / max(len(g) - 1, 1)
        out.loc[g.index, "chem_mean_change_d5_d7"] = np.nanmean(d7 - g.d5.to_numpy())
        out.loc[g.index, "chem_d7_next_dose"] = np.append(d7[1:], np.nan)
        out.loc[g.index, "chem_max_abs_nae_contrast_d7"] = np.nanmax(np.abs(g.nae.to_numpy())) if np.isfinite(g.nae).any() else np.nan
    return out.astype(float)


def with_chemical_context(features: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    return features.join(chemical_context(features, cases))


def column_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    base = features.drop(columns=v3d.CONTEXT + [c for c in CHEMICAL if c in features])
    sets = v2.column_sets(base)
    representation = [c for c in base.columns if c not in gate.RELIABILITY and c not in v2.BATCH]
    full = representation + gate.RELIABILITY + v2.BATCH
    return {
        "small": sets["small"],
        "small_ctx": sets["small"] + v3d.CONTEXT,
        "full_ctx": full + v3d.CONTEXT,
        "small_chem": sets["small"] + CHEMICAL,
        "small_chem_ctx": sets["small"] + CHEMICAL + v3d.CONTEXT,
        "sigma": sets["sigma"] + v3d.CONTEXT,
    }


# --------------------------------------------------------------------------- tabular candidates

def make_model(spec: dict, cfg: dict):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    kind = spec["kind"]
    impute = SimpleImputer(strategy="median", keep_empty_features=True)
    if kind == "extra_trees":
        return make_pipeline(impute, ExtraTreesRegressor(**{**cfg["extra_trees"], "n_jobs": 1}, random_state=cfg["seed"]))
    if kind == "ridge":
        return make_pipeline(impute, StandardScaler(), Ridge(alpha=spec["alpha"]))
    if kind == "lightgbm":
        from lightgbm import LGBMRegressor
        return make_pipeline(impute, LGBMRegressor(**spec["params"], random_state=cfg["seed"], n_jobs=1, verbose=-1))
    if kind == "mlp":
        return make_pipeline(impute, StandardScaler(), MLPRegressor(**spec["params"], random_state=cfg["seed"]))
    raise KeyError(kind)


def fit_predict_tabular(spec: dict, cfg: dict, x: pd.DataFrame, cases: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    d7 = x.normalized_ns_d7.to_numpy()
    if spec["kind"] == "persistence":
        return d7[test]
    if spec["kind"] == "control":
        return np.zeros(len(test))
    usable = cases.usable.to_numpy(dtype=bool)
    fit_idx = train[usable[train]]
    y = cases.target.to_numpy()[fit_idx]
    residual = spec.get("residual", False)
    if residual:
        y = y - d7[fit_idx]
    cols = spec["columns_resolved"]
    model = make_model(spec, cfg)
    model.fit(x.iloc[fit_idx][cols], y)
    pred = model.predict(x.iloc[test][cols])
    return pred + d7[test] if residual else pred


def split_plan(cases: pd.DataFrame) -> list[dict]:
    plan = []
    for k, dates, train, test in v3d.outer_splits(cases):
        plan.append({"outer": k, "level": "outer", "inner": -1, "train": train, "test": test})
        for j, _, itrain, itest in v3d.inner_splits(cases, train):
            plan.append({"outer": k, "level": "inner", "inner": j, "train": itrain, "test": itest})
    return plan


def run_tabular_job(name, spec, cfg, x, cases, split):
    with threadpool_limits(limits=1):
        pred = fit_predict_tabular(spec, cfg, x, cases, split["train"], split["test"])
    return pd.DataFrame({"outer": split["outer"], "level": split["level"], "inner": split["inner"],
                         "position": split["test"], "candidate": name, "pred": pred})


def stage_tabular(out: Path, jobs: int, only: list[str] | None) -> None:
    proto = protocol()
    cfg = data.protocol()
    loaded = v3d.load(out)
    cases = loaded["cases"]
    x = with_chemical_context(loaded["features"], cases)
    sets = column_sets(x)
    specs = {n: s for n, s in proto["candidates"].items() if s["kind"] in ("persistence", "control", "extra_trees", "ridge", "lightgbm", "mlp")}
    if only:
        specs = {n: s for n, s in specs.items() if n in only}
    for s in specs.values():
        s["columns_resolved"] = sets.get(s.get("columns", "small"), [])
    plan = split_plan(cases)
    started = time.monotonic()
    frames = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
        delayed(run_tabular_job)(name, spec, cfg, x, cases, split) for name, spec in specs.items() for split in plan)
    target = out / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for name in specs:
        part = pd.concat([f for f in frames if f.candidate.iat[0] == name], ignore_index=True)
        part.to_csv(target / f"{name}.csv", index=False)
    print(json.dumps({"stage": "tabular", "candidates": list(specs), "splits": len(plan), "seconds": time.monotonic() - started}))


# --------------------------------------------------------------------------- TabPFN

def stage_tabpfn(out: Path, only: list[str] | None) -> None:
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    proto = protocol()
    loaded = v3d.load(out)
    cases = loaded["cases"]
    x = with_chemical_context(loaded["features"], cases)
    sets = column_sets(x)
    specs = {n: s for n, s in proto["candidates"].items() if s["kind"] == "tabpfn" and (not only or n in only)}
    usable = cases.usable.to_numpy(dtype=bool)
    d7 = x.normalized_ns_d7.to_numpy()
    plan = split_plan(cases)
    target = out / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for name, spec in specs.items():
        started = time.monotonic()
        cols = sets[spec["columns"]]
        rows = []
        for split in plan:
            fit_idx = split["train"][usable[split["train"]]]
            y = cases.target.to_numpy()[fit_idx] - (d7[fit_idx] if spec.get("residual") else 0)
            torch.manual_seed(proto["seed"])
            # TabPFN v2 weights (Hollmann et al., Nature 2025): downloadable without a licence login, unlike later versions.
            model = TabPFNRegressor.create_default_for_version(ModelVersion(spec["model_version"]), **spec.get("params", {}),
                                                               random_state=proto["seed"], device="cuda")
            model.fit(x.iloc[fit_idx][cols].to_numpy(dtype=np.float32), y)
            pred = model.predict(x.iloc[split["test"]][cols].to_numpy(dtype=np.float32))
            pred = pred + (d7[split["test"]] if spec.get("residual") else 0)
            rows.append(pd.DataFrame({"outer": split["outer"], "level": split["level"], "inner": split["inner"],
                                      "position": split["test"], "candidate": name, "pred": pred}))
        pd.concat(rows, ignore_index=True).to_csv(target / f"{name}.csv", index=False)
        print(json.dumps({"stage": "tabpfn", "candidate": name, "seconds": time.monotonic() - started}), flush=True)


# --------------------------------------------------------------------------- PCT

def stage_pct(out: Path, only: list[str] | None, levels: list[str]) -> None:
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA starts: deterministic cuBLAS
    import torch
    import neuroforecast_v3_pct as pct

    proto = protocol()
    torch.use_deterministic_algorithms(True)   # bit-identical reruns on the same GPU and software
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loaded = v3d.load(out)
    x, cases = loaded["features"], loaded["cases"]
    twin = loaded["twin"]
    wells = v3d.load_wells(out)
    batches = pct.build_batches(wells, cases, x, twin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = {n: s for n, s in proto["candidates"].items() if s["kind"] == "pct" and (not only or n in only)}
    plan = [s for s in split_plan(cases) if s["level"] in levels]
    target = out / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for name, spec in specs.items():
        started = time.monotonic()
        cfg = pct.Config(**spec["params"])
        rows = []
        for split in plan:
            if split["level"] == "inner" and not spec["seeds_inner"]:
                continue  # ablations are reported on the outer folds only
            train_dates = sorted(set(cases.date.iloc[split["train"]]))
            test_dates = sorted(set(cases.date.iloc[split["test"]]))
            # Labels are allowed only for conditions in this split's training positions: every other chemical on a
            # training date (held out at this level or at the outer level) is purged, exactly as for tabular models.
            on_train_dates = cases.date.isin(train_dates).to_numpy()
            purge = set(cases.identity[on_train_dates]) - set(cases.identity.iloc[split["train"]])
            seeds = spec["seeds_outer"] if split["level"] == "outer" else spec["seeds_inner"]
            result = pct.fit_predict([batches[d] for d in train_dates], [batches[d] for d in test_dates], purge, cfg,
                                     [proto["seed"] + s for s in range(seeds)], device)
            test = split["test"]
            q = np.stack([result[int(p)] for p in test])
            frame = pd.DataFrame({"outer": split["outer"], "level": split["level"], "inner": split["inner"],
                                  "position": test, "candidate": name, "pred": q[:, 1], "q10": q[:, 0],
                                  "q90": q[:, 2], "seed_sd": q[:, 3]})
            if split["level"] == "outer":
                frame = frame.join(pd.DataFrame(q[:, 4:], columns=[f"twin__{c}" for c in twin.columns]))
            rows.append(frame)
            print(json.dumps({"candidate": name, "outer": split["outer"], "level": split["level"], "inner": split["inner"],
                              "elapsed": round(time.monotonic() - started, 1)}), flush=True)
        pd.concat(rows, ignore_index=True).to_csv(target / f"{name}.csv", index=False)
        print(json.dumps({"stage": "pct", "candidate": name, "seconds": time.monotonic() - started}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["lock", "tabular", "tabpfn", "pct"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--levels", nargs="*", default=["inner", "outer"])
    args = parser.parse_args()
    if args.stage == "lock":
        write_lock(args.out)
        return
    verify_lock(args.out)
    if args.stage == "tabular":
        stage_tabular(args.out, args.jobs, args.only)
    elif args.stage == "tabpfn":
        stage_tabpfn(args.out, args.only)
    else:
        stage_pct(args.out, args.only, args.levels)


if __name__ == "__main__":
    main()
