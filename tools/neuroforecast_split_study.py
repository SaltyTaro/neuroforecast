"""NeuroForecast split-optimism study (protocol: experiments/neuroforecast_split_study_protocol.json).

How much do weaker validation designs overstate day-12 forecast accuracy and interval coverage, compared with the
batch-and-chemical design a lab meets in use? Five models, four designs, five fold repetitions, on the 41 public
batches of the validation harness (tools/neuroforecast_harness.py).

    python -X utf8 -B tools/neuroforecast_split_study.py run --out E:/kaggle/AI4S/experiments/neuroforecast_split_study
    python -X utf8 -B tools/neuroforecast_split_study.py analyze --out <dir>     # recompute summary.json from oof.csv.gz
    python -X utf8 -B tools/neuroforecast_split_study.py run --smoke --out <scratch dir>   # 1 repetition, 2 folds; no lock

`run` locks the SHA-256 of the protocol, this script and the harness on its first call and refuses to run if any of
them changes afterwards.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_harness as h  # noqa: E402

PROTOCOL = h.ROOT / "experiments/neuroforecast_split_study_protocol.json"
LOCKED = [PROTOCOL, Path(__file__).resolve(), Path(h.__file__).resolve()]
MODELS = ("carry_forward", "calibrated_day7", "ridge", "lightgbm", "extra_trees_v2")
LEARNED = ("calibrated_day7", "ridge", "lightgbm", "extra_trees_v2")
LEVELS = (0.8, 0.9)
REPETITIONS = 5
PRIMARY = "extra_trees_v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_lock(out: Path) -> dict:
    current = {p.relative_to(h.ROOT).as_posix(): sha256(p) for p in LOCKED}
    path = out / "lock.json"
    if path.exists():
        lock = json.loads(path.read_text(encoding="utf-8"))
        if lock["sha256"] != current:
            raise SystemExit(f"Locked files changed since {lock['locked_utc']}: refusing to run.\nlocked {lock['sha256']}\nnow    {current}")
        return lock
    lock = {"locked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "sha256": current}
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    return lock


# --------------------------------------------------------------------------- models

def make(name: str):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    impute = SimpleImputer(strategy="median", keep_empty_features=True)
    if name == "calibrated_day7":
        return IsotonicRegression(out_of_bounds="clip")
    if name == "ridge":
        return make_pipeline(impute, StandardScaler(), Ridge(alpha=30.0))
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return make_pipeline(impute, LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                                   objective="l1", random_state=h.SEED, n_jobs=1, verbose=-1))
    if name == "extra_trees_v2":
        return make_pipeline(impute, ExtraTreesRegressor(n_estimators=256, max_depth=12, min_samples_leaf=4, max_features=1.0,
                                                         random_state=h.SEED, n_jobs=1))
    raise KeyError(name)


RESIDUAL_TARGET = {"ridge", "lightgbm"}


def fit_predict(name: str, x: np.ndarray, day7: np.ndarray, y: np.ndarray, fit: np.ndarray, test: np.ndarray) -> np.ndarray:
    if name == "carry_forward":
        return day7[test]
    if name == "calibrated_day7":
        return make(name).fit(day7[fit], y[fit]).predict(day7[test])
    target = y[fit] - day7[fit] if name in RESIDUAL_TARGET else y[fit]
    pred = make(name).fit(x[fit], target).predict(x[test])
    return pred + day7[test] if name in RESIDUAL_TARGET else pred


# --------------------------------------------------------------------------- run

def run(out: Path, smoke: bool = False) -> Path:
    if not smoke:
        check_lock(out)
    out.mkdir(parents=True, exist_ok=True)
    bench = h.load_benchmark()
    cases, x = bench.cases, bench.features[h.SMALL_COLUMNS].to_numpy(dtype=float)
    usable = cases.usable.to_numpy(dtype=bool)
    day7, y = cases.day7.to_numpy(), cases.target.to_numpy()
    repetitions = 1 if smoke else REPETITIONS
    # LightGBM 4.6 names numpy columns itself, so scikit-learn 1.9 warns at predict time; the predictions are unaffected.
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    frames = []
    with threadpool_limits(limits=1):
        for design in h.DESIGNS:
            for rep in range(repetitions):
                fold = h.fold_assignment(cases, design, rep)
                pred = {m: np.full(len(cases), np.nan) for m in MODELS}
                half = {(m, lv): np.full(len(cases), np.nan) for m in MODELS for lv in LEVELS}
                folds = range(2) if smoke else range(h.FOLDS)
                for k in folds:
                    train, test = h.split(cases, design, fold, k)
                    fit = train[usable[train]]
                    residuals = {m: [] for m in MODELS}
                    for _, itrain, itest in h.inner_splits(cases, design, train, rep, k):
                        ifit, iscore = itrain[usable[itrain]], itest[usable[itest]]
                        for m in MODELS:
                            residuals[m].append(np.abs(fit_predict(m, x, day7, y, ifit, iscore) - y[iscore]))
                    for m in MODELS:
                        pred[m][test] = fit_predict(m, x, day7, y, fit, test)
                        scores = np.concatenate(residuals[m])
                        for lv in LEVELS:
                            half[(m, lv)][test] = h.conformal_quantile(scores, lv)
                    print(f"{design:<16} rep {rep} fold {k}: trained on {len(fit)}, tested {len(test)}", flush=True)
                frame = cases.loc[:, ["case_id", "date", "identity", "dose", "target", "usable"]].assign(design=design, repetition=rep)
                for m in MODELS:
                    frame[f"pred__{m}"] = pred[m]
                    for lv in LEVELS:
                        frame[f"half{int(lv * 100)}__{m}"] = half[(m, lv)]
                frames.append(frame.loc[usable & np.isfinite(pred[PRIMARY])])
    table = pd.concat(frames, ignore_index=True)
    path = out / "oof.csv.gz"
    table.to_csv(path, index=False, lineterminator="\n", compression={"method": "gzip", "mtime": 0})
    return path


# --------------------------------------------------------------------------- analysis

def batch_losses(frame: pd.DataFrame, model: str) -> pd.Series:
    """Per-batch date-macro loss, averaged over repetitions."""
    per_rep = [h.date_losses(g.assign(p=g[f"pred__{model}"]), "p") for _, g in frame.groupby("repetition")]
    return pd.concat(per_rep, axis=1).mean(axis=1)


def apparent_coverage(frame: pd.DataFrame, model: str, level: float) -> dict:
    stats = []
    for _, g in frame.groupby("repetition"):
        w = g[f"half{int(level * 100)}__{model}"]
        stats.append(h.coverage(g.assign(low=g[f"pred__{model}"] - w, high=g[f"pred__{model}"] + w), level))
    return {k: float(np.mean([s[k] for s in stats])) for k in ["pooled", "batch_mean", "worst_fifth", "mean_width"]}


def transfer_coverage(source: pd.DataFrame, target: pd.DataFrame, model: str, level: float) -> dict:
    pooled, worst = [], []
    for rep, g in target.groupby("repetition"):
        s = source.loc[source.repetition == rep]
        q = h.conformal_quantile(np.abs(s[f"pred__{model}"] - s.target), level)
        hit = (g[f"pred__{model}"] - g.target).abs() <= q
        pooled.append(float(hit.mean()))
        worst.append(h.worst_fifth(hit.groupby(g.date).mean()))
    return {"pooled": float(np.mean(pooled)), "worst_fifth": float(np.mean(worst))}


def analyze(out: Path) -> dict:
    table = pd.read_csv(out / "oof.csv.gz", dtype={"identity": str})
    by = {d: table.loc[table.design == d] for d in h.DESIGNS}
    reference = by["batch_chemical"]
    losses = {(d, m): batch_losses(by[d], m) for d in h.DESIGNS for m in MODELS}
    result = {"protocol_sha256": sha256(PROTOCOL), "status": "development study on 41 batches; not a sealed test",
              "conditions": int(reference.loc[reference.repetition == 0].shape[0]), "batches": int(reference.date.nunique()),
              "repetitions": int(table.repetition.nunique()), "designs": {}}
    for d in h.DESIGNS:
        entry = {"date_macro_mae": {}, "optimism": {}, "difference_vs_batch_chemical": {}, "apparent_coverage": {}, "transfer_coverage": {}}
        for m in MODELS:
            mae, ref = float(losses[(d, m)].mean()), float(losses[("batch_chemical", m)].mean())
            entry["date_macro_mae"][m] = mae
            entry["optimism"][m] = 1 - mae / ref
            if d != "batch_chemical":
                entry["difference_vs_batch_chemical"][m] = h.paired(losses[(d, m)], losses[("batch_chemical", m)])
            entry["apparent_coverage"][m] = {str(lv): apparent_coverage(by[d], m, lv) for lv in LEVELS}
            entry["transfer_coverage"][m] = {str(lv): transfer_coverage(by[d], reference, m, lv) for lv in LEVELS}
        order = sorted(MODELS, key=lambda m: entry["date_macro_mae"][m])
        entry["ranking"] = order
        entry["best_model"] = order[0]
        result["designs"][d] = entry
    ref_order = result["designs"]["batch_chemical"]["ranking"]
    for d in h.DESIGNS:
        order = result["designs"][d]["ranking"]
        result["designs"][d]["kendall_tau_vs_batch_chemical"] = float(kendalltau([order.index(m) for m in MODELS],
                                                                                  [ref_order.index(m) for m in MODELS])[0])
    chem, rand = result["designs"]["chemical"], result["designs"]["random_condition"]
    result["hypotheses"] = {
        "H1_primary": bool(chem["optimism"][PRIMARY] >= 0.05 and chem["difference_vs_batch_chemical"][PRIMARY]["high"] < 0),
        "H2": bool(rand["optimism"][PRIMARY] >= 0.05 and rand["difference_vs_batch_chemical"][PRIMARY]["high"] < 0),
        "H3": bool(chem["transfer_coverage"][PRIMARY]["0.8"]["pooled"] <= 0.77 and rand["transfer_coverage"][PRIMARY]["0.8"]["pooled"] <= 0.77),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def render(result: dict) -> str:
    lines = [f"Split-optimism study: {result['conditions']} conditions, {result['batches']} batches, {result['repetitions']} repetitions",
             f"{'date-macro MAE':<16}" + "".join(f"{m:>16}" for m in MODELS)]
    for d, e in result["designs"].items():
        lines.append(f"{d:<16}" + "".join(f"{e['date_macro_mae'][m]:>16.3f}" for m in MODELS))
    lines.append("optimism vs batch_chemical (extra_trees_v2): " + ", ".join(
        f"{d} {e['optimism'][PRIMARY]:.1%}" for d, e in result["designs"].items() if d != "batch_chemical"))
    lines.append("transfer coverage at 0.80 on batch_chemical residuals (extra_trees_v2): " + ", ".join(
        f"{d} {e['transfer_coverage'][PRIMARY]['0.8']['pooled']:.3f}" for d, e in result["designs"].items()))
    lines.append("best model per design: " + ", ".join(f"{d} {e['best_model']}" for d, e in result["designs"].items()))
    lines.append("hypotheses: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in result["hypotheses"].items()))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["run", "analyze"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="One repetition, two folds per design; no lock (for testing)")
    args = parser.parse_args()
    if args.stage == "run":
        run(args.out, smoke=args.smoke)
    if not args.smoke:
        print(render(analyze(args.out)))


if __name__ == "__main__":
    main()
