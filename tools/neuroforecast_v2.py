"""NeuroForecast v2: develop on the 17 known dates, then score 28 earlier EPA dates exactly once.

Stages (see experiments/neuroforecast_v2_protocol.json):
  prepare   build development cases (with targets) and external inputs (days 5 and 7 only; no targets)
  develop   nested leave-one-date-out evaluation, component selection, final fits, hashed lock file
  external  requires --unseal and an unchanged lock; builds external targets and scores Arms A and B once
  audit     replays saved models and checks that no external row reached any fit

Frozen v1 code is imported unchanged. All outputs go to E:.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_nfa_intake as intake  # noqa: E402
import neuroforecast_benchmark as bench  # noqa: E402
import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_representation as rep  # noqa: E402

PROTOCOL = data.ROOT / "experiments/neuroforecast_v2_protocol.json"
OUT = Path("E:/kaggle/AI4S/experiments/neuroforecast_external_v2")
DEV_DATES = [20160720, 20160803, 20160907, 20160921, 20161013, 20161109, 20170201, 20170222, 20170412,
             20170628, 20170712, 20170809, 20170816, 20170913, 20170920, 20171004, 20171011]
USABLE_MIN = 5.0
LAMBDAS = [0.25, 0.5, 0.75, 1.0]
BLEND_GRID = [(a, b) for a in [-1.0, -0.5, 0.0, 0.5, 1.0] for b in [-1.0, -0.5, 0.0, 0.5, 1.0]]
BATCH = ["batch_sd_d7", "batch_changed_frac_d7", "batch_ctrl_median_d7"]
TRIVIAL = ["persistence_day7", "control_reference"]
LEARNED = ["v1_refit", "small_refit", "anchored_residual", "anchored_residual_batch", "batch_blend"]
FORECASTERS = TRIVIAL + LEARNED
TRUST = ["sigma_v2", "extremity", "predicted_change", "day7_magnitude", "low_control_activity"]
WARNING = ["highest_dose", "day7_magnitude_random_ties", "abs_v1_refit", "abs_forecast", "abs_predicted_change"]
BUDGET = 0.2
RETAIN = 0.8
LEVELS = [0.8, 0.9]
D_H = [1, 2, 3, 5]
D_C = [0.25, 0.5, 0.75, 1.0]
D_AGREEMENT = 0.95


# --------------------------------------------------------------------------- data

def v2_config() -> dict:
    cfg = dict(data.protocol())
    cfg["feature_readouts"] = [f for f in cfg["feature_readouts"] if f not in intake.NOT_IN_RELEASE] + ["mi"]
    return cfg


def release_frame() -> pd.DataFrame:
    release = intake.read_release()
    frame, _ = intake.to_frozen_schema(release, intake.crosswalk(release))
    return frame


def batch_covariates(inputs: pd.DataFrame, early: pd.DataFrame) -> pd.DataFrame:
    """Per-date summaries of day-7 inputs only: spread and share of changed conditions, and control activity."""
    dates = inputs.index.get_level_values("date")
    d7 = inputs.normalized_ns_d7
    table = pd.DataFrame({
        "batch_sd_d7": d7.groupby(dates).std(ddof=1),
        "batch_changed_frac_d7": (d7.abs() >= 0.5).groupby(dates).mean(),
    })
    controls = early.loc[(early.DIV == 7) & (early.dose == 0)].groupby(["date", "Plate.SN"])["ns.n"].median()
    table["batch_ctrl_median_d7"] = np.arcsinh(controls).groupby(level=0).median()
    return table.reindex(dates).set_axis(inputs.index)


def features_from(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Everything a model may see, built from day-5 and day-7 rows only."""
    early = raw.loc[raw.DIV.isin([5, 7])]
    if set(early.DIV) - {5, 7}:
        raise ValueError("Only day-5 and day-7 rows may reach the feature builder")
    inputs = data.build_inputs(early, cfg)
    rel = gate.reliability_indicators(early, cfg).reindex(inputs.index)
    rel.pop("rel__plates_spanned")
    transformed = rep.representation(inputs, cfg)
    return transformed.join(rel[gate.RELIABILITY]).join(batch_covariates(inputs, early))


def column_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    representation = [c for c in features.columns if c not in gate.RELIABILITY and c not in BATCH]
    v1 = [c for c in representation if not c.endswith("d5")]
    small = rep.selected_columns("extra_trees_activity_coordination", features[representation])
    return {"v1": v1, "small": small, "batch": v1 + BATCH, "sigma": v1 + gate.RELIABILITY + BATCH}


def outcome_quality(raw: pd.DataFrame, cfg: dict) -> pd.Series:
    return gate.outcome_quality(raw, cfg)


def sample_ids(raw: pd.DataFrame) -> pd.Series:
    """EPA sample ID per case (the D decision unit)."""
    treated = raw.loc[(raw.dose > 0) & raw.eligible_identity]
    return treated.groupby(data.KEY).spid.agg(lambda s: sorted(set(s))[0])


# --------------------------------------------------------------------------- models

def et(cfg: dict):
    return gate.single_threaded(bench.fitted_model("extra_trees_v2", cfg))


def fit(frame: pd.DataFrame, columns: list[str], y: np.ndarray, cfg: dict):
    model = et(cfg)
    model.fit(frame[columns], y)
    return model


def blend_weight(frame: pd.DataFrame, a: float, b: float) -> np.ndarray:
    return np.clip(a + b * frame.batch_sd_d7.fillna(0).to_numpy(), 0, 1)


def macro(cases: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    frame = cases.assign(prediction=prediction)
    if mask is not None:
        frame = frame.loc[mask]
    return gate.macro_mae(frame)


def purged_folds(cases: pd.DataFrame, positions: np.ndarray):
    """Leave one date out among `positions`; purge the test date's identities from training."""
    subset = cases.iloc[positions]
    for date in sorted(subset.date.unique()):
        test = positions[subset.date.to_numpy() == date]
        chemicals = set(cases.iloc[test].identity)
        train = positions[(subset.date.to_numpy() != date) & ~subset.identity.isin(chemicals).to_numpy()]
        yield int(date), train, test


def raw_predictions(features, cases, y, usable, cols, train, test, cfg) -> dict[str, np.ndarray]:
    """Fit the four base learners on usable training cases; predict the test cases."""
    fit_idx = train[usable[train]]
    x_train, x_test = features.iloc[fit_idx], features.iloc[test]
    d7_train = x_train.normalized_ns_d7.to_numpy()
    out = {}
    out["v1_refit"] = fit(x_train, cols["v1"], y[fit_idx], cfg).predict(x_test[cols["v1"]])
    out["small_refit"] = fit(x_train, cols["small"], y[fit_idx], cfg).predict(x_test[cols["small"]])
    out["g"] = fit(x_train, cols["v1"], y[fit_idx] - d7_train, cfg).predict(x_test[cols["v1"]])
    out["g_batch"] = fit(x_train, cols["batch"], y[fit_idx] - d7_train, cfg).predict(x_test[cols["batch"]])
    return out


def combine(base: dict[str, np.ndarray], x: pd.DataFrame, params: dict) -> dict[str, np.ndarray]:
    d7 = x.normalized_ns_d7.to_numpy()
    w = blend_weight(x, *params["blend"])
    return {
        "persistence_day7": d7,
        "control_reference": np.zeros(len(x)),
        "v1_refit": base["v1_refit"],
        "small_refit": base["small_refit"],
        "anchored_residual": d7 + params["lambda"] * base["g"],
        "anchored_residual_batch": d7 + params["lambda_batch"] * base["g_batch"],
        "batch_blend": w * base["v1_refit"] + (1 - w) * d7,
    }


def tune(base_oof: dict[str, np.ndarray], x: pd.DataFrame, cases: pd.DataFrame, mask: np.ndarray) -> dict:
    """Choose lambda, lambda_batch and (a, b) by date-macro MAE on out-of-fold predictions (usable targets)."""
    d7 = x.normalized_ns_d7.to_numpy()
    lam = min(LAMBDAS, key=lambda v: (macro(cases, d7 + v * base_oof["g"], mask), v))
    lam_b = min(LAMBDAS, key=lambda v: (macro(cases, d7 + v * base_oof["g_batch"], mask), v))
    blend = min(BLEND_GRID, key=lambda ab: (macro(cases, blend_weight(x, *ab) * base_oof["v1_refit"] + (1 - blend_weight(x, *ab)) * d7, mask), ab))
    return {"lambda": lam, "lambda_batch": lam_b, "blend": blend}


def oof_base(features, cases, y, usable, cols, positions, cfg):
    """Out-of-fold base predictions over `positions` by purged leave-one-date-out."""
    oof = {k: np.full(len(cases), np.nan) for k in ["v1_refit", "small_refit", "g", "g_batch"]}
    for _, train, test in purged_folds(cases, positions):
        for name, values in raw_predictions(features, cases, y, usable, cols, train, test, cfg).items():
            oof[name][test] = values
    return {k: v[positions] for k, v in oof.items()}


def outer_fold(date, train, test, features, cases, y, usable, cols, cfg):
    """One outer fold: inner tuning and nested selection on `train`, then predictions for `test`."""
    with threadpool_limits(limits=1):
        inner = oof_base(features, cases, y, usable, cols, train, cfg)
        x_inner, c_inner, m_inner = features.iloc[train], cases.iloc[train], usable[train]
        params = tune(inner, x_inner, c_inner, m_inner)
        inner_pred = combine(inner, x_inner, params)
        inner_scores = {name: macro(c_inner, pred, m_inner) for name, pred in inner_pred.items()}
        base = raw_predictions(features, cases, y, usable, cols, train, test, cfg)
        outer_pred = combine(base, features.iloc[test], params)
    selected = min(FORECASTERS, key=lambda n: (inner_scores[n], FORECASTERS.index(n)))
    return {"date": date, "train": train, "test": test, "params": params, "inner_scores": inner_scores,
            "selected": selected, "outer": outer_pred, "inner": inner_pred, "base": base}


# --------------------------------------------------------------------------- metrics

def trust_scores(name: str, frame: pd.DataFrame, forecast: np.ndarray, sigma: np.ndarray | None) -> np.ndarray:
    d7 = frame.normalized_ns_d7.to_numpy()
    if name == "sigma_v2":
        return sigma
    if name == "extremity":
        return np.abs(forecast)
    if name == "predicted_change":
        return np.abs(forecast - d7)
    if name == "day7_magnitude":
        return np.abs(d7)
    if name == "low_control_activity":
        return -frame["rel__plate_ctrl_median_ns_d7"].to_numpy()
    raise KeyError(name)


def risk_coverage(score: np.ndarray, error: np.ndarray) -> float:
    """Mean retained-case MAE over retention 50%..100%, retaining the lowest scores first (stable order)."""
    order = np.argsort(score, kind="stable")
    ordered = error[order]
    n = len(error)
    values = [ordered[: math.ceil(r / 100 * n)].mean() for r in range(50, 101)]
    return float(np.mean(values))


def expected_detections(frame: pd.DataFrame, score: np.ndarray, budget: float = BUDGET) -> dict:
    """Early-quiet detections at the budget, ties broken at random (expected value), pooled over dates."""
    work = frame.assign(score=score, positive=np.abs(frame.target) >= 1).loc[np.abs(frame.day7) < 0.5]
    detected = positives = flagged = random = 0.0
    for _, group in work.groupby("date"):
        k = math.ceil(budget * len(group))
        if not k:
            continue
        cutoff = np.sort(group.score.to_numpy())[::-1][k - 1]
        above, tied = group[group.score > cutoff], group[group.score == cutoff]
        detected += above.positive.sum() + tied.positive.sum() * (k - len(above)) / len(tied)
        positives += group.positive.sum()
        flagged += k
        random += group.positive.sum() * k / len(group)
    return {"expected_detected": float(detected), "positives": int(positives), "flagged": int(flagged),
            "recall": float(detected / positives) if positives else None, "random_expected": float(random)}


def direction_agreement(frame: pd.DataFrame, score: np.ndarray, change: np.ndarray, budget: float = BUDGET) -> dict:
    work = frame.assign(score=score, change=change, positive=np.abs(frame.target) >= 1).loc[np.abs(frame.day7) < 0.5]
    hits = []
    for _, group in work.groupby("date"):
        k = math.ceil(budget * len(group))
        top = group.sort_values("score", ascending=False, kind="stable").head(k)
        hits.append(top.loc[top.positive])
    hits = pd.concat(hits) if hits else work.iloc[:0]
    same = np.sign(hits.change) == np.sign(hits.target - hits.day7)
    return {"detections": int(len(hits)), "same_direction": int(same.sum())}


def warning_score(name: str, frame: pd.DataFrame, forecast: np.ndarray, v1_forecast: np.ndarray) -> np.ndarray:
    d7 = frame.day7.to_numpy()
    return {
        "highest_dose": frame.dose.to_numpy(dtype=float),
        "day7_magnitude_random_ties": np.abs(d7),
        "abs_v1_refit": np.abs(v1_forecast),
        "abs_forecast": np.abs(forecast),
        "abs_predicted_change": np.abs(forecast - d7),
    }[name]


def per_date(frame: pd.DataFrame, column: str) -> dict:
    return {str(k): float(v) for k, v in gate.date_losses(frame.assign(prediction=frame[column])).items()}


# --------------------------------------------------------------------------- stages

def write(path: Path, payload) -> None:
    intake.write_json(path, payload)


def code_hashes() -> dict:
    files = [Path(__file__), PROTOCOL, Path(intake.__file__), Path(data.__file__), Path(bench.__file__),
             Path(rep.__file__), Path(gate.__file__), data.PROTOCOL_PATH, rep.PROTOCOL]
    return {p.name: data.digest(p) for p in files}


def prepare(out: Path) -> None:
    target = out / "prepared"
    if (target / "manifest.json").exists():
        raise ValueError("Prepared data exist; preserve them")
    target.mkdir(parents=True, exist_ok=True)
    cfg = v2_config()
    raw = release_frame()
    dev_raw = raw.loc[raw.date.isin(DEV_DATES)]
    ext_raw = raw.loc[~raw.date.isin(DEV_DATES)]
    dev_features = features_from(dev_raw, cfg)
    dev_targets = data.build_targets(dev_raw, cfg)
    common = dev_features.index.intersection(dev_targets.index, sort=False)
    dev_features, dev_targets = dev_features.loc[common], dev_targets.loc[common]
    quality = outcome_quality(dev_raw, cfg).reindex(common)
    dev_cases = common.to_frame(index=False)
    dev_cases["case_id"] = gate.case_ids(common)
    dev_cases["target"] = dev_targets.to_numpy()
    dev_cases["day7"] = dev_features.normalized_ns_d7.to_numpy()
    dev_cases["day12_ctrl_median_ns"] = quality.to_numpy()
    dev_cases["usable"] = dev_cases.day12_ctrl_median_ns >= USABLE_MIN
    dev_cases["spid"] = sample_ids(dev_raw).reindex(common).to_numpy()
    # External: features from days 5 and 7 only. No target, no day-9/12 value is built here.
    ext_features = features_from(ext_raw.loc[ext_raw.DIV.isin([5, 7])], cfg)
    ext_cases = ext_features.index.to_frame(index=False)
    ext_cases["case_id"] = gate.case_ids(ext_features.index)
    ext_cases["day7"] = ext_features.normalized_ns_d7.to_numpy()
    ext_cases["spid"] = sample_ids(ext_raw.loc[ext_raw.DIV == 7]).reindex(ext_features.index).to_numpy()
    dev_ids = set(dev_cases.identity)
    ext_cases["primary"] = ~ext_cases.identity.isin(dev_ids)
    dev_features.reset_index(drop=True).to_csv(target / "dev_features.csv", index=False)
    dev_cases.to_csv(target / "dev_cases.csv", index=False)
    ext_features.reset_index(drop=True).to_csv(target / "ext_features.csv", index=False)
    ext_cases.to_csv(target / "ext_cases.csv", index=False)
    manifest = {
        "dev_cases": len(dev_cases), "dev_usable_cases": int(dev_cases.usable.sum()), "dev_dates": len(DEV_DATES),
        "dev_identities": int(dev_cases.identity.nunique()),
        "ext_input_cases": len(ext_cases), "ext_primary_input_cases": int(ext_cases.primary.sum()),
        "ext_dates": int(ext_cases.date.nunique()), "ext_identities": int(ext_cases.identity.nunique()),
        "external_outcomes_built": False,
        "feature_readouts": cfg["feature_readouts"],
        "files": {p.name: data.digest(p) for p in sorted(target.glob("*.csv"))},
        "code": code_hashes(),
    }
    write(target / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def load(out: Path, prefix: str):
    target = out / "prepared"
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if data.digest(target / name) != digest:
            raise ValueError(f"Prepared file changed: {name}")
    features = pd.read_csv(target / f"{prefix}_features.csv")
    cases = pd.read_csv(target / f"{prefix}_cases.csv", dtype={"identity": str, "spid": str})
    return features, cases


def bioactivity(columns: list[str], spids: set[str]) -> pd.DataFrame:
    """Read EPA per-sample calls for the given samples only: other rows are skipped at read time."""
    path = intake.RAW_DIR / "Bioactivity_bin_tbl_comp_methods.csv"
    keys = pd.read_csv(path, usecols=["spid"]).spid
    skip = {i + 1 for i, s in enumerate(keys) if s not in spids}
    table = pd.read_csv(path, usecols=["spid"] + columns, skiprows=lambda i: i in skip)
    if not set(table.spid) <= spids:
        raise ValueError("A call outside the permitted samples was read")
    return table.drop_duplicates("spid").set_index("spid")


def d_units(cases: pd.DataFrame, forecast: np.ndarray, declined: np.ndarray, flagged: np.ndarray) -> pd.DataFrame:
    work = cases.assign(abs_forecast=np.abs(forecast), declined=declined, flagged=flagged)
    return work.groupby("spid").agg(max_abs_forecast=("abs_forecast", "max"), any_declined=("declined", "any"),
                                    any_flagged=("flagged", "any"), conditions=("case_id", "size"))


def d_policy(units: pd.DataFrame, h: int, c: float) -> pd.Series:
    active = units["div7.hitsum"] >= h
    inactive = (units["div7.hitsum"] == 0) & (units.max_abs_forecast < c) & ~units.any_declined & ~units.any_flagged
    return pd.Series(np.select([active, inactive], ["active", "inactive"], "continue"), index=units.index)


def d_score(units: pd.DataFrame, calls: pd.Series, reference: pd.Series) -> dict:
    early = calls != "continue"
    agree = (calls[early] == np.where(reference[early], "active", "inactive"))
    return {"units": int(len(units)), "early": int(early.sum()), "early_share": float(early.mean()),
            "early_agreement": float(agree.mean()) if early.any() else None,
            "early_active": int((calls == "active").sum()), "early_inactive": int((calls == "inactive").sum())}


def develop(out: Path, jobs: int) -> None:
    lock_path = out / "development_lock.json"
    if lock_path.exists():
        raise ValueError("A development lock exists; preserve it")
    started = time.monotonic()
    cfg = v2_config()
    features, cases = load(out, "dev")
    cols = column_sets(features)
    y = cases.target.to_numpy()
    usable = cases.usable.to_numpy(dtype=bool)
    positions = np.arange(len(cases))
    folds = list(purged_folds(cases, positions))
    results = Parallel(n_jobs=jobs, backend="loky")(
        delayed(outer_fold)(date, train, test, features, cases, y, usable, cols, cfg) for date, train, test in folds)

    oof = {name: np.full(len(cases), np.nan) for name in FORECASTERS}
    nested = np.full(len(cases), np.nan)
    fold_log = []
    for r in results:
        for name in FORECASTERS:
            oof[name][r["test"]] = r["outer"][name]
        nested[r["test"]] = r["outer"][r["selected"]]
        fold_log.append({"date": r["date"], "selected": r["selected"], "params": r["params"],
                         "inner_scores": r["inner_scores"], "train_cases": int(len(r["train"])), "test_cases": int(len(r["test"]))})
    scored = cases.assign(**{f"pred__{k}": v for k, v in oof.items()}, pred__nested=nested)
    scored.to_csv(out / "dev_oof_predictions.csv", index=False)
    forecast_scores = {name: macro(cases, oof[name], usable) for name in FORECASTERS}
    product = min(FORECASTERS, key=lambda n: (forecast_scores[n], FORECASTERS.index(n)))
    nested_score = macro(cases, nested, usable)

    # Sigma for the product: per outer fold, fit on that fold's inner out-of-fold absolute residuals.
    sigma_oof = np.full(len(cases), np.nan)
    with threadpool_limits(limits=1):
        for r in results:
            train_fit = r["train"][usable[r["train"]]]
            inner_res = np.abs(r["inner"][product] - y[r["train"]])[usable[r["train"]]]
            model = et(cfg).fit(features.iloc[train_fit][cols["sigma"]], np.round(inner_res, gate.RESIDUAL_DECIMALS))
            sigma_oof[r["test"]] = np.maximum(model.predict(features.iloc[r["test"]][cols["sigma"]]), gate.SIGMA_FLOOR)

    u = usable
    error = np.abs(oof[product] - y)
    trust_values = {name: trust_scores(name, features, oof[product], sigma_oof) for name in TRUST}
    trust_aurc = {name: risk_coverage(trust_values[name][u], error[u]) for name in TRUST}
    trust_aurc["random_expected"] = float(error[u].mean())
    trust = min(TRUST, key=lambda n: (trust_aurc[n], TRUST.index(n)))
    threshold = float(np.quantile(trust_values[trust][u], RETAIN))

    # Intervals: scaled by sigma when sigma is the locked trust score, symmetric otherwise.
    residual = np.abs(oof[product] - y)
    if trust == "sigma_v2":
        quantiles = {str(level): gate.conformal_quantile(residual[u] / sigma_oof[u], level) for level in LEVELS}
    else:
        quantiles = {str(level): gate.conformal_quantile(residual[u], level) for level in LEVELS}

    frame = cases.assign(prediction=oof[product])
    warning = {name: expected_detections(frame.loc[u], warning_score(name, frame, oof[product], oof["v1_refit"])[u]) for name in WARNING}
    warn = max(WARNING, key=lambda n: (warning[n]["expected_detected"], -WARNING.index(n)))
    warning_direction = direction_agreement(frame.loc[u], warning_score(warn, frame, oof[product], oof["v1_refit"])[u], (oof[product] - frame.day7.to_numpy())[u])

    # Decision D on development samples that were never tested on an external date.
    ext_features, ext_cases = load(out, "ext")
    dev_spids = set(cases.spid.dropna()) - set(ext_cases.spid.dropna())
    bio = bioactivity(["div7.hitsum", "auc.hitsum"], dev_spids)
    declined = trust_values[trust] > threshold
    flagged = np.sinh(features["rel__plate_ctrl_median_ns_d7"].to_numpy()) < 1
    units = d_units(cases.loc[cases.spid.isin(dev_spids)], oof[product][cases.spid.isin(dev_spids).to_numpy()],
                    declined[cases.spid.isin(dev_spids).to_numpy()], flagged[cases.spid.isin(dev_spids).to_numpy()])
    units = units.join(bio, how="inner")
    reference = units["auc.hitsum"] >= 1
    d0_calls = pd.Series(np.where(units["div7.hitsum"] >= 1, "active", "inactive"), index=units.index)
    d_grid = {f"h={h},c={c}": d_score(units, d_policy(units, h, c), reference) for h in D_H for c in D_C}
    feasible = [(h, c) for h in D_H for c in D_C if (d_grid[f"h={h},c={c}"]["early_agreement"] or 0) >= D_AGREEMENT]
    d_locked = max(feasible, key=lambda hc: (d_grid[f"h={hc[0]},c={hc[1]}"]["early_share"], -hc[0], -hc[1])) if feasible else None

    # Final fits on all development dates for the external test (usable targets only).
    fit_idx = positions[usable]
    x = features.iloc[fit_idx]
    d7_fit = x.normalized_ns_d7.to_numpy()
    # The outer folds already hold the 17-date out-of-fold base predictions; tune the final parameters on them.
    top = {k: np.full(len(cases), np.nan) for k in ["v1_refit", "small_refit", "g", "g_batch"]}
    for r in results:
        for k in top:
            top[k][r["test"]] = r["base"][k]
    params = tune(top, features, cases, usable)
    models_dir = out / "models"
    models_dir.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        final = {
            "v1_refit": fit(x, cols["v1"], y[fit_idx], cfg),
            "small_refit": fit(x, cols["small"], y[fit_idx], cfg),
            "g": fit(x, cols["v1"], y[fit_idx] - d7_fit, cfg),
            "g_batch": fit(x, cols["batch"], y[fit_idx] - d7_fit, cfg),
            "sigma_v2": et(cfg).fit(features.iloc[fit_idx][cols["sigma"]], np.round(np.abs(oof[product] - y)[fit_idx], gate.RESIDUAL_DECIMALS)),
        }
    column_of = {"v1_refit": "v1", "small_refit": "small", "g": "v1", "g_batch": "batch", "sigma_v2": "sigma"}
    model_hashes = {}
    for name, model in final.items():
        path = models_dir / f"final_{name}.joblib"
        joblib.dump({"model": model, "columns": cols[column_of[name]]}, path, compress=3)
        model_hashes[path.name] = data.digest(path)

    development = {
        "dev_cases": int(len(cases)), "dev_usable_cases": int(u.sum()), "folds": fold_log,
        "forecast_date_macro_mae_usable": forecast_scores, "product": product, "nested_selection_estimate": nested_score,
        "nested_selected_by_fold": {str(r["date"]): r["selected"] for r in results},
        "forecast_all_cases_sensitivity": {name: macro(cases, oof[name]) for name in FORECASTERS},
        "per_date_usable": {name: per_date(scored.loc[u].assign(p=oof[name][u]), "p") for name in FORECASTERS},
        "trust_aurc": trust_aurc, "trust_locked": trust, "trust_threshold_p80": threshold,
        "retained_mae_at_lock": float(error[u & (trust_values[trust] <= threshold)].mean()),
        "all_mae": float(error[u].mean()),
        "extremity_retained_mae_same_retention": float(error[u][np.argsort(np.abs(oof[product][u]), kind="stable")[: int(round(RETAIN * u.sum()))]].mean()),
        "conformal_quantiles": quantiles, "interval_variant": "scaled" if trust == "sigma_v2" else "symmetric",
        "warning": warning, "warning_locked": warn, "warning_direction": warning_direction,
        "decision_D": {"units": int(len(units)), "reference_active": int(reference.sum()), "baseline_D0": d_score(units, d0_calls, reference),
                        "grid": d_grid, "locked": {"h": d_locked[0], "c": d_locked[1]} if d_locked else None},
        "final_params": params,
        "seconds": time.monotonic() - started,
    }
    write(out / "development_results.json", development)
    lock = {
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": data.digest(PROTOCOL), "code": code_hashes(),
        "prepared_manifest_sha256": data.digest(out / "prepared/manifest.json"),
        "development_results_sha256": data.digest(out / "development_results.json"),
        "dev_oof_sha256": data.digest(out / "dev_oof_predictions.csv"),
        "models": model_hashes,
        "locked": {"product": product, "final_params": params, "trust": trust, "trust_threshold": threshold,
                   "interval_variant": development["interval_variant"], "conformal_quantiles": quantiles,
                   "warning": warn, "decision_D": development["decision_D"]["locked"]},
        "external_outcomes_read": False,
    }
    write(lock_path, lock)
    print(json.dumps({k: development[k] for k in ["product", "forecast_date_macro_mae_usable", "nested_selection_estimate", "trust_aurc",
                                                 "trust_locked", "warning_locked", "final_params", "seconds"]}, indent=2))
    print(json.dumps({"warning": warning, "decision_D": development["decision_D"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "develop", "external", "audit"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--unseal", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out)
    elif args.stage == "develop":
        develop(args.out, args.jobs)
    else:
        raise SystemExit(f"Stage {args.stage} is implemented after the development lock (see protocol).")


if __name__ == "__main__":
    main()
