"""Export the shipped models to a compact binary the browser app can evaluate without a server.

Each forest is flattened into typed arrays. The exporter then re-predicts through the same
traversal the JavaScript performs and refuses to write anything that does not match scikit-learn.

  --set v1   the three v1 forests into app/models (the default, unchanged)
  --set v2   the locked v2 forecast and difficulty forests into app/models/v2
  --set all  both

The v2 layout is exact: thresholds and leaf values stay float64, and the traversal casts each input
to float32 before comparing, as scikit-learn does, so the browser reproduces its predictions bit for bit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import neuroforecast_data as data
import neuroforecast_gate as gate
import neuroforecast_triage as triage
import neuroforecast_triage_v2 as triage_v2
import neuroforecast_v2 as v2

LEAF = -2
MODELS = {
    "forecast": "final_extra_trees_relative_day7",
    "difficulty": "final_sigma",
    "comparator": "final_extra_trees_activity_coordination",
}
V2_MODELS = {
    "forecast": "final_small_refit",
    "difficulty": "final_sigma_v2",
}
V2_FIXTURES = ["tests/fixtures/v2_external_day5_day7.csv", "tests/fixtures/v2_reserve_day5_day7.csv"]
V2_TOLERANCE = 1e-12


def flatten(pipeline) -> dict:
    """Flatten a fitted Extra Trees pipeline into arrays: nodes concatenated, one offset per tree."""
    forest = pipeline.named_steps["extratreesregressor"]
    imputer = pipeline.named_steps["simpleimputer"]
    feature, threshold, left, right, value, offsets = [], [], [], [], [], [0]
    for estimator in forest.estimators_:
        t = estimator.tree_
        base = offsets[-1]
        feature.append(t.feature.astype(np.int16))
        threshold.append(t.threshold.astype(np.float32))
        # child indices are tree-local; shift them into the concatenated array
        left.append(np.where(t.children_left < 0, -1, t.children_left + base).astype(np.int32))
        right.append(np.where(t.children_right < 0, -1, t.children_right + base).astype(np.int32))
        value.append(t.value.reshape(-1).astype(np.float32))
        offsets.append(base + t.node_count)
    return {
        "feature": np.concatenate(feature),
        "threshold": np.concatenate(threshold),
        "left": np.concatenate(left),
        "right": np.concatenate(right),
        "value": np.concatenate(value),
        "offsets": np.array(offsets, dtype=np.int32),
        "medians": imputer.statistics_.astype(np.float32),
    }


def traverse(flat: dict, X: np.ndarray) -> np.ndarray:
    """The exact traversal the JavaScript implements: impute, then walk each tree, then average."""
    filled = np.where(np.isnan(X), flat["medians"][None, :], X)
    feature, threshold, left, right, value, offsets = (flat[k] for k in ["feature", "threshold", "left", "right", "value", "offsets"])
    out = np.zeros(len(filled))
    for row_index, row in enumerate(filled):
        total = 0.0
        for tree in range(len(offsets) - 1):
            node = int(offsets[tree])
            while feature[node] != LEAF:
                node = int(left[node]) if row[feature[node]] <= threshold[node] else int(right[node])
            total += float(value[node])
        out[row_index] = total / (len(offsets) - 1)
    return out


def pack(flat: dict) -> bytes:
    """Little-endian: int16 feature, float32 threshold, int32 left/right, float32 value, then offsets and medians."""
    parts = [
        flat["feature"].astype("<i2").tobytes(),
        flat["threshold"].astype("<f4").tobytes(),
        flat["left"].astype("<i4").tobytes(),
        flat["right"].astype("<i4").tobytes(),
        flat["value"].astype("<f4").tobytes(),
        flat["offsets"].astype("<i4").tobytes(),
        flat["medians"].astype("<f4").tobytes(),
    ]
    return b"".join(parts)


def run(out: Path, bundle_dir: Path, lock_path: Path, reserve_path: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    bundle = triage.load_bundle(bundle_dir, lock_path, reserve_path)
    cfg = data.protocol()

    # A real feature matrix to validate against, from the shipped example.
    example = pd.read_csv(data.ROOT / "examples/reserve_20171011_usable_day5_day7.csv", dtype={"identity": str})
    frame, _ = triage.build_features(example, cfg)

    manifest = {"models": {}, "constants": {}, "generated_from": {}}
    for label, name in MODELS.items():
        loaded = joblib.load(bundle_dir / f"{name}.joblib")
        pipeline, columns = loaded["model"], loaded["columns"]
        flat = flatten(pipeline)
        X = frame[columns].to_numpy(dtype=float)
        expected = pipeline.predict(frame[columns])
        got = traverse(flat, X)
        difference = float(np.max(np.abs(expected - got)))
        if difference > 1e-5:
            raise SystemExit(f"{label}: flattened traversal disagrees with scikit-learn by {difference}")
        payload = pack(flat)
        (out / f"{label}.bin").write_bytes(payload)
        manifest["models"][label] = {
            "file": f"{label}.bin",
            "trees": int(len(flat["offsets"]) - 1),
            "nodes": int(len(flat["feature"])),
            "columns": columns,
            "bytes": len(payload),
            "max_abs_difference_vs_sklearn": difference,
        }
        manifest["generated_from"][label] = data.digest(bundle_dir / f"{name}.joblib")
        print(f"{label:11s} {len(flat['offsets'])-1} trees, {len(flat['feature'])} nodes, {len(payload)/1e6:.2f} MB, max diff {difference:.2e}")

    manifest["constants"] = {
        "readouts": cfg["feature_readouts"],
        "pseudocount": cfg["target_pseudocount"],
        "min_replicates": cfg["min_replicates_per_case_day"],
        "reliability_columns": gate.RELIABILITY,
        "sigma_floor": gate.SIGMA_FLOOR,
        "sigma_threshold": bundle["sigma_threshold"],
        "scaled_quantile": bundle["scaled_quantile"],
        "interval_level": bundle["level"],
        "budget": bundle["budget"],
        "quiet_limit": triage.QUIET_LIMIT,
        "flag_thresholds": bundle["flag_thresholds"],
        "evidence": bundle["evidence"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- v2

def flatten_exact(pipeline) -> dict:
    """Flatten without rounding: one float64 per node holds the threshold (split node) or the leaf value."""
    forest = pipeline.named_steps["extratreesregressor"]
    imputer = pipeline.named_steps["simpleimputer"]
    if forest.n_outputs_ != 1:
        raise SystemExit("Only single-output regression forests can be exported")
    feature, split, left, right, offsets = [], [], [], [], [0]
    for estimator in forest.estimators_:
        t = estimator.tree_
        base = offsets[-1]
        leaf = t.children_left < 0
        if not np.array_equal(leaf, t.feature == LEAF):
            raise SystemExit("A tree marks leaves inconsistently")
        feature.append(t.feature.astype(np.int16))
        split.append(np.where(leaf, t.value.reshape(-1), t.threshold).astype(np.float64))
        left.append(np.where(leaf, -1, t.children_left + base).astype(np.int32))
        right.append(np.where(leaf, -1, t.children_right + base).astype(np.int32))
        offsets.append(base + t.node_count)
    medians = imputer.statistics_.astype(np.float64)
    if not np.all(np.isfinite(medians)):
        raise SystemExit("The imputer holds a non-finite median")
    return {
        "feature": np.concatenate(feature),
        "left": np.concatenate(left),
        "right": np.concatenate(right),
        "split": np.concatenate(split),
        "offsets": np.array(offsets, dtype=np.int32),
        "medians": medians,
    }


def traverse_exact(flat: dict, X: np.ndarray) -> np.ndarray:
    """The traversal app/triage.js implements for v2, vectorised over rows.

    Impute with the float64 medians, cast to float32 (scikit-learn's tree input type), compare against
    the float64 threshold, then add the leaves tree by tree in order and divide, as the forest does.
    """
    filled = np.where(np.isnan(X), flat["medians"][None, :], X).astype(np.float32).astype(np.float64)
    feature, left, right, split, offsets = (flat[k] for k in ["feature", "left", "right", "split", "offsets"])
    rows = np.arange(len(filled))
    total = np.zeros(len(filled))
    for tree in range(len(offsets) - 1):
        node = np.full(len(filled), offsets[tree], dtype=np.int64)
        active = feature[node] != LEAF
        while active.any():
            at = node[active]
            go_left = filled[rows[active], feature[at]] <= split[at]
            node[active] = np.where(go_left, left[at], right[at])
            active = feature[node] != LEAF
        total += split[node]
    return total / (len(offsets) - 1)


def pack_exact(flat: dict) -> bytes:
    """Little-endian: int16 feature, int32 left, int32 right, float64 split, int32 offsets."""
    return b"".join([
        flat["feature"].astype("<i2").tobytes(),
        flat["left"].astype("<i4").tobytes(),
        flat["right"].astype("<i4").tobytes(),
        flat["split"].astype("<f8").tobytes(),
        flat["offsets"].astype("<i4").tobytes(),
    ])


def v2_feature_frames() -> dict[str, pd.DataFrame]:
    """Real v2 feature matrices from the shipped fixtures, built by the Python v2 pipeline itself."""
    cfg = v2.v2_config()
    frames = {}
    for relative in V2_FIXTURES:
        observed = pd.read_csv(data.ROOT / relative, dtype={"identity": str})
        frames[relative] = v2.features_from(triage_v2.normalize(observed, cfg["feature_readouts"]), cfg)
    return frames


def run_v2(out: Path, bundle_dir: Path, lock_path: Path) -> dict:
    bundle = triage_v2.load_bundle(bundle_dir, lock_path)  # refuses models that differ from the lock
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    cfg = v2.v2_config()
    frames = v2_feature_frames()
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": {
            "layout": "little-endian int16 feature[nodes], int32 left[nodes], int32 right[nodes], float64 split[nodes], int32 offsets[trees+1]",
            "leaf": LEAF,
            "split": "threshold at a split node, leaf value at a leaf",
            "traversal": "impute NaN with the float64 median, round to float32, go left when x <= threshold; sum leaves tree by tree, divide by trees",
        },
        "models": {}, "constants": {}, "generated_from": {},
    }
    for label, name in V2_MODELS.items():
        path = bundle_dir / f"{name}.joblib"
        loaded = joblib.load(path)
        pipeline, columns = loaded["model"], loaded["columns"]
        flat = flatten_exact(pipeline)
        worst, rows = 0.0, 0
        for frame in frames.values():
            expected = pipeline.predict(frame[columns])
            got = traverse_exact(flat, frame[columns].to_numpy(dtype=float))
            worst = max(worst, float(np.max(np.abs(expected - got))))
            rows += len(frame)
        if worst > V2_TOLERANCE:
            raise SystemExit(f"v2 {label}: flattened traversal disagrees with scikit-learn by {worst}")
        payload = pack_exact(flat)
        (out / f"{label}.bin").write_bytes(payload)
        manifest["models"][label] = {
            "file": f"{label}.bin",
            "source": f"models/v2/{name}.joblib",
            "trees": int(len(flat["offsets"]) - 1),
            "nodes": int(len(flat["feature"])),
            "columns": columns,
            "medians": [float(m) for m in flat["medians"]],
            "bytes": len(payload),
            "checked_rows": rows,
            "max_abs_difference_vs_sklearn": worst,
        }
        manifest["generated_from"][f"{name}.joblib"] = data.digest(path)
        print(f"v2 {label:10s} {len(flat['offsets'])-1} trees, {len(flat['feature'])} nodes, {len(payload)/1e6:.2f} MB, "
              f"max diff {worst:.2e} over {rows} fixture conditions")
    manifest["generated_from"]["development_lock.json"] = data.digest(lock_path)

    locked = lock["locked"]
    manifest["constants"] = {
        "readouts": cfg["feature_readouts"],
        "cutoff_days": cfg["cutoff_days"],
        "pseudocount": cfg["target_pseudocount"],
        "min_replicates": cfg["min_replicates_per_case_day"],
        "reliability_columns": gate.RELIABILITY,
        "batch_columns": v2.BATCH,
        "sigma_floor": gate.SIGMA_FLOOR,
        "product": locked["product"],
        "trust": locked["trust"],
        "trust_threshold": bundle["threshold"],
        "interval_level": float(triage_v2.LEVEL),
        "interval_quantile": bundle["quantile"],
        "evidence": triage_v2.EVIDENCE,
        "limits": triage_v2.LIMITS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=["v1", "v2", "all"], default="v1", help="which model set to export")
    parser.add_argument("--out", type=Path, default=data.ROOT / "app/models")
    parser.add_argument("--bundle", type=Path, default=triage.DEFAULT_BUNDLE)
    parser.add_argument("--lock", type=Path, default=triage.DEFAULT_LOCK)
    parser.add_argument("--reserve", type=Path, default=triage.DEFAULT_RESERVE)
    parser.add_argument("--out-v2", type=Path, default=data.ROOT / "app/models/v2")
    parser.add_argument("--bundle-v2", type=Path, default=triage_v2.BUNDLE)
    parser.add_argument("--lock-v2", type=Path, default=triage_v2.LOCK)
    args = parser.parse_args()
    if args.set in ("v1", "all"):
        manifest = run(args.out, args.bundle, args.lock, args.reserve)
        print(json.dumps({k: v["max_abs_difference_vs_sklearn"] for k, v in manifest["models"].items()}, indent=2))
    if args.set in ("v2", "all"):
        manifest = run_v2(args.out_v2, args.bundle_v2, args.lock_v2)
        print(json.dumps({k: v["max_abs_difference_vs_sklearn"] for k, v in manifest["models"].items()}, indent=2))
