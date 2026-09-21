"""Export the shipped models to a compact binary the browser app can evaluate without a server.

Each forest is flattened into typed arrays. The exporter then re-predicts through the same
traversal the JavaScript performs and refuses to write anything that does not match scikit-learn.
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

LEAF = -2
MODELS = {
    "forecast": "final_extra_trees_relative_day7",
    "difficulty": "final_sigma",
    "comparator": "final_extra_trees_activity_coordination",
}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=data.ROOT / "app/models")
    parser.add_argument("--bundle", type=Path, default=triage.DEFAULT_BUNDLE)
    parser.add_argument("--lock", type=Path, default=triage.DEFAULT_LOCK)
    parser.add_argument("--reserve", type=Path, default=triage.DEFAULT_RESERVE)
    args = parser.parse_args()
    manifest = run(args.out, args.bundle, args.lock, args.reserve)
    print(json.dumps({k: v["max_abs_difference_vs_sklearn"] for k, v in manifest["models"].items()}, indent=2))
