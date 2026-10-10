"""Train the final plate-context transformer on all 41 batches and export it for the browser app's twin panel.

The twin panel is a research preview: the PCT's day-9/day-12 functional-profile forecasts were cross-validated
on 41 batches (docs/neuroforecast_v3_results.md) but never scored on a sealed test, and the PCT is not the
shipped day-12 forecaster (v2 is).

Steps
  1. Fit 5 seeds on every batch with the locked v3 configuration (deterministic CUDA); save state dicts.
  2. Write app/models/v3/pct.bin (float32 weights, seed after seed) and pct.json (layout, normalizer, constants).
  3. Check a float64 NumPy forward pass, written to mirror app/twin.js line for line, against PyTorch on all
     41 batches; refuse to export if any forecast differs by more than 1e-4.
  4. Write tests/fixtures/v3_twin_<example>.json: PyTorch outputs for the app's example CSVs, which
     tests/test_web_twin.mjs must reproduce in JavaScript.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_data as data  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402
import neuroforecast_v3_data as v3d  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = v3d.OUT / "models"
WEB_DIR = ROOT / "app/models/v3"
FIXTURES = ROOT / "tests/fixtures"
EXAMPLES = {"external_20160120": ROOT / "app/examples/v2/external_20160120_day5_day7.csv",
            "reserve_20171004": ROOT / "app/examples/v2/reserve_20171004_low_quality_day5_day7.csv",
            "reserve_20171011": ROOT / "app/examples/v2/reserve_20171011_usable_day5_day7.csv"}
TOLERANCE = 1e-4


def torch_setup():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


# --------------------------------------------------------------------------- NumPy mirror of app/twin.js

def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def numpy_forward(p: dict, cfg: dict, x, plate, chem, cond, cond_x, n_cond):
    """One sequence (no padding). p: float64 arrays keyed by state-dict name."""
    d, heads = cfg["d"], cfg["heads"]
    dk = d // heads
    lin = lambda a, name: a @ p[name + ".weight"].T + p[name + ".bias"]  # noqa: E731
    h = lin(gelu(lin(x, "embed.0")), "embed.2")
    same = np.stack([plate[:, None] == plate[None, :],
                     (chem[:, None] == chem[None, :]) & (chem[:, None] >= 0),
                     (cond[:, None] == cond[None, :]) & (cond[:, None] >= 0)]).astype(float)
    n = len(x)
    for i in range(cfg["layers"]):
        pre = f"blocks.{i}."
        a = layer_norm(h, p[pre + "norm1.weight"], p[pre + "norm1.bias"])
        qkv = lin(a, pre + "qkv").reshape(n, 3, heads, dk)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                       # [n, heads, dk]
        logits = np.einsum("ihd,jhd->hij", q, k) / math.sqrt(dk) + np.einsum("hr,rij->hij", p[pre + "rel"], same)
        logits -= logits.max(-1, keepdims=True)
        attn = np.exp(logits)
        attn /= attn.sum(-1, keepdims=True)
        out = np.einsum("hij,jhd->ihd", attn, v).reshape(n, d)
        h = h + lin(out, pre + "proj")
        f = layer_norm(h, p[pre + "norm2.weight"], p[pre + "norm2.bias"])
        h = h + lin(gelu(lin(f, pre + "ff.0")), pre + "ff.3")
    h = layer_norm(h, p["norm.weight"], p["norm.bias"])
    pooled = np.zeros((n_cond, d))
    for c in range(n_cond):
        members = cond == c
        if members.any():
            pooled[c] = h[members].mean(0)
    head = lin(gelu(lin(np.concatenate([pooled, cond_x[:, :cfg["n_cond_in"]]], 1), "cond.0")), "cond.3")
    q50 = head[:, 1]
    return np.stack([q50 - softplus(head[:, 0]), q50, q50 + softplus(head[:, 2])], 1), head[:, 3:]


# --------------------------------------------------------------------------- example CSVs -> batches

def wells_from_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """The v3 well table for day-5/day-7 rows only (later days absent)."""
    keys = ["date", "Plate.SN", "well"]
    static = raw.sort_values("DIV").groupby(keys).agg(dose=("dose", "first"), identity=("identity", "first"),
                                                       eligible=("eligible_identity", "first"), trt=("trt", "first"))
    wide = raw.pivot_table(index=keys, columns="DIV", values=v3d.READOUTS, aggfunc="first", dropna=False)
    wide.columns = [f"{field}__d{day}" for field, day in wide.columns]
    for field in v3d.READOUTS:
        for day in v3d.DAYS:
            if f"{field}__d{day}" not in wide:
                wide[f"{field}__d{day}"] = np.nan
    wide = wide[[f"{field}__d{day}" for day in v3d.DAYS for field in v3d.READOUTS]]
    return static.join(wide).reset_index()


def example_batches(path: Path, pct, twin_columns: list[str]):
    import neuroforecast_triage_v2 as cli

    cfg = v2.v2_config()
    raw = cli.normalize(pd.read_csv(path, dtype={"identity": str, "Plate.SN": str, "well": str}), cfg["feature_readouts"])
    features = v2.features_from(raw, cfg)
    cases = features.index.to_frame(index=False).assign(target=np.nan, usable=False)
    twin = pd.DataFrame(np.nan, index=range(len(cases)), columns=twin_columns)
    wells = wells_from_rows(raw)
    # well_inputs marks edge wells from the maxima of the table it is given; training saw full 48-well plates
    # (rows A-F, columns 1-8), and the browser uses those maxima, so a fixture must contain full plates too.
    rows = wells.well.str[0].map(lambda s: ord(s.upper()) - ord("A"))
    cols = pd.to_numeric(wells.well.str[1:], errors="coerce") - 1
    if rows.max() != 5 or cols.max() != 7:
        raise ValueError(f"{path.name}: not a full 48-well layout; edge flags would differ from training")
    batches = pct.build_batches(wells, cases, features.reset_index(drop=True), twin)
    return batches, cases


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=v3d.OUT)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    torch = torch_setup()
    import neuroforecast_v3_pct as pct

    proto = v3d.protocol()
    spec = proto["candidates"]["pct"]
    cfg = pct.Config(**spec["params"])
    loaded = v3d.load(args.out)
    cases, x, twin = loaded["cases"], loaded["features"], loaded["twin"]
    batches = pct.build_batches(v3d.load_wells(args.out), cases, x, twin)
    dates = sorted(cases.date.unique())
    device = torch.device("cuda")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [proto["seed"] + s for s in range(spec["seeds_outer"])]
    models, norm = [], None
    for seed in seeds:
        path = MODEL_DIR / f"final_pct_seed{seed}.pt"
        if path.exists() and not args.retrain:
            saved = torch.load(path, weights_only=False)
            model = pct.PCT(*saved["dims"], cfg.d, cfg.layers, cfg.heads, cfg.dropout, cfg.context, cfg.relational,
                            saved["n_cond_in"]).to(device)
            model.load_state_dict(saved["state"])
            model.eval()
            norm = saved["norm"]
        else:
            model, norm = pct.train([batches[d] for d in dates], set(), cfg, seed, device)
            dims = (batches[dates[0]].x.shape[1], batches[dates[0]].aux.shape[1], batches[dates[0]].twin.shape[1])
            torch.save({"state": model.state_dict(), "norm": norm, "dims": dims, "n_cond_in": model.n_cond_in, "seed": seed}, path)
        models.append(model)
        print(f"seed {seed} ready", flush=True)

    # ---- weights for the browser
    names = [k for k in models[0].state_dict() if not k.startswith("aux.")]
    layout, offset = [], 0
    for k in names:
        shape = list(models[0].state_dict()[k].shape)
        size = int(np.prod(shape))
        layout.append({"name": k, "shape": shape, "offset": offset, "size": size})
        offset += size
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    blob = np.concatenate([m.state_dict()[k].detach().cpu().numpy().astype("<f4").ravel() for m in models for k in names])
    (WEB_DIR / "pct.bin").write_bytes(blob.tobytes())
    raw_wells = v3d.load_wells(args.out)
    rows = raw_wells.well.str[0].map(lambda s: ord(s.upper()) - ord("A"))
    cols = pd.to_numeric(raw_wells.well.str[1:], errors="coerce") - 1
    manifest = {
        "label": "research preview: cross-validated on 41 batches, not sealed-test validated; not the shipped forecaster",
        "seeds": seeds, "floats_per_seed": offset, "layout": layout,
        "config": {"d": cfg.d, "layers": cfg.layers, "heads": cfg.heads, "n_cond_in": models[0].n_cond_in},
        "normalizer": {"mu": norm.mu.tolist(), "sd": norm.sd.tolist()},
        "readouts": v3d.READOUTS, "cond_columns": pct.COND_COLUMNS, "twin_targets": twin.columns.tolist(),
        "row_max": int(rows.max()), "col_max": int(cols.max()),
        "evidence": {"cv_batches": 41, "twin_vs_carry_forward_wins": "36 of 36 targets",
                     "twin_vs_per_target_extra_trees": "about equal (16 of 36 targets, mean error ratio 1.02)"},
        "bin_sha256": data.digest(WEB_DIR / "pct.bin"),
    }
    (WEB_DIR / "pct.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"pct.bin {blob.nbytes / 1e6:.1f} MB, {offset} floats per seed", flush=True)

    # ---- NumPy mirror against PyTorch on every batch
    params = [{k: m.state_dict()[k].detach().cpu().numpy().astype(np.float64) for k in names} for m in models]
    mcfg = manifest["config"]
    worst = 0.0
    for date in dates:
        b = batches[date]
        ref = pct.predict(models[0], norm, [b], device)   # seed 0 only: PyTorch reference
        xn = norm.x(b.x).astype(np.float64)
        cx = np.nan_to_num(b.cond_x).astype(np.float32).astype(np.float64)
        quant, tw = numpy_forward(params[0], mcfg, xn, b.plate, b.chem, b.cond, cx, len(b.case_pos))
        for c, pos in enumerate(b.case_pos):
            if int(pos) in ref:
                mine = np.concatenate([quant[c] + b.cond_x[c, 1], tw[c]])
                worst = max(worst, float(np.abs(mine - ref[int(pos)]).max()))
    print(f"NumPy mirror vs PyTorch, 41 batches, seed 0: max |diff| = {worst:.2e}", flush=True)
    if worst > TOLERANCE:
        raise SystemExit("The NumPy mirror disagrees with PyTorch; do not ship")

    # ---- fixtures for the JavaScript test (5-seed ensemble, PyTorch)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for label, path in EXAMPLES.items():
        ex_batches, ex_cases = example_batches(path, pct, twin.columns.tolist())
        out = []
        for date, b in ex_batches.items():
            runs = [pct.predict(m, norm, [b], device) for m in models]
            for pos in runs[0]:
                stack = np.stack([r[pos] for r in runs]).mean(0)
                row = ex_cases.iloc[pos]
                out.append({"date": int(row.date), "identity": str(row.identity), "dose": float(row.dose),
                            "q10": float(stack[0]), "q50": float(stack[1]), "q90": float(stack[2]),
                            "twin": [float(v) for v in stack[3:]]})
        (FIXTURES / f"v3_twin_{label}.json").write_text(json.dumps({"tolerance": TOLERANCE, "conditions": out}, indent=1) + "\n", encoding="utf-8")
        print(f"fixture {label}: {len(out)} conditions", flush=True)


if __name__ == "__main__":
    main()
