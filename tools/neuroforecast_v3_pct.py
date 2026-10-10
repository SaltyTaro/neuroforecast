"""Plate-context transformer (PCT): forecast day-12 network activity by reading a whole culture batch at once.

Every physical well of a batch (all plates, controls and treated wells) is one token built from its day-5
and day-7 recordings only. Attention runs across the batch with learned relational biases for wells that
share a plate, a chemical, or a condition, so a forecast can draw on the batch's own controls (how this
culture is maturing) and on the chemical's whole dose series. Identity, date and plate labels never enter
as features: the model is permutation-equivariant over wells, plates and chemicals.

Heads
  condition   quantiles 0.1 / 0.5 / 0.9 of the change from day 7 to day 12 (the v2 target), from the mean
              embedding of the condition's wells plus its day-5/day-7 values; the 0.5 quantile is the forecast
  well (aux)  every well's day-9 and day-12 readouts and its day-12 activity relative to its plate's
              controls, for controls too: a multi-task signal about how cultures develop

Labels of any chemical held out in the current test fold are masked in training batches (the chemical purge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuroforecast_representation as rep  # noqa: E402
import neuroforecast_v3_data as v3d  # noqa: E402

READOUTS = v3d.READOUTS
QUANTILES = (0.1, 0.5, 0.9)
NS = READOUTS.index("ns.n")
# Condition-level inputs: the v2 'small' columns (day-5/day-7 relative activity, dose, five control contrasts).
# Column 1 is day-7 relative activity, the anchor the forecast is a change from.
COND_COLUMNS = rep.selected_columns("extra_trees_activity_coordination", None)
COND_MINIMAL = 3


# --------------------------------------------------------------------------- tensors

def _plate_control_median(wells: pd.DataFrame, column: str) -> pd.Series:
    controls = wells.loc[wells.dose == 0]
    return wells[["date", "Plate.SN"]].merge(controls.groupby(["date", "Plate.SN"])[column].median().rename("m").reset_index(),
                                             on=["date", "Plate.SN"], how="left").m.to_numpy()


def well_inputs(wells: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Token features from days 5 and 7 only."""
    cols, names = [], []
    for day in (5, 7):
        for f in READOUTS:
            v = wells[f"{f}__d{day}"].to_numpy(dtype=float)
            c = _plate_control_median(wells, f"{f}__d{day}")
            denom = np.abs(v) + np.abs(c)
            contrast = np.divide(v - c, denom, out=np.zeros_like(v), where=(denom != 0) & np.isfinite(denom))
            cols += [np.arcsinh(v), contrast, np.isnan(v).astype(float)]
            names += [f"raw__{f}__d{day}", f"contrast__{f}__d{day}", f"missing__{f}__d{day}"]
        ns = wells[f"ns.n__d{day}"].to_numpy(dtype=float)
        cns = _plate_control_median(wells, f"ns.n__d{day}")
        cols.append(np.log2((ns + 1) / (cns + 1)))
        names.append(f"rel_ns__d{day}")
    control = (wells.dose == 0).to_numpy(dtype=float)
    dose = wells.dose.to_numpy(dtype=float)
    log_dose = np.where(dose > 0, np.log10(np.where(dose > 0, dose, 1)), 0.0)
    row = wells.well.str[0].map(lambda s: ord(s.upper()) - ord("A")).to_numpy(dtype=float)
    col = pd.to_numeric(wells.well.str[1:], errors="coerce").to_numpy(dtype=float) - 1
    edge = ((row == 0) | (row == row.max()) | (col == 0) | (col == np.nanmax(col))).astype(float)
    cols += [control, log_dose, row / 5.0, col / 7.0, edge]
    names += ["is_control", "log10_dose", "row", "col", "edge"]
    x = np.column_stack(cols)
    return x, names


def well_aux_labels(wells: pd.DataFrame) -> np.ndarray:
    """Later-day labels per well: arcsinh readouts on days 9 and 12 and day-12 activity relative to plate controls."""
    cols = [np.arcsinh(wells[f"{f}__d{day}"].to_numpy(dtype=float)) for day in (9, 12) for f in READOUTS]
    ns = wells["ns.n__d12"].to_numpy(dtype=float)
    cols.append(np.log2((ns + 1) / (_plate_control_median(wells, "ns.n__d12") + 1)))
    return np.column_stack(cols)


@dataclass
class Batch:
    date: int
    x: np.ndarray            # [L, F] token inputs (days 5/7)
    plate: np.ndarray        # [L] plate index within the batch
    chem: np.ndarray         # [L] chemical group within the batch (-1 for controls)
    cond: np.ndarray         # [L] condition index within the batch (-1 if none)
    identity: np.ndarray     # [L] identity string ('' for controls)
    aux: np.ndarray          # [L, A] later-day well labels (NaN where missing)
    case_pos: np.ndarray     # [C] position of each condition in the cases table
    cond_x: np.ndarray       # [C, K] condition inputs (COND_COLUMNS); column 1 is day-7 relative activity
    target: np.ndarray       # [C]
    usable: np.ndarray       # [C] bool
    cond_identity: np.ndarray = field(default=None)
    twin: np.ndarray = field(default=None)  # [C, T] later-day functional profile (NaN where missing)


def build_batches(wells: pd.DataFrame, cases: pd.DataFrame, features: pd.DataFrame, twin: pd.DataFrame) -> dict[int, Batch]:
    x_all, names = well_inputs(wells)
    aux_all = well_aux_labels(wells)
    key = list(zip(cases.date, cases.identity, cases.dose))
    case_index = {k: i for i, k in enumerate(key)}
    batches = {}
    for date, idx in wells.groupby("date").indices.items():
        w = wells.iloc[idx]
        plates = {p: i for i, p in enumerate(sorted(w["Plate.SN"].unique()))}
        group = np.where(w.dose > 0, np.where(w.identity.notna(), w.identity.astype(str), "TRT:" + w.trt.astype(str)), "")
        chem_ids = {g: i for i, g in enumerate(sorted(set(group) - {""}))}
        cond_pos, cond_of_well = [], np.full(len(w), -1)
        lookup = {}
        for j, (dose, ident, elig) in enumerate(zip(w.dose, w.identity, w.eligible)):
            if dose > 0 and bool(elig) and isinstance(ident, str):
                pos = case_index.get((int(date), ident, float(dose)))
                if pos is not None:
                    if pos not in lookup:
                        lookup[pos] = len(cond_pos)
                        cond_pos.append(pos)
                    cond_of_well[j] = lookup[pos]
        cond_pos = np.array(cond_pos, dtype=int)
        c = cases.iloc[cond_pos]
        f = features.iloc[cond_pos]
        batches[int(date)] = Batch(
            date=int(date), x=x_all[idx].astype(np.float32),
            plate=w["Plate.SN"].map(plates).to_numpy(), chem=np.array([chem_ids.get(g, -1) for g in group]),
            cond=cond_of_well, identity=np.where(w.identity.notna() & (w.dose > 0), w.identity.astype(str), ""),
            aux=aux_all[idx].astype(np.float32), case_pos=cond_pos,
            cond_x=np.nan_to_num(f[COND_COLUMNS].to_numpy(dtype=np.float32)),
            target=c.target.to_numpy(dtype=np.float32), usable=c.usable.to_numpy(dtype=bool),
            cond_identity=c.identity.to_numpy(), twin=twin.iloc[cond_pos].to_numpy(dtype=np.float32))
    return batches


# --------------------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dk = heads, d // heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.rel = nn.Parameter(torch.zeros(heads, 3))  # same plate, same chemical, same condition

    def forward(self, h, relations, pad, context: bool):
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(b, n, 3, self.heads, self.dk).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        logits = logits + torch.einsum("hr,brij->bhij", self.rel, relations)
        eye = torch.eye(n, dtype=torch.bool, device=h.device)[None, None]
        mask = pad[:, None, None, :]
        if not context:  # ablation: every well attends only to itself
            mask = mask | ~eye
        logits = logits.masked_fill(mask & ~eye, float("-inf"))  # a token always sees itself: no empty rows
        attn = torch.softmax(logits, dim=-1)
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(b, n, d)
        h = h + self.drop(self.proj(out))
        return h + self.drop(self.ff(self.norm2(h)))


class PCT(nn.Module):
    def __init__(self, n_in: int, n_aux: int, n_twin: int, d: int, layers: int, heads: int, dropout: float,
                 context: bool = True, relational: bool = True, n_cond_in: int = COND_MINIMAL):
        super().__init__()
        self.context, self.relational, self.n_cond_in = context, relational, n_cond_in
        self.embed = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.aux = nn.Linear(d, n_aux)
        self.cond = nn.Sequential(nn.Linear(d + n_cond_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, len(QUANTILES) + n_twin))

    def forward(self, x, plate, chem, cond, keep, cond_x, n_cond: int):
        """x [B,L,F]; plate/chem/cond [B,L] (-1 = none); keep [B,L] bool; cond_x [B,C,3].
        Returns ordered quantile deltas [B,C,3], twin [B,C,T], well aux [B,L,A] and kept-well counts [B,C]."""
        same_plate = plate[:, :, None] == plate[:, None, :]
        same_chem = (chem[:, :, None] == chem[:, None, :]) & (chem[:, :, None] >= 0)
        same_cond = (cond[:, :, None] == cond[:, None, :]) & (cond[:, :, None] >= 0)
        relations = torch.stack([same_plate, same_chem, same_cond], 1).float()
        if not self.relational:
            relations = torch.zeros_like(relations)
        pad = ~keep
        h = self.embed(x)
        for block in self.blocks:
            h = block(h, relations, pad, self.context)
        h = self.norm(h)
        onehot = (cond[:, None, :] == torch.arange(n_cond, device=x.device)[None, :, None]) & keep[:, None, :]
        counts = onehot.sum(-1)
        pooled = (onehot.float() / counts.clamp(min=1)[..., None]) @ h      # [B,C,d] mean over each condition's kept wells
        head = self.cond(torch.cat([pooled, cond_x[..., :self.n_cond_in]], dim=-1))
        delta, twin = head[..., :3], head[..., 3:]
        delta = torch.cat([delta[..., 1:2] - nn.functional.softplus(delta[..., 0:1]), delta[..., 1:2],
                           delta[..., 1:2] + nn.functional.softplus(delta[..., 2:3])], dim=-1)  # ordered quantiles
        return delta, twin, self.aux(h), counts


# --------------------------------------------------------------------------- training

@dataclass
class Config:
    d: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    aux_weight: float = 0.5
    twin_weight: float = 0.25
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_dates: int = 4
    keep_plate: float = 0.85
    drop_well: float = 0.1
    context: bool = True
    relational: bool = True
    cond_features: str = "minimal"   # 'minimal' (day-5, day-7, dose) or 'small' (all COND_COLUMNS)
    ema: float = 0.0                 # >0: predict with an exponential moving average of the weights


class Normalizer:
    def __init__(self, batches: list[Batch]):
        x = np.concatenate([b.x for b in batches])
        self.mu, self.sd = np.nanmean(x, 0), np.nanstd(x, 0)
        self.sd[~np.isfinite(self.sd) | (self.sd < 1e-6)] = 1.0
        self.mu[~np.isfinite(self.mu)] = 0.0
        aux = np.concatenate([b.aux for b in batches])
        self.amu, self.asd = np.nanmean(aux, 0), np.nanstd(aux, 0)
        self.asd[~np.isfinite(self.asd) | (self.asd < 1e-6)] = 1.0
        self.amu[~np.isfinite(self.amu)] = 0.0

    def x(self, x):
        z = (x - self.mu) / self.sd
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def aux(self, a):
        return ((a - self.amu) / self.asd).astype(np.float32)


class Staged:
    """A set of batches padded to common sizes and moved to the device once."""

    def __init__(self, batches: list[Batch], norm: Normalizer, purge: set[str], device):
        n_tok = max(len(b.plate) for b in batches)
        n_cond = max(max(len(b.case_pos) for b in batches), 1)
        B, F, A, T = len(batches), norm.mu.shape[0], norm.amu.shape[0], batches[0].twin.shape[1]
        x = np.zeros((B, n_tok, F), np.float32)
        plate = np.full((B, n_tok), -1, np.int64)
        chem, cond = plate.copy(), plate.copy()
        valid = np.zeros((B, n_tok), bool)
        aux = np.full((B, n_tok, A), np.nan, np.float32)
        cond_x = np.zeros((B, n_cond, batches[0].cond_x.shape[1]), np.float32)
        y = np.zeros((B, n_cond), np.float32)
        w = np.zeros((B, n_cond), np.float32)
        twin = np.full((B, n_cond, T), np.nan, np.float32)
        purge_list = sorted(purge)
        for i, b in enumerate(batches):
            m, c = len(b.plate), len(b.case_pos)
            x[i, :m] = norm.x(b.x)
            plate[i, :m], chem[i, :m], cond[i, :m] = b.plate, b.chem, b.cond
            valid[i, :m] = True
            a = norm.aux(b.aux)
            if purge_list:
                a[np.isin(b.identity, purge_list)] = np.nan
            aux[i, :m] = a
            if c:
                cond_x[i, :c] = b.cond_x
                y[i, :c] = np.nan_to_num(b.target - b.cond_x[:, 1])
                ok = b.usable & np.isfinite(b.target) & ~np.isin(b.cond_identity, purge_list)
                w[i, :c] = ok
                block = np.full((c, T), np.nan, np.float32)
                block[ok] = b.twin[ok]
                twin[i, :c] = block
        t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
        self.x, self.plate, self.chem, self.cond, self.valid = t(x), t(plate), t(chem), t(cond), t(valid)
        self.aux, self.cond_x, self.y, self.w, self.twin = t(aux), t(cond_x), t(y), t(w), t(twin)
        self.n_cond, self.batches = n_cond, batches

    def take(self, idx):
        return (self.x[idx], self.plate[idx], self.chem[idx], self.cond[idx], self.valid[idx], self.cond_x[idx],
                self.y[idx], self.w[idx], self.aux[idx], self.twin[idx])


def augment_keep(plate, cond, valid, cfg, gen):
    """Random plate dropout (at least one plate kept per batch) and treated-well dropout, on the device."""
    n_plates = int(plate.max().item()) + 1
    dev = plate.device
    keep_plate = torch.rand(plate.shape[0], n_plates, generator=gen, device=dev) < cfg.keep_plate
    present = torch.zeros_like(keep_plate)
    present.scatter_(1, plate.clamp(min=0), valid)
    keep_plate &= present
    none = ~keep_plate.any(1)
    if bool(none.any()):
        score = torch.where(present, torch.rand(present.shape, generator=gen, device=dev), torch.full(present.shape, -1.0, device=dev))
        pick = score.argmax(1)
        rows = torch.nonzero(none).flatten()
        keep_plate[rows, pick[rows]] = True
    on_kept_plate = torch.gather(keep_plate, 1, plate.clamp(min=0))
    survive = (cond < 0) | (torch.rand(cond.shape, generator=gen, device=dev) >= cfg.drop_well)
    return valid & on_kept_plate & survive


def pinball(pred, y, w):
    losses = []
    for j, q in enumerate(QUANTILES):
        e = y - pred[..., j]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    loss = torch.stack(losses, -1)
    loss = loss[..., 1] * 2 + loss[..., 0] + loss[..., 2]   # the median is the forecast: weight it double
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def train(batches: list[Batch], purge: set[str], cfg: Config, seed: int, device) -> tuple[PCT, Normalizer]:
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = Normalizer(batches)
    staged = Staged(batches, norm, purge, device)
    n_cond_in = COND_MINIMAL if cfg.cond_features == "minimal" else len(COND_COLUMNS)
    model = PCT(batches[0].x.shape[1], batches[0].aux.shape[1], batches[0].twin.shape[1], cfg.d, cfg.layers, cfg.heads,
                cfg.dropout, cfg.context, cfg.relational, n_cond_in).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(batches) / cfg.batch_dates)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1)
    averaged = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.ema)) if cfg.ema > 0 else None
    model.train()
    for _ in range(cfg.epochs):
        order = rng.permutation(len(batches))
        for s in range(steps_per_epoch):
            idx = torch.as_tensor(order[s * cfg.batch_dates:(s + 1) * cfg.batch_dates], device=device)
            x, plate, chem, cond, valid, cx, y, w, aux, tw = staged.take(idx)
            keep = augment_keep(plate, cond, valid, cfg, gen)
            delta, twin_pred, aux_pred, counts = model(x, plate, chem, cond, keep, cx, staged.n_cond)
            w = w * (counts > 0)
            loss = pinball(delta, y, w)
            if cfg.twin_weight > 0:
                ok = torch.isfinite(tw) & (counts > 0)[..., None]
                diff = torch.where(ok, twin_pred - torch.nan_to_num(tw), torch.zeros_like(twin_pred))
                loss = loss + cfg.twin_weight * diff.abs().sum() / ok.sum().clamp(min=1)
            if cfg.aux_weight > 0:
                ok = torch.isfinite(aux) & keep[..., None]
                diff = torch.where(ok, aux_pred - torch.nan_to_num(aux), torch.zeros_like(aux_pred))
                loss = loss + cfg.aux_weight * nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="sum") / ok.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if averaged is not None:
                averaged.update_parameters(model)
    if averaged is not None:
        model = averaged.module
    model.eval()
    return model, norm


@torch.no_grad()
def predict(model: PCT, norm: Normalizer, batches: list[Batch], device) -> dict[int, np.ndarray]:
    """Per case position: [q10, q50, q90, twin...] on the day-12 scale (the median is the forecast)."""
    out = {}
    for b in batches:
        staged = Staged([b], norm, set(), device)
        x, plate, chem, cond, valid, cx, _, _, _, _ = staged.take(torch.arange(1, device=device))
        delta, twin, _, counts = model(x, plate, chem, cond, valid, cx, staged.n_cond)
        c_n = len(b.case_pos)
        delta = delta[0, :c_n].cpu().numpy() + b.cond_x[:, 1:2]
        twin, counts = twin[0, :c_n].cpu().numpy(), counts[0, :c_n].cpu().numpy()
        for c in range(c_n):
            if counts[c] > 0:
                out[int(b.case_pos[c])] = np.concatenate([delta[c], twin[c]])
    return out


def fit_predict(train_batches: list[Batch], test_batches: list[Batch], purge: set[str], cfg: Config,
                seeds: list[int], device) -> dict[int, np.ndarray]:
    """Seed ensemble: per case, the mean of each quantile across seeds, plus the seed spread of the median."""
    runs = []
    for seed in seeds:
        model, norm = train(train_batches, purge, cfg, seed, device)
        runs.append(predict(model, norm, test_batches, device))
    out = {}
    for pos in runs[0]:
        stack = np.stack([r[pos] for r in runs])
        mean = stack.mean(0)
        out[pos] = np.concatenate([mean[:3], [stack[:, 1].std()], mean[3:]])  # q10, q50, q90, seed sd, twin...
    return out
