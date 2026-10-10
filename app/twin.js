/* NeuroForecast twin panel (research preview): the plate-context transformer (PCT) in plain JavaScript.
 *
 * Reads the same day-5/day-7 well rows as the v2 tool and forecasts, per condition, the day-9 and day-12
 * functional profile (relative network spikes and the control contrast of all 17 readouts). The PCT was
 * cross-validated on 41 batches but never scored on a sealed test, and it is not the shipped forecaster.
 *
 * Mirrors tools/neuroforecast_v3_pct.py (well_inputs, build_batches, PCT.forward, predict) and the NumPy
 * reference in tools/export_pct_for_web.py; tests/test_web_twin.mjs checks it against PyTorch within 1e-4.
 */

const SQRT1_2 = Math.SQRT1_2;
const SQRT_PI = Math.sqrt(Math.PI);

/* erf to about 1e-15: Taylor series near zero, continued fraction for erfc further out. */
export function erf(x) {
  const a = Math.abs(x);
  if (a < 2.2) {
    let term = a, sum = a;
    const a2 = a * a;
    for (let n = 1; n < 60; n++) {
      term *= -a2 / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
    }
    return Math.sign(x) * (2 / SQRT_PI) * sum;
  }
  let f = a;
  for (let k = 60; k >= 1; k--) f = a + (k / 2) / f;
  return Math.sign(x) * (1 - Math.exp(-a * a) / (SQRT_PI * f));
}

const gelu = (x) => 0.5 * x * (1 + erf(x * SQRT1_2));
const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));

function median(values) {
  const v = values.filter((x) => !Number.isNaN(x)).sort((a, b) => a - b);
  if (!v.length) return NaN;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/* ---------------------------------------------------------------- weights */

export class PCTEnsemble {
  constructor(buffer, manifest) {
    this.manifest = manifest;
    const all = new Float32Array(buffer);
    if (all.length !== manifest.floats_per_seed * manifest.seeds.length) throw new Error("pct.bin does not match pct.json");
    this.seeds = manifest.seeds.map((_, s) => {
      const params = {};
      for (const entry of manifest.layout) {
        const start = s * manifest.floats_per_seed + entry.offset;
        params[entry.name] = { data: all.subarray(start, start + entry.size), shape: entry.shape };
      }
      return params;
    });
    this.mu = manifest.normalizer.mu;
    this.sd = manifest.normalizer.sd;
  }
}

/* y[n, out] = x[n, in] W^T + b, with W stored [out, in] (PyTorch Linear). */
function linear(x, n, inDim, p) {
  const W = p.weight.data, b = p.bias.data, outDim = p.weight.shape[0];
  const y = new Float64Array(n * outDim);
  for (let i = 0; i < n; i++) {
    const xi = i * inDim, yi = i * outDim;
    for (let o = 0; o < outDim; o++) {
      let s = b[o];
      const wo = o * inDim;
      for (let k = 0; k < inDim; k++) s += x[xi + k] * W[wo + k];
      y[yi + o] = s;
    }
  }
  return y;
}

function layerNorm(x, n, d, w, b) {
  const y = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    let mu = 0;
    for (let k = 0; k < d; k++) mu += x[i * d + k];
    mu /= d;
    let v = 0;
    for (let k = 0; k < d; k++) v += (x[i * d + k] - mu) ** 2;
    const inv = 1 / Math.sqrt(v / d + 1e-5);
    for (let k = 0; k < d; k++) y[i * d + k] = (x[i * d + k] - mu) * inv * w.data[k] + b.data[k];
  }
  return y;
}

function mapInPlace(a, fn) {
  for (let i = 0; i < a.length; i++) a[i] = fn(a[i]);
  return a;
}

const lin = (p, name) => ({ weight: p[name + ".weight"], bias: p[name + ".bias"] });

/* One seed, one batch: returns {quant: [C][3] (change from day 7), twin: [C][T]}. */
function forward(p, cfg, x, nIn, plate, chem, cond, condX, nCond, nCondIn) {
  const n = plate.length, d = cfg.d, H = cfg.heads, dk = d / H;
  let h = linear(mapInPlace(linear(x, n, nIn, lin(p, "embed.0")), gelu), n, d, lin(p, "embed.2"));
  const scale = 1 / Math.sqrt(dk);
  for (let layer = 0; layer < cfg.layers; layer++) {
    const pre = `blocks.${layer}.`;
    const a = layerNorm(h, n, d, p[pre + "norm1.weight"], p[pre + "norm1.bias"]);
    const qkv = linear(a, n, d, lin(p, pre + "qkv"));          // [n, 3d]: (t, head, j) -> t*d + head*dk + j
    const rel = p[pre + "rel"].data;                            // [H, 3]
    const out = new Float64Array(n * d);
    const logits = new Float64Array(n);
    for (let hd = 0; hd < H; hd++) {
      for (let i = 0; i < n; i++) {
        const qi = i * 3 * d + hd * dk;
        let max = -Infinity;
        for (let j = 0; j < n; j++) {
          const kj = j * 3 * d + d + hd * dk;
          let s = 0;
          for (let t = 0; t < dk; t++) s += qkv[qi + t] * qkv[kj + t];
          s *= scale;
          if (plate[i] === plate[j]) s += rel[hd * 3];
          if (chem[i] >= 0 && chem[i] === chem[j]) s += rel[hd * 3 + 1];
          if (cond[i] >= 0 && cond[i] === cond[j]) s += rel[hd * 3 + 2];
          logits[j] = s;
          if (s > max) max = s;
        }
        let z = 0;
        for (let j = 0; j < n; j++) { logits[j] = Math.exp(logits[j] - max); z += logits[j]; }
        const oi = i * d + hd * dk;
        for (let j = 0; j < n; j++) {
          const w = logits[j] / z;
          const vj = j * 3 * d + 2 * d + hd * dk;
          for (let t = 0; t < dk; t++) out[oi + t] += w * qkv[vj + t];
        }
      }
    }
    const proj = linear(out, n, d, lin(p, pre + "proj"));
    for (let k = 0; k < h.length; k++) h[k] += proj[k];
    const f = layerNorm(h, n, d, p[pre + "norm2.weight"], p[pre + "norm2.bias"]);
    const ff = linear(mapInPlace(linear(f, n, d, lin(p, pre + "ff.0")), gelu), n, 4 * d, lin(p, pre + "ff.3"));
    for (let k = 0; k < h.length; k++) h[k] += ff[k];
  }
  h = layerNorm(h, n, d, p["norm.weight"], p["norm.bias"]);
  const width = d + nCondIn;
  const pooled = new Float64Array(nCond * width);
  const counts = new Array(nCond).fill(0);
  for (let i = 0; i < n; i++) {
    const c = cond[i];
    if (c < 0) continue;
    counts[c] += 1;
    for (let k = 0; k < d; k++) pooled[c * width + k] += h[i * d + k];
  }
  for (let c = 0; c < nCond; c++) {
    for (let k = 0; k < d; k++) pooled[c * width + k] /= Math.max(counts[c], 1);
    for (let k = 0; k < nCondIn; k++) pooled[c * width + d + k] = condX[c][k];
  }
  const head = linear(mapInPlace(linear(pooled, nCond, width, lin(p, "cond.0")), gelu), nCond, d, lin(p, "cond.3"));
  const outDim = p["cond.3.weight"].shape[0];
  const quant = [], twin = [];
  for (let c = 0; c < nCond; c++) {
    const r = head.subarray(c * outDim, (c + 1) * outDim);
    const q50 = r[1];
    quant.push([q50 - softplus(r[0]), q50, q50 + softplus(r[2])]);
    twin.push(Array.from(r.subarray(3)));
  }
  return { quant, twin, counts };
}

/* ---------------------------------------------------------------- tokens */

/* neuroforecast_v3_pct.well_inputs on the wells of one batch (days 5 and 7 only). */
function wellFeatures(wells, manifest) {
  const readouts = manifest.readouts;
  const controls = new Map();
  for (const w of wells) {
    if (w.dose !== 0) continue;
    if (!controls.has(w.plate)) controls.set(w.plate, []);
    controls.get(w.plate).push(w);
  }
  const ctrl = new Map();
  for (const [plate, list] of controls) {
    const m = {};
    for (const day of [5, 7]) for (const f of readouts) m[`${f}|${day}`] = median(list.map((w) => w.values[day][f]));
    ctrl.set(plate, m);
  }
  return wells.map((w) => {
    const c = ctrl.get(w.plate) || {};
    const row = [];
    for (const day of [5, 7]) {
      for (const f of readouts) {
        const v = w.values[day][f];
        const cm = c[`${f}|${day}`] ?? NaN;
        const denom = Math.abs(v) + Math.abs(cm);
        row.push(Math.asinh(v), denom !== 0 && Number.isFinite(denom) ? (v - cm) / denom : 0, Number.isNaN(v) ? 1 : 0);
      }
      const ns = w.values[day]["ns.n"], cns = c[`ns.n|${day}`] ?? NaN;
      row.push(Math.log2((ns + 1) / (cns + 1)));
    }
    const r = w.well.charAt(0).toUpperCase().charCodeAt(0) - 65;
    const col = Number(w.well.slice(1)) - 1;
    const edge = r === 0 || r === manifest.row_max || col === 0 || col === manifest.col_max ? 1 : 0;
    row.push(w.dose === 0 ? 1 : 0, w.dose > 0 ? Math.log10(w.dose) : 0, r / 5, col / 7, edge);
    return row;
  });
}

/* Group day-5/day-7 rows into wells per batch, as neuroforecast_v3_data.well_table does. */
function wellsByBatch(rows) {
  const wells = new Map();
  for (const row of [...rows].sort((a, b) => a.DIV - b.DIV)) {
    const key = `${row.date}|${row.plate}|${row.well}`;
    if (!wells.has(key)) {
      wells.set(key, { date: row.date, plate: row.plate, well: row.well, dose: row.dose, identity: row.identity,
        eligible: row.eligible, trt: row.trt, values: { 5: {}, 7: {} } });
    }
    wells.get(key).values[row.DIV] = row.values;
  }
  const byDate = new Map();
  for (const w of wells.values()) {
    for (const day of [5, 7]) if (!w.values[day] || !Object.keys(w.values[day]).length) w.values[day] = new Proxy({}, { get: () => NaN });
    if (!byDate.has(w.date)) byDate.set(w.date, []);
    byDate.get(w.date).push(w);
  }
  return byDate;
}

/* Forecast every condition in `cases` (from buildFeaturesV2). Returns Map case.key -> {q10, q50, q90, twin}. */
export function twinForecast(rows, cases, model) {
  const manifest = model.manifest;
  const cfg = manifest.config;
  const nCondIn = cfg.n_cond_in;
  const out = new Map();
  for (const [date, wells] of wellsByBatch(rows)) {
    const batchCases = cases.filter((c) => c.date === date);
    if (!batchCases.length) continue;
    const index = new Map(batchCases.map((c, i) => [c.key, i]));
    const plates = [...new Set(wells.map((w) => w.plate))].sort();
    const groups = wells.map((w) => (w.dose > 0 ? (w.identity !== null ? w.identity : "TRT:" + w.trt) : ""));
    const chemIds = new Map([...new Set(groups.filter((g) => g !== ""))].sort().map((g, i) => [g, i]));
    const plate = wells.map((w) => plates.indexOf(w.plate));
    const chem = groups.map((g) => (g === "" ? -1 : chemIds.get(g)));
    const cond = wells.map((w) => {
      if (!(w.dose > 0 && w.eligible && w.identity !== null)) return -1;
      const i = index.get(`${w.date}|${w.identity}|${w.dose}`);
      return i === undefined ? -1 : i;
    });
    const raw = wellFeatures(wells, manifest);
    const nIn = raw[0].length;
    const x = new Float64Array(wells.length * nIn);
    raw.forEach((r, i) => r.forEach((v, k) => {
      const z = (v - model.mu[k]) / model.sd[k];
      x[i * nIn + k] = Number.isFinite(z) ? Math.fround(z) : 0;
    }));
    const condX = batchCases.map((c) => manifest.cond_columns.map((name) => {
      const v = c.features[name];
      return Number.isFinite(v) ? Math.fround(v) : 0;
    }));
    const runs = model.seeds.map((p) => forward(p, cfg, x, nIn, plate, chem, cond, condX, batchCases.length, nCondIn));
    batchCases.forEach((c, i) => {
      if (!runs[0].counts[i]) return;
      const anchor = condX[i][1];
      const avg = (pick) => runs.reduce((s, r) => s + pick(r), 0) / runs.length;
      out.set(c.key, {
        q10: avg((r) => r.quant[i][0]) + anchor, q50: avg((r) => r.quant[i][1]) + anchor, q90: avg((r) => r.quant[i][2]) + anchor,
        twin: runs[0].twin[i].map((_, k) => avg((r) => r.twin[i][k])),
      });
    });
  }
  return out;
}
