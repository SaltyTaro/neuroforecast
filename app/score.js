/* The validation harness's scorer in the browser: mirrors neuroforecast_harness.score on the reference split
   (batches and chemicals held out, the v3 outer folds). Point metrics, batch wins and coverage are exact;
   the paired bootstrap uses its own random stream, so its interval can differ from Python's in the third decimal.
   tests/test_web_score.mjs holds it to the Python scorer. */

import { parseCSV, quantile } from "./triage.js";

export const RESAMPLES = 10000;
export const SEED = 20261008;
const BASELINES = ["carry_forward", "calibrated_day7", "v2", "v2_plus_transformer"];

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const num = (s) => (s === "" || s === undefined ? NaN : Number(s));

/* The benchmark: usable conditions of the 41 batches with their reference forecasts, keyed by case_id. */
export function loadBenchmark(casesText, referenceText) {
  const reference = new Map(parseCSV(referenceText).rows.map((r) => [r.case_id, r]));
  const cases = [];
  for (const r of parseCSV(casesText).rows) {
    if (r.usable !== "True") continue;
    const ref = reference.get(r.case_id);
    if (!ref) throw new Error(`reference forecasts miss ${r.case_id}`);
    const row = { case_id: r.case_id, date: Number(r.date), identity: r.identity, target: num(r.target) };
    for (const b of BASELINES) row[b] = num(ref[b]);
    cases.push(row);
  }
  return cases;
}

/* Per-batch loss: absolute error averaged over doses within a chemical, then over chemicals within the batch. */
export function dateLosses(rows, key) {
  const byBatch = new Map();
  for (const r of rows) {
    if (!byBatch.has(r.date)) byBatch.set(r.date, new Map());
    const chem = byBatch.get(r.date);
    if (!chem.has(r.identity)) chem.set(r.identity, []);
    chem.get(r.identity).push(Math.abs(r[key] - r.target));
  }
  const out = new Map();
  for (const date of [...byBatch.keys()].sort((a, b) => a - b)) {
    out.set(date, mean([...byBatch.get(date).values()].map(mean)));
  }
  return out;
}

/* A small, seeded generator (mulberry32) for the bootstrap. */
function generator(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* Batch-paired bootstrap of mean(a) - mean(b): 95% interval and how many batches a is better on. */
export function paired(a, b, seed = SEED, n = RESAMPLES) {
  const dates = [...a.keys()].filter((d) => b.has(d));
  const d = dates.map((k) => a.get(k) - b.get(k));
  const random = generator(seed);
  const boot = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    let s = 0;
    for (let j = 0; j < d.length; j += 1) s += d[Math.floor(random() * d.length)];
    boot[i] = s / d.length;
  }
  return { difference: mean(d), low: quantile([...boot], 0.025), high: quantile([...boot], 0.975),
    batches_better: d.filter((x) => x < 0).length, batches: d.length };
}

function coverage(rows, level) {
  const perBatch = new Map();
  for (const r of rows) {
    if (!perBatch.has(r.date)) perBatch.set(r.date, []);
    perBatch.get(r.date).push(r.target >= r.low && r.target <= r.high ? 1 : 0);
  }
  const batch = [...perBatch.values()].map(mean).sort((x, y) => x - y);
  const worst = batch.slice(0, Math.max(1, Math.floor(batch.length / 5)));
  return {
    nominal: level, pooled: mean(rows.map((r) => (r.target >= r.low && r.target <= r.high ? 1 : 0))),
    batch_mean: mean(batch), worst_fifth: mean(worst),
    "share_of_batches_within_0.1": batch.filter((c) => c >= level - 0.1).length / batch.length,
    mean_width: mean(rows.map((r) => r.high - r.low)),
  };
}

/* Score a predictions CSV (case_id, prediction, optional low/high, optional decline) for every usable condition. */
export function scoreForecast(cases, predictionsText, level = 0.8) {
  const parsed = parseCSV(predictionsText);
  if (!parsed.header.includes("case_id") || !parsed.header.includes("prediction")) {
    throw new Error("The predictions file needs case_id and prediction columns.");
  }
  const given = new Map();
  for (const r of parsed.rows) {
    if (given.has(r.case_id)) throw new Error(`The predictions repeat a case_id (${r.case_id}).`);
    given.set(r.case_id, r);
  }
  const hasInterval = parsed.header.includes("low") && parsed.header.includes("high");
  const hasDecline = parsed.header.includes("decline");
  const rows = [];
  let missing = 0;
  for (const c of cases) {
    const p = given.get(c.case_id);
    const prediction = p ? num(p.prediction) : NaN;
    if (!Number.isFinite(prediction)) { missing += 1; continue; }
    const row = { ...c, prediction };
    if (hasInterval) {
      row.low = num(p.low);
      row.high = num(p.high);
      if (!Number.isFinite(row.low) || !Number.isFinite(row.high)) throw new Error("low and high must be finite for every usable condition.");
    }
    if (hasDecline) row.decline = ["1", "true", "True", "TRUE"].includes(String(p.decline).trim());
    rows.push(row);
  }
  if (missing) throw new Error(`${missing} of ${cases.length} usable conditions have no finite prediction.`);
  const losses = dateLosses(rows, "prediction");
  const values = [...losses.values()];
  const result = {
    design: "batch_chemical", conditions: rows.length, batches: losses.size, chemicals: new Set(rows.map((r) => r.identity)).size,
    date_macro_mae: mean(values), pooled_mae: mean(rows.map((r) => Math.abs(r.prediction - r.target))),
    worst_batch_mae: Math.max(...values), against: {}, per_batch: Object.fromEntries(losses),
  };
  for (const b of BASELINES) {
    const base = dateLosses(rows, b);
    result.against[b] = { date_macro_mae: mean([...base.values()]), ...paired(losses, base), per_batch: Object.fromEntries(base) };
  }
  if (hasInterval) result.intervals = coverage(rows, level);
  if (hasDecline) {
    const error = rows.map((r) => Math.abs(r.prediction - r.target));
    const kept = rows.filter((r) => !r.decline), declined = rows.filter((r) => r.decline);
    const keptError = kept.map((r) => Math.abs(r.prediction - r.target));
    result.refusal = {
      declined: declined.length, share_kept: kept.length / rows.length, kept_pooled_mae: mean(keptError),
      declined_pooled_mae: declined.length ? mean(declined.map((r) => Math.abs(r.prediction - r.target))) : null,
      kept_date_macro_mae: mean([...dateLosses(kept, "prediction").values()]),
      error_reduction_from_declining: 1 - mean(keptError) / mean(error),
    };
  }
  return result;
}
