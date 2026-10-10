/* Closing the loop after day 12: score a day-7 report against the batch's day-12 recordings, and write the
   admission record a lab files with the data. Mirrors neuroforecast_triage_v2.day12_check and admission_record;
   tests/test_web_day12_check.mjs holds both to the Python tool and to the sealed test's evidence. */

import { DAY12_USABLE_MIN } from "./triage.js";

export const RECORD_SCHEMA = "neuroforecast-admission-record/1";
export const CHECK_DEFINITION = "Day-12 truth is the sealed test's target: log2 of the replicate mean of (network spikes + 1) / "
  + "(same-plate day-12 control median + 1). A condition is scored when its day-12 control median is at least "
  + `${DAY12_USABLE_MIN} network spikes. Batch errors are mean absolute errors averaged within each chemical, then over `
  + "chemicals; provisional and declined errors are plain means over conditions.";
const BATCH_RULE = "reference unusable when at least half the conditions sit on plates whose day-7 control median is below one network spike";

const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : null);
const finiteOrNull = (v) => (Number.isFinite(v) ? v : null);
const byCodeUnit = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

function chemicalAveraged(rows, column) {
  if (!rows.length) return null;
  const byChemical = new Map();
  for (const r of rows) {
    if (!byChemical.has(r.identity)) byChemical.set(r.identity, []);
    byChemical.get(r.identity).push(r[column]);
  }
  return mean([...byChemical.keys()].sort(byCodeUnit).map((k) => mean(byChemical.get(k))));
}

/* Per condition: the measured day 12 beside the forecast and carry-forward. Per batch: the published summaries. */
export function day12Check(conditions, truth) {
  const rows = new Map();
  for (const c of conditions) {
    const t = truth.get(c.key);
    const measured = t ? t.measured : NaN;
    const referenceUsable = t ? t.usable : false;
    const row = {
      key: c.key, date: c.date, identity: c.identity, measured, reference: t ? t.reference : NaN, referenceUsable,
      profile: t ? t.profile : null,
      scored: Number.isFinite(measured) && referenceUsable,
      forecastError: Math.abs(c.forecast - measured),
      carryError: Math.abs(c.persistence - measured),
      inside: measured >= c.low && measured <= c.high,
      provisional: c.usable,
    };
    rows.set(c.key, row);
  }
  const batches = [];
  for (const date of [...new Set(conditions.map((c) => c.date))].sort((a, b) => a - b)) {
    const group = conditions.filter((c) => c.date === date).map((c) => rows.get(c.key));
    const scored = group.filter((r) => r.scored);
    const provisional = scored.filter((r) => r.provisional);
    const declined = scored.filter((r) => !r.provisional);
    batches.push({
      batch: date,
      conditions_in_report: group.length,
      conditions_measured: group.filter((r) => Number.isFinite(r.measured)).length,
      conditions_scored: scored.length,
      reference_unusable: group.filter((r) => Number.isFinite(r.measured) && !r.referenceUsable).length,
      forecast_error: chemicalAveraged(scored, "forecastError"),
      carry_forward_error: chemicalAveraged(scored, "carryError"),
      forecast_closer: scored.filter((r) => r.forecastError < r.carryError).length,
      interval80_coverage: mean(scored.map((r) => (r.inside ? 1 : 0))),
      provisional: { conditions: provisional.length, forecast_error: mean(provisional.map((r) => r.forecastError)),
        carry_forward_error: mean(provisional.map((r) => r.carryError)) },
      declined: { conditions: declined.length, forecast_error: mean(declined.map((r) => r.forecastError)),
        carry_forward_error: mean(declined.map((r) => r.carryError)) },
    });
  }
  return { definition: CHECK_DEFINITION, rows, batches };
}

export async function sha256Hex(bytes) {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/* ------------------------------------------------- the lab's own calibration (mirrors lab_calibration) */

export const MIN_HISTORY_BATCHES = 2;
export const CALIBRATION_METHOD = "split-conformal quantile of |measured - forecast| / difficulty over the history's scored conditions";

/* The ceil((n+1) level)-th smallest score: the split-conformal quantile with the finite-sample correction. */
export function conformalQuantile(scores, level) {
  const s = [...scores].sort((a, b) => a - b);
  const rank = Math.min(s.length, Math.ceil((s.length + 1) * level));
  return s[rank - 1];
}

const sameModels = (a, b) => Boolean(a && b) && typeof a === "object" && Object.keys(a).length === Object.keys(b).length
  && Object.keys(b).every((k) => a[k] === b[k]);

/* Refit the interval scale from a lab's earlier checked admission records ({file, bytes}). Every scored condition
   contributes |measured - forecast| / difficulty. Records must come from these locked models and carry a day-12
   check; a record of a batch being analysed is refused, so a batch never calibrates its own intervals. */
export async function labCalibration({ records, lockedModels, constants, excludeBatches }) {
  const scores = [];
  const batches = new Set();
  const sources = [];
  for (const { file, bytes } of records) {
    let record;
    try {
      record = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    } catch (error) {
      throw new Error(`${file}: not a JSON admission record (${error.message})`);
    }
    if (!record || typeof record !== "object" || record.schema !== RECORD_SCHEMA) throw new Error(`${file}: not a ${RECORD_SCHEMA} record`);
    if (!sameModels(record.locked_models, lockedModels)) {
      throw new Error(`${file}: made with different models; only records from these locked models can recalibrate them`);
    }
    if (!record.day12_check) throw new Error(`${file}: has no day-12 check; check that batch against its day-12 file first`);
    const own = [...new Set(record.conditions.map((c) => Number(c.batch)))].filter((b) => excludeBatches.has(b)).sort((a, b) => a - b);
    if (own.length) throw new Error(`${file}: is a record of batch ${own[0]}, the batch being analysed; a batch cannot calibrate itself`);
    let used = 0;
    for (const c of record.conditions) {
      const check = c.day12 || {};
      const measured = check.measured_log2, forecast = c.forecast_log2, difficulty = c.difficulty;
      if (check.reference_usable && measured != null && forecast != null && difficulty) {
        scores.push(Math.abs(measured - forecast) / difficulty);
        batches.add(Number(c.batch));
        used += 1;
      }
    }
    sources.push({ file, sha256: await sha256Hex(bytes), conditions_used: used });
  }
  if (batches.size < MIN_HISTORY_BATCHES) {
    throw new Error(`Recalibration needs checked records from at least ${MIN_HISTORY_BATCHES} batches; these cover ${batches.size}`);
  }
  const shipped = constants.interval_quantile;
  return {
    method: CALIBRATION_METHOD, level: constants.interval_level, shipped_scale: shipped,
    scale: conformalQuantile(scores, constants.interval_level),
    history_batches: [...batches].sort((a, b) => a - b), history_conditions: scores.length,
    history_coverage_at_shipped_scale: scores.filter((s) => s <= shipped).length / scores.length,
    records: sources,
  };
}

/* Intervals at the lab's scale; forecasts, difficulty and verdicts are unchanged. */
export function withScale(result, scale) {
  const conditions = result.conditions.map((c) => ({ ...c, low: c.forecast - scale * c.difficulty, high: c.forecast + scale * c.difficulty }));
  return { ...result, conditions };
}

function decision(c) {
  const reasons = [];
  if (!c.batchUsable) reasons.push("batch_reference_unusable");
  if (!c.trusted) reasons.push("difficulty_above_threshold");
  if (c.flags.low_reference_activity) reasons.push("low_reference_activity");
  if (c.flags.few_controls) reasons.push("few_controls");
  return [c.usable ? "provisional forecast" : "measure day 12", reasons];
}

/* The record a lab files with the batch. `input` and `check.input` carry {file, bytes}; `calibration` is the
   lab's own interval calibration when one was applied. */
export async function admissionRecord({ result, constants, lockedModels, input, check = null, calibration = null, now = new Date() }) {
  const conditions = [...result.conditions]
    .sort((a, b) => a.date - b.date || byCodeUnit(String(a.name), String(b.name)) || a.dose - b.dose)
    .map((c) => {
      const [verdict, reasons] = decision(c);
      const entry = {
        batch: c.date, chemical: String(c.name), identity: String(c.identity), dose_uM: c.dose,
        day7_log2: finiteOrNull(c.day7), forecast_log2: finiteOrNull(c.forecast),
        interval80_log2: [finiteOrNull(c.low), finiteOrNull(c.high)],
        carry_forward_log2: finiteOrNull(c.persistence), difficulty: finiteOrNull(c.difficulty),
        decision: verdict, reasons,
      };
      if (check) {
        const r = check.result.rows.get(c.key);
        entry.day12 = {
          measured_log2: finiteOrNull(r.measured), reference_usable: r.referenceUsable,
          forecast_abs_error: finiteOrNull(r.forecastError), carry_forward_abs_error: finiteOrNull(r.carryError),
          inside_interval80: Number.isFinite(r.measured) ? r.inside : null,
        };
      }
      return entry;
    });
  const batches = result.batches.map((b) => {
    const group = result.conditions.filter((c) => c.date === b.date);
    return {
      batch: b.date, verdict: b.usable ? "usable" : "reference unusable", reasons: b.usable ? [] : ["low_reference_activity"],
      conditions: b.conditions, chemicals: b.chemicals, declined: b.declined,
      provisional: group.filter((c) => c.usable).length,
      day7_reference_network_spikes: finiteOrNull(b.referenceActivity), low_reference_fraction: finiteOrNull(b.lowReferenceFraction),
    };
  });
  const record = {
    schema: RECORD_SCHEMA,
    created_utc: now.toISOString().replace(/\.\d{3}Z$/, "Z"),
    tool: { name: "NeuroForecast", model: "v2", implementation: "browser" },
    locked_models: lockedModels,
    input: { file: input.file, sha256: await sha256Hex(input.bytes), days: [5, 7] },
    decision_rule: {
      trust_threshold: constants.trust_threshold, interval_level: constants.interval_level,
      interval_scale: calibration ? calibration.scale : constants.interval_quantile, batch_rule: BATCH_RULE,
    },
    interval_calibration: calibration,
    batches,
    conditions,
    day12_check: null,
    evidence: constants.evidence,
    limits: constants.limits,
  };
  if (check) {
    record.day12_check = {
      input: { file: check.input.file, sha256: await sha256Hex(check.input.bytes), days: [12] },
      definition: check.result.definition,
      batches: check.result.batches,
    };
  }
  return record;
}
