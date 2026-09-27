/* NeuroForecast trust triage - browser port of tools/neuroforecast_triage.py.
 *
 * Pure computation, no DOM. Mirrors the audited Python pipeline exactly: plate-control
 * aggregation, control-relative contrasts, day-7 reliability indicators, forest inference,
 * conformal intervals and the locked trust threshold. Verified against the Python outputs
 * by tests/test_web_triage.mjs, which fails the build on any disagreement above 1e-6.
 */

const LEAF = -2;
const DECISION_DAY = 7;
const NUMERIC = new Set(["DIV", "dose", "date"]);

/* ---------------------------------------------------------------- statistics
 * pandas-compatible: NaN is skipped, quantiles use linear interpolation. */

function finite(values) {
  const out = [];
  for (const v of values) if (Number.isFinite(v)) out.push(v);
  return out;
}

export function quantile(values, q) {
  const x = finite(values).sort((a, b) => a - b);
  if (!x.length) return NaN;
  const pos = q * (x.length - 1);
  const lower = Math.floor(pos);
  const frac = pos - lower;
  return frac === 0 ? x[lower] : x[lower] + frac * (x[lower + 1] - x[lower]);
}

export const median = (values) => quantile(values, 0.5);

export function mean(values) {
  const x = finite(values);
  return x.length ? x.reduce((a, b) => a + b, 0) / x.length : NaN;
}

function stdev(values) {
  const x = finite(values);
  if (x.length < 2) return NaN;
  const m = x.reduce((a, b) => a + b, 0) / x.length;
  return Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / (x.length - 1));
}

const countFinite = (values) => finite(values).length;
const fractionEqual = (values, target) => {
  const x = finite(values);
  return x.length ? x.filter((v) => v === target).length / x.length : NaN;
};
const fractionMissing = (values) => {
  let missing = 0;
  for (const v of values) if (!Number.isFinite(v)) missing += 1;
  return values.length ? missing / values.length : NaN;
};

/* Python's f"{x:.12g}", used to build case identifiers that match the saved tables. */
export function g12(x) {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  const exponent = Math.floor(Math.log10(Math.abs(x)));
  if (exponent < -4 || exponent >= 12) {
    return x.toExponential(11).replace(/\.?0+e/, "e").replace(/e([+-])(\d)$/, "e$10$2");
  }
  let s = x.toPrecision(12);
  if (s.includes(".")) s = s.replace(/\.?0+$/, "");
  return s;
}

/* ---------------------------------------------------------------- csv */

export function parseCSV(text) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n").filter((line) => line.length);
  if (!lines.length) throw new Error("The file is empty.");
  const header = splitRow(lines[0]);
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cells = splitRow(lines[i]);
    if (cells.length === 1 && cells[0] === "") continue;
    const row = {};
    for (let c = 0; c < header.length; c += 1) {
      const key = header[c];
      const raw = cells[c] === undefined ? "" : cells[c];
      row[key] = raw;
    }
    rows.push(row);
  }
  return { header, rows };
}

function splitRow(line) {
  const cells = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"') {
        if (line[i + 1] === '"') { current += '"'; i += 1; } else quoted = false;
      } else current += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { cells.push(current); current = ""; }
    else current += ch;
  }
  cells.push(current);
  return cells.map((c) => c.trim());
}

const asNumber = (value) => {
  if (value === "" || value === "NA" || value === "NaN" || value === undefined) return NaN;
  const n = Number(value);
  return Number.isNaN(n) ? NaN : n;
};

const asBool = (value) => {
  const v = String(value).trim().toLowerCase();
  if (v === "true" || v === "1") return true;
  if (v === "false" || v === "0") return false;
  throw new Error(`eligible_identity must be audited true/false values, found "${value}".`);
};

/* ---------------------------------------------------------------- inputs */

export function prepareRows(parsed, readouts) {
  const required = ["date", "Plate.SN", "well", "DIV", "dose", "units", "trt", "identity", "eligible_identity"];
  for (const column of required) {
    if (!parsed.header.includes(column)) throw new Error(`The file is missing the "${column}" column. Export audited well-level rows, not a summary table.`);
  }
  const rows = parsed.rows.map((raw) => {
    const row = {
      date: asNumber(raw.date),
      plate: raw["Plate.SN"],
      well: raw.well,
      DIV: asNumber(raw.DIV),
      dose: asNumber(raw.dose),
      units: raw.units,
      trt: raw.trt,
      identity: raw.identity === "" ? null : raw.identity,
      eligible: asBool(raw.eligible_identity),
      values: {},
    };
    for (const field of readouts) row.values[field] = asNumber(raw[field]);
    return row;
  });

  const days = new Set(rows.map((r) => r.DIV));
  const later = [...days].filter((d) => d > DECISION_DAY).sort((a, b) => a - b);
  if (later.length) throw new Error(`This tool decides at day ${DECISION_DAY}. The file contains day ${later.join(" and ")} recordings; remove them so the forecast cannot use information the experiment would not have yet.`);
  if (!days.has(DECISION_DAY)) throw new Error("No day-7 recordings found. Day 7 is the decision point.");
  const units = new Set(rows.map((r) => r.units));
  if (units.size !== 1 || !units.has("uM")) throw new Error(`Expected concentrations in uM, found ${[...units].join(", ")}.`);
  const seen = new Set();
  for (const r of rows) {
    const key = `${r.date}|${r.plate}|${r.well}|${r.DIV}`;
    if (seen.has(key)) throw new Error(`Duplicate recording for plate ${r.plate} well ${r.well} on day ${r.DIV}. Audit duplicates before triage.`);
    seen.add(key);
  }
  return { rows, hasDay5: days.has(5) };
}

/* Mirrors neuroforecast_data._observed_day: plate controls, then case aggregation. */
function observedDay(rows, day, constants) {
  const readouts = constants.readouts;
  const current = rows.filter((r) => r.DIV === day);
  const plateControls = new Map();
  for (const row of current) {
    if (row.dose !== 0) continue;
    const key = `${row.date}|${row.plate}`;
    if (!plateControls.has(key)) plateControls.set(key, []);
    plateControls.get(key).push(row);
  }
  const controlMedian = new Map();
  for (const [key, wells] of plateControls) {
    const summary = {};
    for (const field of readouts) summary[field] = median(wells.map((w) => w.values[field]));
    controlMedian.set(key, summary);
  }

  const cases = new Map();
  for (const row of current) {
    if (!row.eligible || !(row.dose > 0)) continue;
    const control = controlMedian.get(`${row.date}|${row.plate}`);
    const key = `${row.date}|${row.identity}|${g12(row.dose)}`;
    if (!cases.has(key)) cases.set(key, { key, date: row.date, identity: row.identity, dose: row.dose, trt: row.trt, wells: [], plates: new Set() });
    const entry = cases.get(key);
    entry.wells.push({ row, control });
    entry.plates.add(row.plate);
  }

  const out = new Map();
  for (const [key, entry] of cases) {
    const relative = entry.wells.map(({ row, control }) => {
      const c = control ? control["ns.n"] : NaN;
      return (row.values["ns.n"] + constants.pseudocount) / (c + constants.pseudocount);
    });
    if (entry.wells.length < constants.min_replicates) continue;
    if (countFinite(relative) < constants.min_replicates) continue;
    const summary = { key, date: entry.date, identity: entry.identity, dose: entry.dose, trt: entry.trt, replicates: entry.wells.length, plates: [...entry.plates], raw: {}, control: {}, missing: {} };
    for (const field of constants.readouts) {
      summary.raw[field] = mean(entry.wells.map(({ row }) => row.values[field]));
      summary.control[field] = mean(entry.wells.map(({ control }) => (control ? control[field] : NaN)));
      summary.missing[field] = fractionMissing(entry.wells.map(({ row }) => row.values[field]));
    }
    summary.normalized_ns = Math.log2(mean(relative));
    out.set(key, summary);
  }
  return out;
}

function contrast(raw, control) {
  const denominator = Math.abs(raw) + Math.abs(control);
  if (denominator === 0) return 0;
  return (raw - control) / denominator;
}

/* Mirrors neuroforecast_gate.reliability_indicators: day-7 replicate and plate-control quality. */
function reliabilityIndicators(rows, constants) {
  const day = rows.filter((r) => r.DIV === DECISION_DAY);
  const plates = new Map();
  for (const row of day) {
    if (row.dose !== 0) continue;
    const key = `${row.date}|${row.plate}`;
    if (!plates.has(key)) plates.set(key, []);
    plates.get(key).push(row);
  }
  const plateStats = new Map();
  for (const [key, wells] of plates) {
    const ns = wells.map((w) => w.values["ns.n"]);
    const nae = wells.map((w) => w.values["nAE"]);
    const med = median(ns);
    plateStats.set(key, {
      count: countFinite(ns),
      medianRaw: med,
      arcsinhMedian: Math.asinh(med),
      zeroFraction: fractionEqual(ns, 0),
      iqrRatio: (quantile(ns, 0.75) - quantile(ns, 0.25)) / (med + 1),
      silentFraction: fractionEqual(nae, 0),
    });
  }

  const cases = new Map();
  for (const row of day) {
    if (!row.eligible || !(row.dose > 0)) continue;
    const stats = plateStats.get(`${row.date}|${row.plate}`);
    const control = stats ? stats.medianRaw : NaN;
    const key = `${row.date}|${row.identity}|${g12(row.dose)}`;
    if (!cases.has(key)) cases.set(key, { relative: [], stats: [] });
    const entry = cases.get(key);
    entry.relative.push(Math.log2((row.values["ns.n"] + constants.pseudocount) / (control + constants.pseudocount)));
    entry.stats.push(stats);
  }

  const out = new Map();
  for (const [key, entry] of cases) {
    const pick = (field) => mean(entry.stats.map((s) => (s ? s[field] : NaN)));
    out.set(key, {
      "rel__rep_count_d7": entry.relative.length,
      "rel__rep_sd_log2_ns_d7": stdev(entry.relative),
      "rel__plate_ctrl_count_d7": pick("count"),
      "rel__plate_ctrl_median_ns_d7": pick("arcsinhMedian"),
      "rel__plate_ctrl_zero_ns_frac_d7": pick("zeroFraction"),
      "rel__plate_ctrl_iqr_ratio_d7": pick("iqrRatio"),
      "rel__plate_ctrl_silent_frac_d7": pick("silentFraction"),
      referenceActivity: mean(entry.stats.map((s) => (s ? s.medianRaw : NaN))),
    });
  }
  return out;
}

export function buildFeatures(rows, constants) {
  const day7 = observedDay(rows, DECISION_DAY, constants);
  const day5 = observedDay(rows, 5, constants);
  const reliability = reliabilityIndicators(rows, constants);
  const keys = [...day7.keys()].sort((a, b) => {
    const A = day7.get(a); const B = day7.get(b);
    return A.date - B.date || String(A.identity).localeCompare(String(B.identity)) || A.dose - B.dose;
  });

  const cases = [];
  for (const key of keys) {
    const seven = day7.get(key);
    const five = day5.get(key);
    const rel = reliability.get(key) || {};
    const features = {
      normalized_ns_d7: seven.normalized_ns,
      log10_dose: Math.log10(seven.dose),
    };
    for (const field of constants.readouts) {
      features[`contrast__${field}__d7`] = contrast(seven.raw[field], seven.control[field]);
      features[`missing__${field}__d7`] = seven.missing[field];
    }
    if (five) {
      features.normalized_ns_d5 = five.normalized_ns;
      for (const field of constants.readouts) {
        features[`contrast__${field}__d5`] = contrast(five.raw[field], five.control[field]);
        features[`missing__${field}__d5`] = five.missing[field];
      }
    }
    for (const column of constants.reliability_columns) features[column] = rel[column];
    for (const name of Object.keys(features)) {
      if (!Number.isFinite(features[name])) features[name] = NaN;
    }
    cases.push({
      key, date: seven.date, identity: seven.identity, name: seven.trt, dose: seven.dose,
      replicates: seven.replicates, plates: seven.plates,
      referenceActivity: rel.referenceActivity, features,
      hasDay5: Boolean(five),
    });
  }
  return cases;
}

/* ---------------------------------------------------------------- forest */

export class Forest {
  constructor(buffer, spec) {
    const nodes = spec.nodes;
    const trees = spec.trees;
    const features = spec.columns.length;
    let at = 0;
    const take = (Type, count) => {
      const bytes = count * Type.BYTES_PER_ELEMENT;
      const view = new Type(buffer.slice(at, at + bytes));
      at += bytes;
      return view;
    };
    this.feature = take(Int16Array, nodes);
    this.threshold = take(Float32Array, nodes);
    this.left = take(Int32Array, nodes);
    this.right = take(Int32Array, nodes);
    this.value = take(Float32Array, nodes);
    this.offsets = take(Int32Array, trees + 1);
    this.medians = take(Float32Array, features);
    this.columns = spec.columns;
    if (at !== buffer.byteLength) throw new Error(`Model file length mismatch: read ${at} of ${buffer.byteLength} bytes.`);
  }

  vector(features) {
    const x = new Float64Array(this.columns.length);
    for (let i = 0; i < this.columns.length; i += 1) {
      const value = features[this.columns[i]];
      x[i] = Number.isFinite(value) ? value : this.medians[i];
    }
    return x;
  }

  predict(features) {
    const x = this.vector(features);
    const trees = this.offsets.length - 1;
    let total = 0;
    for (let t = 0; t < trees; t += 1) {
      let node = this.offsets[t];
      while (this.feature[node] !== LEAF) {
        node = x[this.feature[node]] <= this.threshold[node] ? this.left[node] : this.right[node];
      }
      total += this.value[node];
    }
    return total / trees;
  }
}

/* ---------------------------------------------------------------- triage */

function qualityFlags(features, thresholds) {
  return {
    low_reference_activity: Math.sinh(features["rel__plate_ctrl_median_ns_d7"]) < 1,
    few_controls: features["rel__plate_ctrl_count_d7"] < 4,
    replicates_disagree: features["rel__rep_sd_log2_ns_d7"] > thresholds.rep_sd_log2_ns_d7_p90,
    controls_disagree: features["rel__plate_ctrl_iqr_ratio_d7"] > thresholds.plate_ctrl_iqr_ratio_d7_p90,
  };
}

function rankWithin(conditions, subset, budget, prefix) {
  const byBatch = new Map();
  for (const condition of subset) {
    if (!byBatch.has(condition.date)) byBatch.set(condition.date, []);
    byBatch.get(condition.date).push(condition);
  }
  for (const group of byBatch.values()) {
    const ordered = [...group].sort((a, b) => Math.abs(b.forecast) - Math.abs(a.forecast));
    const keep = Math.ceil(budget * ordered.length);
    ordered.forEach((condition, index) => {
      condition[`${prefix}Rank`] = index + 1;
      condition[`${prefix}Flagged`] = index < keep;
    });
  }
  for (const condition of conditions) {
    if (condition[`${prefix}Rank`] === undefined) {
      condition[`${prefix}Rank`] = null;
      condition[`${prefix}Flagged`] = false;
    }
  }
}

export function triage(cases, models, constants) {
  const conditions = cases.map((entry) => {
    const forecast = models.forecast.predict(entry.features);
    const difficulty = Math.max(models.difficulty.predict(entry.features), constants.sigma_floor);
    const half = constants.scaled_quantile * difficulty;
    const trusted = difficulty <= constants.sigma_threshold;
    const comparator = entry.hasDay5 ? models.comparator.predict(entry.features) : null;
    return {
      ...entry,
      forecast,
      low: forecast - half,
      high: forecast + half,
      difficulty,
      trusted,
      verdict: trusted ? "forecast usable" : "declined: measure day 12 directly",
      persistence: entry.features.normalized_ns_d7,
      comparator,
      day7: entry.features.normalized_ns_d7,
      quiet: Math.abs(entry.features.normalized_ns_d7) < constants.quiet_limit,
      flags: qualityFlags(entry.features, constants.flag_thresholds),
    };
  });
  for (const condition of conditions) condition.anyFlag = Object.values(condition.flags).some(Boolean);

  rankWithin(conditions, conditions, constants.budget, "review");
  rankWithin(conditions, conditions.filter((c) => c.quiet), constants.budget, "earlyWarning");

  const batches = [];
  const dates = [...new Set(conditions.map((c) => c.date))].sort((a, b) => a - b);
  for (const date of dates) {
    const group = conditions.filter((c) => c.date === date);
    const lowFraction = group.filter((c) => c.flags.low_reference_activity).length / group.length;
    batches.push({
      date,
      conditions: group.length,
      plates: [...new Set(group.flatMap((c) => c.plates))].sort(),
      referenceActivity: median(group.map((c) => c.referenceActivity)),
      flagged: group.filter((c) => c.anyFlag).length,
      declined: group.filter((c) => !c.trusted).length,
      quiet: group.filter((c) => c.quiet).length,
      reviewList: group.filter((c) => c.reviewFlagged).length,
      earlyWarningList: group.filter((c) => c.earlyWarningFlagged).length,
      lowReferenceFraction: lowFraction,
      usable: lowFraction < 0.5,
      verdict: lowFraction >= 0.5
        ? "Day-7 reference activity is too low for this batch. Forecasts here are unreliable; measure day 12 directly."
        : "Day-7 measurement quality is usable.",
    });
  }
  return { conditions, batches };
}

/* ================================================================ v2
 *
 * Port of tools/neuroforecast_triage_v2.py, whose features come from neuroforecast_v2.features_from
 * (the frozen v1 builders with 17 readouts, the gate's day-7 reliability indicators, and three
 * per-batch covariates). Each statistic follows the pandas or numpy routine the Python calls, in the
 * same row order, and the forests below reproduce scikit-learn's arithmetic exactly. Verified by
 * tests/test_web_triage_v2.mjs against evaluation/v2_reference, failing above 1e-6.
 */

const V2_DAYS = [5, 7];
const CHANGED_LIMIT = 0.5;       // |day-7 log2 change| counted as "changed" by batch_changed_frac_d7
const LOW_REFERENCE = 1;         // day-7 plate control median below one network spike
const FEW_CONTROLS = 4;
const LOW_BATCH_FRACTION = 0.5;
const PANDAS_NA = new Set(["", "NA", "NaN", "nan", "-NaN", "-nan", "N/A", "n/a", "NULL", "null", "<NA>", "#N/A", "#NA", "None"]);

export const V2_VERDICTS = {
  usable: "forecast usable",
  declined: "declined: measure day 12 directly",
  batchUnusable: "batch reference unusable: measure day 12 directly",
};
export const V2_BATCH_VERDICTS = {
  low: "day-7 reference activity is too low for this batch; forecasts here are unreliable and a day-12 measurement is recommended",
  usable: "day-7 measurement quality is usable",
};

/* ---------------------------------------------------------------- pandas / numpy statistics */

const notNaN = (values) => values.filter((v) => !Number.isNaN(v));

/* groupby mean: compensated (Kahan) summation in row order, NaN skipped. */
function pdMean(values) {
  let sum = 0, compensation = 0, n = 0;
  for (const v of values) {
    if (Number.isNaN(v)) continue;
    n += 1;
    const y = v - compensation;
    const t = sum + y;
    compensation = t - sum - y;
    if (Number.isNaN(compensation)) compensation = 0;
    sum = t;
  }
  return n ? sum / n : NaN;
}

/* groupby std(ddof=1): Welford's online update in row order, NaN skipped. */
function pdStd(values) {
  let n = 0, mean = 0, m2 = 0;
  for (const v of values) {
    if (Number.isNaN(v)) continue;
    n += 1;
    const old = mean;
    mean += (v - old) / n;
    m2 += (v - mean) * (v - old);
  }
  return n <= 1 ? NaN : Math.sqrt(m2 / (n - 1));
}

/* median: NaN skipped, the two middle values averaged as (a + b) / 2 when the count is even. */
function pdMedian(values) {
  const x = notNaN(values).sort((a, b) => a - b);
  const n = x.length;
  if (!n) return NaN;
  return n % 2 ? x[(n - 1) / 2] : (x[n / 2 - 1] + x[n / 2]) / 2;
}

/* numpy percentile, method "linear": virtual index (n - 1) q and numpy's two-sided lerp. */
function npQuantile(values, q) {
  const x = notNaN(values).sort((a, b) => a - b);
  const n = x.length;
  if (!n) return NaN;
  const virtual = (n - 1) * q;
  if (virtual >= n - 1) return x[n - 1];
  const lower = Math.floor(virtual);
  const gamma = virtual - lower;
  const a = x[lower], b = x[lower + 1];
  const diff = b - a;
  return gamma >= 0.5 ? b - diff * (1 - gamma) : a + diff * gamma;
}

/* (x.dropna() == target).mean(), NaN when every value is missing. */
function fractionOf(values, target) {
  const x = notNaN(values);
  return x.length ? x.filter((v) => v === target).length / x.length : NaN;
}

const missingFraction = (values) => values.filter((v) => Number.isNaN(v)).length / values.length;
const noInfinity = (v) => (v === Infinity || v === -Infinity ? NaN : v);
const byCodeUnit = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

function push(map, key, value) {
  const list = map.get(key);
  if (list) list.push(value); else map.set(key, [value]);
}

/* ---------------------------------------------------------------- v2 inputs */

function asInteger(value, column) {
  const n = asNumber(value);
  if (!Number.isInteger(n)) throw new Error(`Column "${column}" must hold whole numbers; found "${value}".`);
  return n;
}

/* Mirrors neuroforecast_triage_v2.normalize: days 5 and 7 only, both required. */
export function prepareRowsV2(parsed, readouts) {
  const required = ["date", "Plate.SN", "well", "DIV", "trt", "dose", "units", ...readouts];
  const absent = required.filter((column) => !parsed.header.includes(column));
  if (absent.length) {
    throw new Error(`The file is missing ${absent.length === 1 ? "the column" : "the columns"} ${absent.map((c) => `"${c}"`).join(", ")}. `
      + "Export audited well-level rows with every v2 readout, not a summary table.");
  }
  const hasIdentity = parsed.header.includes("identity");
  const hasEligible = parsed.header.includes("eligible_identity");
  const rows = parsed.rows.map((raw) => {
    const trt = PANDAS_NA.has(raw.trt) ? "nan" : raw.trt;
    const identity = hasIdentity ? (PANDAS_NA.has(raw.identity) ? null : raw.identity) : trt;
    const row = {
      date: asInteger(raw.date, "date"),
      plate: raw["Plate.SN"],
      well: raw.well,
      DIV: asInteger(raw.DIV, "DIV"),
      dose: asNumber(raw.dose),
      units: raw.units,
      trt,
      identity,
      eligible: hasEligible ? asBool(raw.eligible_identity) : identity !== null,
      values: {},
    };
    for (const field of readouts) row.values[field] = asNumber(raw[field]);
    return row;
  });
  if (!rows.length) throw new Error("The file has no recordings.");

  const days = new Set(rows.map((r) => r.DIV));
  const other = [...days].filter((d) => !V2_DAYS.includes(d)).sort((a, b) => a - b);
  if (other.length) {
    throw new Error(`This tool decides at day 7 and reads only day-5 and day-7 recordings. The file contains day ${other.join(" and ")} `
      + "recordings; remove them so the forecast cannot use information the experiment would not have yet.");
  }
  if (!days.has(5) || !days.has(7)) {
    throw new Error(`The v2 model needs both day-5 and day-7 recordings; this file has day-${[...days][0]} recordings only.`);
  }
  const units = new Set(rows.map((r) => r.units));
  if (units.size !== 1 || !units.has("uM")) throw new Error(`Expected concentrations in uM, found ${[...units].join(", ")}.`);
  const seen = new Set();
  for (const r of rows) {
    const key = `${r.date}|${r.plate}|${r.well}|${r.DIV}`;
    if (seen.has(key)) throw new Error(`Duplicate recording for plate ${r.plate} well ${r.well} on day ${r.DIV}. Audit duplicates before triage.`);
    seen.add(key);
  }
  return { rows };
}

const plateKey = (row) => `${row.date}|${row.plate}`;
const caseKey = (row) => `${row.date}|${row.identity}|${row.dose}`;
const treated = (row) => row.eligible && row.dose > 0 && row.identity !== null;

/* neuroforecast_data._observed_day: same-plate control medians, then per-case means. */
function observedDayV2(rows, day, constants) {
  const readouts = constants.readouts;
  const pseudo = constants.pseudocount;
  const current = rows.filter((r) => r.DIV === day);
  const controlWells = new Map();
  for (const row of current) if (row.dose === 0) push(controlWells, plateKey(row), row);
  const control = new Map();
  for (const [key, wells] of controlWells) {
    const summary = {};
    for (const field of readouts) summary[field] = pdMedian(wells.map((w) => w.values[field]));
    control.set(key, summary);
  }

  const cases = new Map();
  for (const row of current) if (treated(row)) push(cases, caseKey(row), row);
  const out = new Map();
  for (const [key, wells] of cases) {
    const controls = wells.map((w) => control.get(plateKey(w)));
    const relative = wells.map((w, i) => (w.values["ns.n"] + pseudo) / ((controls[i] ? controls[i]["ns.n"] : NaN) + pseudo));
    if (wells.length < constants.min_replicates) continue;
    if (notNaN(relative).length < constants.min_replicates) continue;
    const summary = { date: wells[0].date, identity: wells[0].identity, dose: wells[0].dose, raw: {}, control: {}, missing: {} };
    for (const field of readouts) {
      summary.raw[field] = pdMean(wells.map((w) => w.values[field]));
      summary.control[field] = pdMean(controls.map((c) => (c ? c[field] : NaN)));
      summary.missing[field] = missingFraction(wells.map((w) => w.values[field]));
    }
    summary.normalized = Math.log2(pdMean(relative));
    out.set(key, summary);
  }
  return out;
}

/* neuroforecast_gate.reliability_indicators: day-7 replicate and plate-control quality. */
function reliabilityV2(rows, constants) {
  const pseudo = constants.pseudocount;
  const day = rows.filter((r) => r.DIV === 7);
  const controlWells = new Map();
  for (const row of day) if (row.dose === 0) push(controlWells, plateKey(row), row);
  const plates = new Map();
  for (const [key, wells] of controlWells) {
    const ns = wells.map((w) => w.values["ns.n"]);
    const med = pdMedian(ns);
    plates.set(key, {
      median: med,
      rel__plate_ctrl_count_d7: notNaN(ns).length,
      rel__plate_ctrl_median_ns_d7: Math.asinh(med),
      rel__plate_ctrl_zero_ns_frac_d7: fractionOf(ns, 0),
      rel__plate_ctrl_iqr_ratio_d7: (npQuantile(ns, 0.75) - npQuantile(ns, 0.25)) / (med + 1),
      rel__plate_ctrl_silent_frac_d7: fractionOf(wells.map((w) => w.values.nAE), 0),
    });
  }
  const cases = new Map();
  for (const row of day) if (treated(row)) push(cases, caseKey(row), row);
  const out = new Map();
  const plateColumns = ["rel__plate_ctrl_count_d7", "rel__plate_ctrl_median_ns_d7", "rel__plate_ctrl_zero_ns_frac_d7",
    "rel__plate_ctrl_iqr_ratio_d7", "rel__plate_ctrl_silent_frac_d7"];
  for (const [key, wells] of cases) {
    const stats = wells.map((w) => plates.get(plateKey(w)));
    const relative = wells.map((w, i) => Math.log2((w.values["ns.n"] + pseudo) / ((stats[i] ? stats[i].median : NaN) + pseudo)));
    const entry = { rel__rep_count_d7: wells.length, rel__rep_sd_log2_ns_d7: pdStd(relative) };
    for (const column of plateColumns) entry[column] = pdMean(stats.map((s) => (s ? s[column] : NaN)));
    out.set(key, entry);
  }
  return out;
}

function contrastV2(raw, control) {
  const denominator = Math.abs(raw) + Math.abs(control);
  return denominator === 0 ? 0 : (raw - control) / denominator;
}

/* neuroforecast_v2.features_from: build_inputs -> representation, joined with reliability and batch covariates. */
export function buildFeaturesV2(rows, constants) {
  const readouts = constants.readouts;
  const days = { 5: observedDayV2(rows, 5, constants), 7: observedDayV2(rows, 7, constants) };
  const reliability = reliabilityV2(rows, constants);

  // build_inputs: inner join of the two days, finite day-5 and day-7 references only, sorted by case key.
  const kept = [];
  for (const [key, seven] of days[7]) {
    const five = days[5].get(key);
    if (!five || !Number.isFinite(five.normalized) || !Number.isFinite(seven.normalized)) continue;
    kept.push({ key, date: seven.date, identity: seven.identity, dose: seven.dose, five, seven });
  }
  kept.sort((a, b) => a.date - b.date || byCodeUnit(a.identity, b.identity) || a.dose - b.dose);

  const names = new Map();
  const platesOf = new Map();
  for (const row of rows) {
    if (row.identity !== null && (!names.has(row.identity) || row.trt < names.get(row.identity))) names.set(row.identity, row.trt);
    if (row.DIV === 7 && treated(row)) {
      if (!platesOf.has(caseKey(row))) platesOf.set(caseKey(row), new Set());
      platesOf.get(caseKey(row)).add(row.plate);
    }
  }

  const cases = kept.map((entry) => {
    const features = {
      normalized_ns_d5: entry.five.normalized,
      normalized_ns_d7: entry.seven.normalized,
      log10_dose: noInfinity(Math.log10(entry.dose)),
    };
    for (const day of V2_DAYS) {
      const summary = day === 5 ? entry.five : entry.seven;
      for (const field of readouts) {
        // build_inputs stores arcsinh values with infinities as missing; representation takes sinh back.
        const raw = Math.sinh(noInfinity(Math.asinh(summary.raw[field])));
        const control = Math.sinh(noInfinity(Math.asinh(summary.control[field])));
        features[`contrast__${field}__d${day}`] = contrastV2(raw, control);
        features[`missing__${field}__d${day}`] = summary.missing[field];
      }
    }
    const rel = reliability.get(entry.key) || {};
    for (const column of constants.reliability_columns) features[column] = rel[column] === undefined ? NaN : rel[column];
    return {
      key: entry.key,
      caseId: `${entry.date}|${entry.identity}|${g12(entry.dose)}`,
      date: entry.date,
      identity: entry.identity,
      name: names.get(entry.identity) ?? entry.identity,
      dose: entry.dose,
      replicates: features.rel__rep_count_d7,
      plates: [...(platesOf.get(entry.key) || [])].sort(),
      features,
    };
  });

  // neuroforecast_v2.batch_covariates: per date, from day-7 inputs only.
  const byDate = new Map();
  for (const c of cases) push(byDate, c.date, c.features.normalized_ns_d7);
  const controlWells = new Map();
  for (const row of rows) if (row.DIV === 7 && row.dose === 0) push(controlWells, plateKey(row), row);
  const plateMedians = new Map();
  for (const wells of controlWells.values()) push(plateMedians, wells[0].date, Math.asinh(pdMedian(wells.map((w) => w.values["ns.n"]))));
  const batch = new Map();
  for (const [date, values] of byDate) {
    batch.set(date, {
      batch_sd_d7: pdStd(values),
      batch_changed_frac_d7: values.filter((v) => Math.abs(v) >= CHANGED_LIMIT).length / values.length,
      batch_ctrl_median_d7: plateMedians.has(date) ? pdMedian(plateMedians.get(date)) : NaN,
    });
  }
  for (const c of cases) Object.assign(c.features, batch.get(c.date));
  return cases;
}

/* ---------------------------------------------------------------- v2 forest */

/* Exact layout from tools/export_models_for_web.py --set v2: float64 thresholds and leaf values. Inputs are
 * rounded to float32 before each comparison, as scikit-learn's trees do, and leaves are summed in tree order. */
export class ExactForest {
  constructor(buffer, spec) {
    const nodes = spec.nodes;
    let at = 0;
    const take = (Type, count) => {
      const bytes = count * Type.BYTES_PER_ELEMENT;
      const view = new Type(buffer.slice(at, at + bytes));
      at += bytes;
      return view;
    };
    this.feature = take(Int16Array, nodes);
    this.left = take(Int32Array, nodes);
    this.right = take(Int32Array, nodes);
    this.split = take(Float64Array, nodes);
    this.offsets = take(Int32Array, spec.trees + 1);
    if (at !== buffer.byteLength) throw new Error(`Model file length mismatch: read ${at} of ${buffer.byteLength} bytes.`);
    this.columns = spec.columns;
    this.medians = Float64Array.from(spec.medians);
    if (this.medians.length !== this.columns.length) throw new Error("The model manifest needs one median per column.");
    this.x = new Float32Array(this.columns.length);
  }

  predict(features) {
    const x = this.x;
    for (let i = 0; i < this.columns.length; i += 1) {
      const value = features[this.columns[i]];
      if (value === undefined) throw new Error(`Feature "${this.columns[i]}" was not built.`);
      x[i] = Number.isNaN(value) ? this.medians[i] : value; // SimpleImputer(median), then float32
    }
    const { feature, left, right, split, offsets } = this;
    const trees = offsets.length - 1;
    let total = 0;
    for (let t = 0; t < trees; t += 1) {
      let node = offsets[t];
      while (feature[node] !== LEAF) node = x[feature[node]] <= split[node] ? left[node] : right[node];
      total += split[node];
    }
    return total / trees;
  }
}

/* ---------------------------------------------------------------- v2 triage */

export function triageV2(cases, models, constants) {
  const conditions = cases.map((entry) => {
    const f = entry.features;
    const forecast = models.forecast.predict(f);
    const difficulty = Math.max(models.difficulty.predict(f), constants.sigma_floor);
    const half = constants.interval_quantile * difficulty;
    const reference = Math.sinh(f.rel__plate_ctrl_median_ns_d7);
    return {
      ...entry,
      forecast,
      low: forecast - half,
      high: forecast + half,
      difficulty,
      trusted: difficulty <= constants.trust_threshold,
      day7: f.normalized_ns_d7,
      day5: f.normalized_ns_d5,
      persistence: f.normalized_ns_d7,
      foldChange: 2 ** forecast,
      referenceActivity: reference,
      flags: {
        low_reference_activity: reference < LOW_REFERENCE,
        few_controls: f.rel__plate_ctrl_count_d7 < FEW_CONTROLS,
      },
    };
  });

  const batches = [];
  const dates = [...new Set(conditions.map((c) => c.date))].sort((a, b) => a - b);
  for (const date of dates) {
    const group = conditions.filter((c) => c.date === date);
    const lowReference = group.filter((c) => c.flags.low_reference_activity).length;
    const usable = !(lowReference / group.length >= LOW_BATCH_FRACTION);
    // Display rule from the Python tool: a batch with a silent day-7 reference shows no condition as usable.
    // `trusted` keeps the validated difficulty verdict; only the verdict text is overridden.
    for (const c of group) {
      c.batchUsable = usable;
      c.usable = usable && c.trusted;
      c.verdict = !usable ? V2_VERDICTS.batchUnusable : c.trusted ? V2_VERDICTS.usable : V2_VERDICTS.declined;
    }
    const f = group[0].features;
    batches.push({
      date,
      conditions: group.length,
      chemicals: new Set(group.map((c) => c.identity)).size,
      plates: [...new Set(group.flatMap((c) => c.plates))].sort(),
      referenceActivity: pdMedian(group.map((c) => c.referenceActivity)),
      declined: group.filter((c) => !c.trusted).length,
      lowReference,
      fewControls: group.filter((c) => c.flags.few_controls).length,
      lowReferenceFraction: lowReference / group.length,
      spread: f.batch_sd_d7,
      changedFraction: f.batch_changed_frac_d7,
      usable,
      verdict: usable ? V2_BATCH_VERDICTS.usable : V2_BATCH_VERDICTS.low,
    });
  }
  return { conditions, batches };
}
