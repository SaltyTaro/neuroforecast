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
