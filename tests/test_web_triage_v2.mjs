/* The browser port of the v2 tool must reproduce tools/neuroforecast_triage_v2.py.
 *
 * Runs app/triage.js (v2 pipeline, app/models/v2) over the two fixtures, 983 external and 245 reserve
 * conditions, and compares every forecast, interval bound and difficulty (to 1e-6), and every trust
 * decision, quality flag, verdict and batch verdict (exactly), against evaluation/v2_reference.
 * The shipped page examples are checked the same way. Run: node tests/test_web_triage_v2.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  parseCSV, prepareRowsV2, buildFeaturesV2, triageV2, ExactForest, V2_BATCH_VERDICTS,
} from "../app/triage.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const TOLERANCE = 1e-6;
const read = (relative) => readFileSync(join(ROOT, relative), "utf8");

let failures = 0;
const check = (label, ok, detail = "") => {
  if (!ok) { failures += 1; console.log(`FAIL  ${label} ${detail}`); }
  else console.log(`ok    ${label}${detail ? " " + detail : ""}`);
};

/* --- the exported forests ---------------------------------------------- */
const manifest = JSON.parse(read("app/models/v2/manifest.json"));
const constants = manifest.constants;
const models = {};
for (const [label, spec] of Object.entries(manifest.models)) {
  const bytes = readFileSync(join(ROOT, "app/models/v2", spec.file));
  models[label] = new ExactForest(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), spec);
}
check("two v2 forests loaded", Object.keys(models).length === 2 && models.forecast && models.difficulty,
  Object.entries(manifest.models).map(([k, v]) => `${k}:${v.trees}x${v.nodes}`).join(" "));
check("locked product and trust score", constants.product === "small_refit" && constants.trust === "sigma_v2");
check("17 readouts, cv.time and cv.network absent, mi present", constants.readouts.length === 17
  && !constants.readouts.includes("cv.time") && !constants.readouts.includes("cv.network") && constants.readouts.includes("mi"));

/* --- the Python reference outputs --------------------------------------- */
const bool = (value) => {
  if (value === "True") return true;
  if (value === "False") return false;
  throw new Error(`not a Python boolean: ${value}`);
};
function reference(relative) {
  const table = new Map();
  for (const row of parseCSV(read(relative)).rows) {
    if (table.has(row.case_id)) throw new Error(`duplicate case_id ${row.case_id} in ${relative}`);
    table.set(row.case_id, {
      date: Number(row.date),
      name: row.name,
      forecast: Number(row.forecast_log2),
      low: Number(row.forecast_low),
      high: Number(row.forecast_high),
      difficulty: Number(row.difficulty),
      day7: Number(row.day7_observed_log2),
      persistence: Number(row.persistence_log2),
      replicates: Number(row.replicates),
      trusted: bool(row.forecast_trusted),
      verdict: row.verdict,
      low_reference_activity: bool(row.quality_low_reference_activity),
      few_controls: bool(row.quality_few_controls),
    });
  }
  return table;
}

/* Batch verdicts follow from Python's own flags: the batch is low when at least half its conditions are. */
function expectedBatchVerdicts(table) {
  const byDate = new Map();
  for (const r of table.values()) {
    const entry = byDate.get(r.date) || { n: 0, low: 0 };
    entry.n += 1; entry.low += r.low_reference_activity ? 1 : 0;
    byDate.set(r.date, entry);
  }
  return new Map([...byDate].map(([date, e]) => [date, e.low / e.n >= 0.5 ? V2_BATCH_VERDICTS.low : V2_BATCH_VERDICTS.usable]));
}

const worst = { forecast: 0, low: 0, high: 0, difficulty: 0, day7: 0 };
let compared = 0;
let fixtureConditions = 0;
const allBatches = new Map();

function compare(label, csvText, table, onlyDate = null) {
  const { rows } = prepareRowsV2(parseCSV(csvText), constants.readouts);
  const { conditions, batches } = triageV2(buildFeaturesV2(rows, constants), models, constants);
  const expected = onlyDate === null ? table : new Map([...table].filter(([, r]) => r.date === onlyDate));

  const got = new Map(conditions.map((c) => [c.caseId, c]));
  const missingInJs = [...expected.keys()].filter((id) => !got.has(id));
  const missingInPython = [...got.keys()].filter((id) => !expected.has(id));
  check(`${label}: every case_id present on both sides`, !missingInJs.length && !missingInPython.length && got.size === conditions.length,
    `${conditions.length} JS / ${expected.size} Python`
    + (missingInJs.length ? `; absent in JS: ${missingInJs.slice(0, 3).join(", ")}` : "")
    + (missingInPython.length ? `; absent in Python: ${missingInPython.slice(0, 3).join(", ")}` : ""));

  const mismatches = { trusted: [], low_reference_activity: [], few_controls: [], verdict: [], name: [], replicates: [] };
  const local = { forecast: 0, low: 0, high: 0, difficulty: 0, day7: 0 };
  for (const c of conditions) {
    const e = expected.get(c.caseId);
    if (!e) continue;
    compared += 1;
    local.forecast = Math.max(local.forecast, Math.abs(c.forecast - e.forecast));
    local.low = Math.max(local.low, Math.abs(c.low - e.low));
    local.high = Math.max(local.high, Math.abs(c.high - e.high));
    local.difficulty = Math.max(local.difficulty, Math.abs(c.difficulty - e.difficulty));
    local.day7 = Math.max(local.day7, Math.abs(c.day7 - e.day7), Math.abs(c.persistence - e.persistence));
    if (c.trusted !== e.trusted) mismatches.trusted.push(c.caseId);
    if (c.flags.low_reference_activity !== e.low_reference_activity) mismatches.low_reference_activity.push(c.caseId);
    if (c.flags.few_controls !== e.few_controls) mismatches.few_controls.push(c.caseId);
    if (c.verdict !== e.verdict) mismatches.verdict.push(`${c.caseId} "${c.verdict}" vs "${e.verdict}"`);
    if (c.name !== e.name) mismatches.name.push(`${c.caseId} "${c.name}" vs "${e.name}"`);
    if (c.replicates !== e.replicates) mismatches.replicates.push(c.caseId);
    for (const value of [c.forecast, c.low, c.high, c.difficulty]) {
      if (!Number.isFinite(value)) mismatches.trusted.push(`${c.caseId} non-finite output`);
    }
  }
  for (const [name, list] of Object.entries(mismatches)) {
    check(`${label}: ${name} agrees exactly`, !list.length, list.length ? `${list.length} differ, e.g. ${list[0]}` : "");
  }
  check(`${label}: forecast, interval, difficulty, day 7 within ${TOLERANCE}`, Object.values(local).every((v) => v <= TOLERANCE),
    `max |diff| forecast ${local.forecast.toExponential(2)} low ${local.low.toExponential(2)} high ${local.high.toExponential(2)} `
    + `difficulty ${local.difficulty.toExponential(2)} day7 ${local.day7.toExponential(2)}`);
  for (const k of Object.keys(worst)) worst[k] = Math.max(worst[k], local[k]);

  const verdicts = expectedBatchVerdicts(expected);
  const wrong = batches.filter((b) => b.verdict !== verdicts.get(b.date));
  check(`${label}: batch verdicts agree exactly`, batches.length === verdicts.size && !wrong.length,
    `${batches.length} batches` + (wrong.length ? `; wrong: ${wrong.map((b) => b.date).join(", ")}` : ""));
  for (const b of batches) allBatches.set(b.date, b);
  return { conditions, batches };
}

const external = reference("evaluation/v2_reference/external_triage_v2.csv");
const reserve = reference("evaluation/v2_reference/reserve_triage_v2.csv");
check("reference tables loaded", external.size === 983 && reserve.size === 245, `${external.size} external, ${reserve.size} reserve`);

compare("external", read("tests/fixtures/v2_external_day5_day7.csv"), external);
compare("reserve", read("tests/fixtures/v2_reserve_day5_day7.csv"), reserve);
fixtureConditions = compared;
check("all 1228 fixture conditions compared", fixtureConditions === 1228, `${fixtureConditions} compared`);

const low = [...allBatches.values()].filter((b) => !b.usable).map((b) => b.date);
check("only batch 20171004 is called unusable", low.length === 1 && low[0] === 20171004, `${allBatches.size} batches, low: ${low.join(", ")}`);

/* --- the examples the page ships ---------------------------------------- */
const EXAMPLES = [
  ["app/examples/v2/external_20160120_day5_day7.csv", external, 20160120],
  ["app/examples/v2/reserve_20171004_low_quality_day5_day7.csv", reserve, 20171004],
  ["app/examples/v2/reserve_20171011_usable_day5_day7.csv", reserve, 20171011],
];
for (const [file, table, date] of EXAMPLES) {
  const before = compared;
  const { batches } = compare(file.split("/").pop(), read(file), table, date);
  check(`${date}: example is one batch and matches the fixture rows`, batches.length === 1 && batches[0].date === date,
    `${compared - before} conditions`);
}

/* --- guards ------------------------------------------------------------- */
{
  const parsed = parseCSV(read(EXAMPLES[0][0]));
  const expectError = (label, mutate, pattern) => {
    let message = "";
    try { prepareRowsV2(mutate(parsed), constants.readouts); } catch (error) { message = error.message; }
    check(label, pattern.test(message), message ? `"${message.slice(0, 90)}…"` : "no error raised");
  };
  expectError("day-12 rows are rejected", (p) => ({ ...p, rows: p.rows.map((r, i) => (i === 0 ? { ...r, DIV: "12" } : r)) }), /day 12/);
  expectError("day-9 rows are rejected", (p) => ({ ...p, rows: p.rows.map((r, i) => (i === 0 ? { ...r, DIV: "9" } : r)) }), /day 9/);
  expectError("day-5 rows are required", (p) => ({ ...p, rows: p.rows.filter((r) => r.DIV !== "5") }), /day-5 and day-7/);
  expectError("day-7 rows are required", (p) => ({ ...p, rows: p.rows.filter((r) => r.DIV !== "7") }), /day-5 and day-7/);
  expectError("a missing v2 readout is named", (p) => ({ ...p, header: p.header.filter((h) => h !== "mi") }), /"mi"/);
  expectError("duplicate recordings are rejected", (p) => ({ ...p, rows: [...p.rows, p.rows[0]] }), /Duplicate/);
}

console.log();
if (failures) { console.log(`${failures} check(s) failed`); process.exit(1); }
console.log(`All checks passed. ${fixtureConditions} conditions (and the ${compared - fixtureConditions} in the page examples) `
  + `reproduce the Python v2 outputs (max |diff| forecast ${worst.forecast.toExponential(2)}, interval ${Math.max(worst.low, worst.high).toExponential(2)}, `
  + `difficulty ${worst.difficulty.toExponential(2)}).`);
