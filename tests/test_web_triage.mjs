/* The browser port must reproduce the audited Python pipeline.
 *
 * Runs app/triage.js over the two shipped reserve examples and compares every forecast,
 * difficulty, interval bound, trust verdict and quality flag against evaluation/reserve_intervals.csv
 * — the table the Python audit verified. Run: node tests/test_web_triage.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { parseCSV, prepareRows, buildFeatures, triage, Forest, g12, quantile, median } from "../app/triage.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const TOLERANCE = 1e-6;

let failures = 0;
const check = (label, ok, detail = "") => {
  if (!ok) { failures += 1; console.log(`FAIL  ${label} ${detail}`); }
  else console.log(`ok    ${label}${detail ? " " + detail : ""}`);
};

/* --- helpers under test ------------------------------------------------- */
check("g12 matches Python's .12g", ["0.0001", "0.0003", "20", "3", "0.03"].every(
  (expected, i) => g12([0.0001, 0.0003, 20.0, 3.0, 0.03][i]) === expected));
check("quantile uses linear interpolation", Math.abs(quantile([1, 2, 3, 4], 0.75) - 3.25) < 1e-12,
  `q75([1,2,3,4])=${quantile([1, 2, 3, 4], 0.75)}`);
check("median skips missing values", median([1, NaN, 3]) === 2);

/* --- load the exported models ------------------------------------------ */
const manifest = JSON.parse(readFileSync(join(ROOT, "app/models/manifest.json"), "utf8"));
const constants = manifest.constants;
const models = {};
for (const [label, spec] of Object.entries(manifest.models)) {
  const bytes = readFileSync(join(ROOT, "app/models", spec.file));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  models[label] = new Forest(buffer, spec);
}
check("three forests loaded", Object.keys(models).length === 3,
  Object.entries(manifest.models).map(([k, v]) => `${k}:${v.trees}x${v.nodes}`).join(" "));

/* --- the audited Python outputs ---------------------------------------- */
const audited = new Map();
{
  const text = readFileSync(join(ROOT, "evaluation/reserve_intervals.csv"), "utf8");
  const { header, rows } = parseCSV(text);
  const index = Object.fromEntries(header.map((name, i) => [name, i]));
  for (const row of rows) {
    audited.set(row.case_id, {
      prediction: Number(row.prediction),
      sigma: Number(row.sigma),
      low: Number(row["scaled_lower_0.8"]),
      high: Number(row["scaled_upper_0.8"]),
      abstain: String(row.abstain).toLowerCase() === "true",
      flags: {
        low_reference_activity: String(row.flag_low_reference_activity).toLowerCase() === "true",
        few_controls: String(row.flag_few_controls).toLowerCase() === "true",
        replicates_disagree: String(row.flag_replicates_disagree).toLowerCase() === "true",
        controls_disagree: String(row.flag_controls_disagree).toLowerCase() === "true",
      },
      day7: Number(row.day7),
    });
    void index;
  }
}
check("audited reserve table loaded", audited.size === 224, `${audited.size} conditions`);

/* --- run the port over both shipped examples --------------------------- */
const EXAMPLES = {
  20170920: "reserve_20170920_usable_day5_day7.csv",
  20171011: "reserve_20171011_usable_day5_day7.csv",
  20171004: "reserve_20171004_low_quality_day5_day7.csv",
};

let compared = 0;
const worst = { forecast: 0, sigma: 0, low: 0, high: 0 };
for (const [date, file] of Object.entries(EXAMPLES)) {
  const text = readFileSync(join(ROOT, "examples", file), "utf8");
  const { rows } = prepareRows(parseCSV(text), constants.readouts);
  const cases = buildFeatures(rows, constants);
  const { conditions, batches } = triage(cases, models, constants);

  check(`${date}: condition count matches Python`,
    conditions.length === [...audited.keys()].filter((k) => k.startsWith(`${date}|`)).length,
    `${conditions.length} conditions`);

  for (const condition of conditions) {
    const expected = audited.get(condition.key);
    if (!expected) { check(`${date}: case ${condition.key} present in audited table`, false); continue; }
    compared += 1;
    worst.forecast = Math.max(worst.forecast, Math.abs(condition.forecast - expected.prediction));
    worst.sigma = Math.max(worst.sigma, Math.abs(condition.difficulty - expected.sigma));
    worst.low = Math.max(worst.low, Math.abs(condition.low - expected.low));
    worst.high = Math.max(worst.high, Math.abs(condition.high - expected.high));
    if (condition.trusted === expected.abstain) check(`${date}: trust verdict for ${condition.key}`, false);
    for (const [name, value] of Object.entries(condition.flags)) {
      if (value !== expected.flags[name]) check(`${date}: flag ${name} for ${condition.key}`, false, `got ${value}`);
    }
    if (Math.abs(condition.day7 - expected.day7) > TOLERANCE) {
      check(`${date}: day-7 value for ${condition.key}`, false, `${condition.day7} vs ${expected.day7}`);
    }
  }

  const batch = batches[0];
  if (String(date) === "20171004") {
    check("20171004: batch is called unusable", !batch.usable && batch.lowReferenceFraction === 1);
    check("20171004: every condition carries a quality flag", batch.flagged === batch.conditions);
  } else {
    check(`${date}: batch is called usable`, batch.usable && batch.lowReferenceFraction === 0);
  }
}

check("every condition compared against the audited table", compared === 224, `${compared} compared`);
check("forecasts match Python", worst.forecast < TOLERANCE, `max |diff| ${worst.forecast.toExponential(2)}`);
check("difficulty matches Python", worst.sigma < TOLERANCE, `max |diff| ${worst.sigma.toExponential(2)}`);
check("interval bounds match Python", Math.max(worst.low, worst.high) < TOLERANCE,
  `max |diff| ${Math.max(worst.low, worst.high).toExponential(2)}`);

/* --- guards ------------------------------------------------------------- */
{
  const text = readFileSync(join(ROOT, "examples", EXAMPLES[20171011]), "utf8");
  const parsed = parseCSV(text);
  const future = { ...parsed, rows: parsed.rows.map((r, i) => (i === 0 ? { ...r, DIV: "12" } : r)) };
  let rejected = false;
  try { prepareRows(future, constants.readouts); } catch (error) { rejected = /day 12/.test(error.message); }
  check("day-12 rows are rejected", rejected);

  const dayFiveOnly = { ...parsed, rows: parsed.rows.filter((r) => r.DIV === "5") };
  let noSeven = false;
  try { prepareRows(dayFiveOnly, constants.readouts); } catch (error) { noSeven = /day-7/.test(error.message); }
  check("day-7 rows are required", noSeven);

  const withoutDay5 = { ...parsed, rows: parsed.rows.filter((r) => r.DIV !== "5") };
  const { rows } = prepareRows(withoutDay5, constants.readouts);
  const cases = buildFeatures(rows, constants);
  const { conditions } = triage(cases, models, constants);
  const full = triage(buildFeatures(prepareRows(parsed, constants.readouts).rows, constants), models, constants);
  const byKey = new Map(full.conditions.map((c) => [c.key, c]));
  const stable = conditions.every((c) => Math.abs(c.forecast - byKey.get(c.key).forecast) < 1e-12
    && Math.abs(c.difficulty - byKey.get(c.key).difficulty) < 1e-12);
  check("dropping day-5 rows changes no forecast or verdict", stable);
  check("comparator is withheld without day-5 rows", conditions.every((c) => c.comparator === null));
}

console.log();
if (failures) { console.log(`${failures} check(s) failed`); process.exit(1); }
console.log(`All checks passed. ${compared} conditions reproduce the audited Python outputs.`);
