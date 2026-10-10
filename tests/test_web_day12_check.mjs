/* The browser day-12 check and admission record must reproduce the Python tool and the sealed test's evidence.
 *
 * For the three held-out example batches the page ships (best, median and the largest loss to carry-forward):
 *   - day-12 truth (target, reference median, usable flag) equals evaluation/external/arm_b_cases.csv (to 1e-12);
 *   - the measured day-12 profile equals the v3 study's twin targets, evaluation/v3/prepared/twin.csv (to 1e-12);
 *   - batch summaries and the admission record equal tools/neuroforecast_triage_v2.py --day12 --record
 *     (tests/fixtures/v2_day12_record_*.json): counts, decisions and reasons exactly, numbers to 1e-9;
 *   - every input hash in the record is the SHA-256 of the file it names.
 * Run: node tests/test_web_day12_check.mjs
 */

import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  parseCSV, prepareRowsV2, prepareRowsDay12, buildFeaturesV2, triageV2, day12TruthV2, ExactForest,
} from "../app/triage.js";
import { day12Check, admissionRecord, labCalibration, withScale } from "../app/check.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const bytesOf = (relative) => readFileSync(join(ROOT, relative));
const read = (relative) => readFileSync(join(ROOT, relative), "utf8");

let failures = 0;
const check = (label, ok, detail = "") => {
  if (!ok) { failures += 1; console.log(`FAIL  ${label} ${detail}`); }
  else console.log(`ok    ${label}${detail ? " " + detail : ""}`);
};

const manifest = JSON.parse(read("app/models/v2/manifest.json"));
const constants = manifest.constants;
const models = {};
for (const [label, spec] of Object.entries(manifest.models)) {
  const b = bytesOf(join("app/models/v2", spec.file));
  models[label] = new ExactForest(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), spec);
}

/* Sealed evidence keyed like the page's cases: date|identity|dose. */
const sealed = new Map();
for (const row of parseCSV(read("evaluation/external/arm_b_cases.csv")).rows) {
  sealed.set(`${Number(row.date)}|${row.identity}|${Number(row.dose)}`, row);
}
const twin = new Map();
{
  const cases = parseCSV(read("evaluation/v3/prepared/cases.csv")).rows;
  const targets = parseCSV(read("evaluation/v3/prepared/twin.csv")).rows;
  if (cases.length !== targets.length) throw new Error("v3 cases and twin targets differ in length");
  cases.forEach((c, i) => twin.set(`${Number(c.date)}|${c.identity}|${Number(c.dose)}`, targets[i]));
}

const num = (s) => (s === "" ? NaN : Number(s));
const close = (a, b, tol) => (a === null && b === null) || (Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tol);
let worstNumber = 0;
function compareValue(path, a, b, tol, problems) {
  if (typeof a === "number" && typeof b === "number") {
    const d = Math.abs(a - b);
    if (Number.isInteger(a) && Number.isInteger(b) && tol === 0 ? a !== b : !(d <= tol)) problems.push(`${path}: ${a} vs ${b}`);
    else worstNumber = Math.max(worstNumber, d);
  } else if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) problems.push(`${path}: length ${a.length} vs ${b.length}`);
    else a.forEach((x, i) => compareValue(`${path}[${i}]`, x, b[i], tol, problems));
  } else if (a && b && typeof a === "object" && typeof b === "object") {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of keys) compareValue(`${path}.${k}`, a[k], b[k], tol, problems);
  } else if (a !== b) problems.push(`${path}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`);
}

const EXAMPLES = [["best", 20160120], ["median", 20141231], ["lost", 20140924]];
let conditionsChecked = 0, truthWorst = 0, profileWorst = 0;
for (const [role, date] of EXAMPLES) {
  const early = `app/examples/v2/external_${date}_day5_day7.csv`;
  const late = `app/examples/v2/external_${date}_day12.csv`;
  const { rows } = prepareRowsV2(parseCSV(read(early)), constants.readouts);
  const result = triageV2(buildFeaturesV2(rows, constants), models, constants);
  const truth = day12TruthV2(prepareRowsDay12(parseCSV(read(late)), constants.readouts).rows, constants);

  // 1. day-12 truth against the sealed test's evidence
  const expected = [...sealed.entries()].filter(([, r]) => Number(r.date) === date);
  let mismatched = 0;
  for (const [key, r] of expected) {
    const t = truth.get(key);
    const ok = t && close(t.measured, num(r.target), 1e-12) && close(t.reference, num(r.day12_ctrl_median_ns), 0)
      && t.usable === (r.usable === "True");
    if (t) truthWorst = Math.max(truthWorst, Math.abs(t.measured - num(r.target)));
    if (!ok) mismatched += 1;
  }
  check(`${date} (${role}): day-12 truth equals the sealed evidence`, mismatched === 0 && truth.size === expected.length,
    `${expected.length} conditions, ${mismatched} mismatched`);

  // 2. measured profile against the v3 twin targets
  let profileBad = 0, profileCompared = 0;
  for (const [key, t] of truth) {
    const target = twin.get(key);
    if (!target) { profileBad += 1; continue; }
    for (const [column, value] of Object.entries(t.profile)) {
      const want = num(target[column]);
      profileCompared += 1;
      if (Number.isNaN(want) && Number.isNaN(value)) continue;
      const d = Math.abs(value - want);
      profileWorst = Math.max(profileWorst, d);
      if (!(d <= 1e-12)) profileBad += 1;
    }
  }
  check(`${date}: measured day-12 profile equals the v3 twin targets`, profileBad === 0, `${profileCompared} values`);

  // 3. batch summaries and the admission record against the Python tool
  const python = JSON.parse(read(`tests/fixtures/v2_day12_record_${date}.json`));
  const checked = day12Check(result.conditions, truth);
  const problems = [];
  compareValue("batches", checked.batches, python.day12_check.batches, 1e-12, problems);
  check(`${date}: batch check equals Python`, problems.length === 0, problems.slice(0, 3).join("; "));

  const record = await admissionRecord({
    result, constants, lockedModels: manifest.generated_from,
    input: { file: early.split("/").pop(), bytes: bytesOf(early) },
    check: { result: checked, input: { file: late.split("/").pop(), bytes: bytesOf(late) } },
  });
  const strip = (r) => ({ ...r, created_utc: null, tool: { ...r.tool, implementation: null },
    input: { ...r.input, sha256: null }, day12_check: r.day12_check && { ...r.day12_check, input: { ...r.day12_check.input, sha256: null } } });
  const recordProblems = [];
  compareValue("record", strip(record), strip(python), 1e-9, recordProblems);
  check(`${date}: admission record equals Python`, recordProblems.length === 0,
    recordProblems.length ? recordProblems.slice(0, 3).join("; ") : `${record.conditions.length} conditions`);
  const sha = (relative) => createHash("sha256").update(bytesOf(relative)).digest("hex");
  check(`${date}: record hashes name the files read`, record.input.sha256 === sha(early) && record.day12_check.input.sha256 === sha(late));
  check(`${date}: locked model hashes match the development lock`,
    JSON.stringify(record.locked_models) === JSON.stringify(python.locked_models));
  conditionsChecked += record.conditions.length;
}

/* The lab's own calibration: two earlier checked records (the median batch and the largest loss) refit the interval
   scale for the 2016 batch, exactly as neuroforecast_triage_v2.py --history does. */
{
  const early = "app/examples/v2/external_20160120_day5_day7.csv";
  const late = "app/examples/v2/external_20160120_day12.csv";
  const history = ["app/examples/v2/record_20140924_checked.json", "app/examples/v2/record_20141231_checked.json"];
  const asRecord = (relative) => ({ file: relative.split("/").pop(), bytes: bytesOf(relative) });
  const { rows } = prepareRowsV2(parseCSV(read(early)), constants.readouts);
  const result = triageV2(buildFeaturesV2(rows, constants), models, constants);
  const own = new Set(result.batches.map((b) => b.date));
  const calibration = await labCalibration({ records: history.map(asRecord), lockedModels: manifest.generated_from, constants, excludeBatches: own });
  const scaled = withScale(result, calibration.scale);
  const truth = day12TruthV2(prepareRowsDay12(parseCSV(read(late)), constants.readouts).rows, constants);
  const checked = day12Check(scaled.conditions, truth);
  const record = await admissionRecord({
    result: scaled, constants, lockedModels: manifest.generated_from, calibration,
    input: asRecord(early), check: { result: checked, input: asRecord(late) },
  });
  const python = JSON.parse(read("tests/fixtures/v2_day12_record_20160120_history.json"));
  const strip = (r) => ({ ...r, created_utc: null, tool: { ...r.tool, implementation: null } });
  const problems = [];
  compareValue("record", strip(record), strip(python), 1e-9, problems);
  check("lab calibration: recalibrated record equals Python", problems.length === 0,
    problems.length ? problems.slice(0, 3).join("; ") : `scale ${calibration.scale.toFixed(3)} from ${calibration.history_conditions} conditions`);
  check("lab calibration: forecasts and verdicts unchanged", scaled.conditions.every((c, i) =>
    c.forecast === result.conditions[i].forecast && c.usable === result.conditions[i].usable && c.difficulty === result.conditions[i].difficulty));
  const refusal = async (records, pattern, label) => {
    let message = "";
    try { await labCalibration({ records, lockedModels: manifest.generated_from, constants, excludeBatches: own }); } catch (error) { message = error.message; }
    check(`lab calibration refuses ${label}`, pattern.test(message), `"${message.slice(0, 90)}"`);
  };
  await refusal([asRecord(history[0])], /at least 2 batches/, "a single batch");
  await refusal([asRecord(history[0]), asRecord("tests/fixtures/v2_day12_record_20160120.json")], /cannot calibrate itself/, "the batch's own record");
  await refusal([asRecord(history[0]), asRecord(late)], /not a JSON admission record/, "a CSV");
  const unchecked = JSON.parse(read(history[0]));
  unchecked.day12_check = null;
  await refusal([{ file: "unchecked.json", bytes: Buffer.from(JSON.stringify(unchecked)) }, asRecord(history[1])], /no day-12 check/, "a record without a check");
  const otherModel = JSON.parse(read(history[0]));
  otherModel.locked_models = { ...otherModel.locked_models, "final_small_refit.joblib": "0".repeat(64) };
  await refusal([{ file: "other.json", bytes: Buffer.from(JSON.stringify(otherModel)) }, asRecord(history[1])], /different models/, "another model's record");
}

/* Guards: the check reads only day-12 rows, and never the forecast path. */
{
  const parsed = parseCSV(read("app/examples/v2/external_20160120_day5_day7.csv"));
  let message = "";
  try { prepareRowsDay12(parsed, constants.readouts); } catch (error) { message = error.message; }
  check("a day-5/7 file is refused as a day-12 file", /no day-12 recordings/.test(message), `"${message}"`);
  message = "";
  try { prepareRowsV2(parseCSV(read("app/examples/v2/external_20160120_day12.csv")), constants.readouts); } catch (error) { message = error.message; }
  check("the forecast path still refuses day-12 rows", /day 12/.test(message), `"${message.slice(0, 80)}…"`);
}

console.log();
if (failures) { console.log(`${failures} check(s) failed`); process.exit(1); }
console.log(`day-12 check matches the sealed evidence and the Python tool on ${conditionsChecked} conditions `
  + `(truth max |diff| ${truthWorst.toExponential(2)}, profile ${profileWorst.toExponential(2)}, record ${worstNumber.toExponential(2)})`);
