// The browser twin panel (app/twin.js) against PyTorch on the app's example batches.
// Fixtures: tests/fixtures/v3_twin_*.json, written by tools/export_pct_for_web.py. Run: node tests/test_web_twin.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { parseCSV, prepareRowsV2, buildFeaturesV2 } from "../app/triage.js";
import { PCTEnsemble, twinForecast, erf } from "../app/twin.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (p) => readFileSync(join(root, p));
const constants = JSON.parse(read("app/models/v2/manifest.json")).constants;
const manifest = JSON.parse(read("app/models/v3/pct.json"));
const bin = read("app/models/v3/pct.bin");
const model = new PCTEnsemble(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), manifest);

// erf against known values
for (const [x, want] of [[0.5, 0.5204998778130465], [1, 0.8427007929497149], [2.5, 0.9995930479825550], [-3, -0.9999779095030014]]) {
  if (Math.abs(erf(x) - want) > 1e-14) throw new Error(`erf(${x}) = ${erf(x)}, expected ${want}`);
}

const examples = {
  external_20160120: "app/examples/v2/external_20160120_day5_day7.csv",
  reserve_20171004: "app/examples/v2/reserve_20171004_low_quality_day5_day7.csv",
  reserve_20171011: "app/examples/v2/reserve_20171011_usable_day5_day7.csv",
};
let worst = 0, checked = 0;
for (const [label, path] of Object.entries(examples)) {
  const fixture = JSON.parse(read(`tests/fixtures/v3_twin_${label}.json`));
  const { rows } = prepareRowsV2(parseCSV(read(path).toString("utf-8")), constants.readouts);
  const cases = buildFeaturesV2(rows, constants);
  const started = Date.now();
  const result = twinForecast(rows, cases, model);
  const ms = Date.now() - started;
  if (result.size !== fixture.conditions.length) throw new Error(`${label}: ${result.size} forecasts, fixture has ${fixture.conditions.length}`);
  for (const want of fixture.conditions) {
    const c = cases.find((k) => k.date === want.date && k.identity === want.identity && Math.abs(k.dose - want.dose) < 1e-12);
    if (!c) throw new Error(`${label}: no JS condition for ${want.date}|${want.identity}|${want.dose}`);
    const got = result.get(c.key);
    const diffs = [got.q10 - want.q10, got.q50 - want.q50, got.q90 - want.q90, ...got.twin.map((v, k) => v - want.twin[k])];
    const d = Math.max(...diffs.map(Math.abs));
    worst = Math.max(worst, d);
    checked += 1;
    if (!(d <= fixture.tolerance)) throw new Error(`${label} ${c.key}: differs from PyTorch by ${d}`);
  }
  console.log(`${label}: ${result.size} conditions, ${ms} ms`);
}
console.log(`twin panel matches PyTorch on ${checked} conditions; max |diff| ${worst.toExponential(2)} (tolerance 1e-4)`);
