/* The browser harness scorer (app/score.js) must reproduce the Python scorer (tools/neuroforecast_harness.score)
 * on the shipped v2 example: errors, per-batch errors, batch wins and interval coverage exactly (to 1e-12), and the
 * bootstrap intervals to within 0.01 (they use a different random stream). It must also refuse incomplete files.
 * The Python reference is tests/fixtures/harness_score_v2_example.json, written by
 * `python -X utf8 -B tools/score_predictions.py --example --json tests/fixtures/harness_score_v2_example.json`.
 * Run: node tests/test_web_score.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { loadBenchmark, scoreForecast } from "../app/score.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (relative) => readFileSync(join(ROOT, relative), "utf8");
let failures = 0;
const check = (label, ok, detail = "") => {
  if (!ok) { failures += 1; console.log(`FAIL  ${label} ${detail}`); } else console.log(`ok    ${label}${detail ? " " + detail : ""}`);
};

const cases = loadBenchmark(read("evaluation/v3/prepared/cases.csv"), read("evaluation/harness/reference_predictions.csv"));
const example = read("examples/harness/v2_reference_predictions.csv");
const web = scoreForecast(cases, example);
const python = JSON.parse(read("tests/fixtures/harness_score_v2_example.json"));

check("conditions, batches, chemicals", web.conditions === python.conditions && web.batches === python.batches && web.chemicals === python.chemicals,
  `${web.conditions} / ${web.batches} / ${web.chemicals}`);
let worst = 0;
const close = (a, b) => { const d = Math.abs(a - b); worst = Math.max(worst, d); return d <= 1e-12; };
check("date-macro, pooled and worst-batch error", close(web.date_macro_mae, python.date_macro_mae) && close(web.pooled_mae, python.pooled_mae)
  && close(web.worst_batch_mae, python.worst_batch_mae), web.date_macro_mae.toFixed(6));
check("per-batch errors", Object.entries(python.per_batch).every(([d, v]) => close(web.per_batch[d], v)), `${Object.keys(python.per_batch).length} batches`);
let intervalGap = 0;
for (const [name, p] of Object.entries(python.against)) {
  const w = web.against[name];
  check(`against ${name}: error, difference and batch wins`, close(w.date_macro_mae, p.date_macro_mae) && close(w.difference, p.difference)
    && w.batches_better === p.batches_better && w.batches === p.batches, `${w.batches_better} of ${w.batches}`);
  intervalGap = Math.max(intervalGap, Math.abs(w.low - p.low), Math.abs(w.high - p.high));
}
check("bootstrap intervals within 0.01 of Python", intervalGap <= 0.01, `max gap ${intervalGap.toFixed(4)}`);
const keys = ["pooled", "batch_mean", "worst_fifth", "share_of_batches_within_0.1", "mean_width"];
check("interval coverage", keys.every((k) => close(web.intervals[k], python.intervals[k])), `pooled ${web.intervals.pooled.toFixed(4)}`);

const lines = example.trim().split("\n");
let message = "";
try { scoreForecast(cases, lines.slice(0, -5).join("\n")); } catch (error) { message = error.message; }
check("refuses missing conditions", /no finite prediction/.test(message), `"${message}"`);
message = "";
try { scoreForecast(cases, [...lines, lines[1]].join("\n")); } catch (error) { message = error.message; }
check("refuses repeated case IDs", /repeat a case_id/.test(message), `"${message.slice(0, 60)}"`);
message = "";
try { scoreForecast(cases, "case,value\n1,2"); } catch (error) { message = error.message; }
check("refuses a file without case_id and prediction", /case_id and prediction/.test(message), `"${message}"`);

console.log();
if (failures) { console.log(`${failures} check(s) failed`); process.exit(1); }
console.log(`browser harness scorer matches Python (max |diff| ${worst.toExponential(2)}; bootstrap gap ${intervalGap.toFixed(4)})`);
