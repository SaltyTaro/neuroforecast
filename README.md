# NeuroForecast — day-7 trust triage for neuronal network assays

**Submission category: Tool & Platform.** AI4S Open Innovation: AI for Life Science (5th Pazhou Algorithm Competition).

A researcher running a chemical-exposure study on microelectrode arrays measures network activity days before the cultures finish developing. This tool reads the **day-7** wells and their plate controls and answers three questions that have to be answered on day 7, not afterwards:

1. **Can I trust today's measurement?** A batch-level quality verdict from the day-7 controls alone.
2. **Can the day-12 outcome be forecast for this condition at all?** A per-condition trust verdict, with an interval, beside the simple comparators.
3. **Where should a limited review budget go?** Two ranked lists, one restricted to conditions that have barely moved by day 7 — where the early readout carries no information.

## The headline result, stated first

We sealed three experiment dates (224 conditions, 32 chemicals absent from training) before any modelling, wrote the success criteria into a hashed protocol, and scored them **once**.

**The learned forecast lost.** Carrying the day-7 readout forward was more accurate than our model: date-macro MAE **0.6088 versus 0.7716**. We do not claim a forecasting advantage over simple methods, and the tool prints the persistence column beside every forecast.

What did generalize to those unseen batches and chemicals:

| Component | Held-out evidence |
| --- | --- |
| **Trust verdict** (day-7 difficulty model) | Declines 14% of conditions that carry **51% of all forecast error**. Retained-case MAE **0.446** versus **0.784** for all conditions. It improves persistence too (0.433 versus 0.595), so the verdict is useful whichever predictor you use. |
| **Measurement-quality flag** | On 20171004 every plate had a day-7 control median of **zero** network spikes; the cultures had not begun firing when the input was taken. A threshold fixed on development fired on all 84 of that batch's conditions and on no others — before any day-12 data existed. That is the batch where the forecast collapsed. |
| **Early-warning ranking** | Among conditions still quiet at day 7, a 20% review budget found **6 of 10** later large changes; ranking by the day-7 change found **0**; chance is 2.1. |
| **Prediction intervals** | Observed **71.4%** coverage at a nominal 80% level — a real shortfall under batch shift, reported as one. |

**Technical report: [docs/technical_report.md](docs/technical_report.md)** — problem, data, method, protocol, both evaluations, the failure analysis, the tool, limitations and reproduction. Full reserve numbers per date: [docs/neuroforecast_reserve_results.md](docs/neuroforecast_reserve_results.md).

## Use it in a browser

**<https://saltytaro.github.io/neuroforecast/app/>** — no install, no account, no upload. The models run
in the page: three Extra Trees forests are shipped as 2.7 MB of typed arrays and evaluated in JavaScript,
so a batch is triaged entirely on your machine.

Open with any of the three sealed-reserve batches, or drop in your own day-7 CSV. The page shows the
batch quality verdict, a 48-well plate map of day-7 activity with the untreated controls outlined, a
chart of the forecast against the "carry day 7 forward" diagonal, and the ranked review lists. Day-9 and
day-12 rows are rejected on load.

The browser port is not a reimplementation that can drift: `tests/test_web_triage.mjs` runs it over all
224 sealed-reserve conditions and compares every forecast, difficulty score, interval bound, trust
verdict and quality flag against the audited Python outputs — agreement to 8e-08, with the build failing
above 1e-06.

## Run it from the command line

Python 3.14 on CPU; no GPU, no paid service, no account. Pinned packages in [tools/requirements-neuroforecast.txt](tools/requirements-neuroforecast.txt).

```bash
pip install -r tools/requirements-neuroforecast.txt
python -X utf8 -B tools/neuroforecast_triage.py --input examples/reserve_20171011_usable_day5_day7.csv
```

Both example inputs are real day-7 wells from the sealed reserve — chemicals and batches the shipped models never saw. The first passes quality; the second is the batch whose cultures had not started firing:

```bash
python -X utf8 -B tools/neuroforecast_triage.py --input examples/reserve_20171004_low_quality_day5_day7.csv
```

```
Batch 20171004: 84 conditions, day-7 reference activity 0.0 network spikes
  measurement quality: day-7 reference activity is too low for this batch;
  forecasts here are unreliable and a day-12 measurement is recommended
```

Day-5 rows are optional; day-9 and day-12 rows are **rejected** by the entry point, so a forecast cannot be made with information the researcher would not have. Write results with `--output table.csv --summary batch.json`.

## Reproduce it in a notebook

[`notebooks/neuroforecast_reproduction.ipynb`](notebooks/neuroforecast_reproduction.ipynb) walks through the whole result in nine cells: it clones this repository, checks every published number against the evidence, runs the tool on both sealed-reserve batches, reproduces the per-date comparison that the learned model lost, and shows what the trust verdict recovered. Parts 1–5 need no download. It runs on Kaggle with **Internet** enabled in the notebook settings, and locally in any Jupyter environment.

## Reproduce the evaluation

Download the public EPA source archive (~160 MB) and verify its checksum, then run the four stages:

```bash
python -X utf8 -B tools/fetch_epa_data.py --out raw/epa_nfa_raw.zip
python -X utf8 -B tools/neuroforecast_gate.py --stage prepare --raw raw/epa_nfa_raw.zip --out runs/gate
python -X utf8 -B tools/neuroforecast_gate.py --stage develop --out runs/gate
python -X utf8 -B tools/neuroforecast_gate.py --stage reserve --unseal --out runs/gate
python -X utf8 -B tools/neuroforecast_gate.py --stage audit --raw raw/epa_nfa_raw.zip --out runs/gate
python -X utf8 -B tools/report_neuroforecast_gate.py --out runs/gate
```

`develop` takes about 8.4 minutes single-threaded and is bit-reproducible: two independent runs produced identical hashes for every prepared file, prediction file, interval file, flag file and fold record. `reserve` takes 12 seconds, requires the unchanged development lock and the explicit `--unseal`, and refuses to run twice.

```bash
python -X utf8 -B -m unittest discover -s tests -v
python -X utf8 -B tools/verify_published_numbers.py
node tests/test_web_triage.mjs
```

26 tests, including end-to-end checks that the shipped tool reproduces the audited reserve forecasts, intervals, trust verdicts and quality flags to 1e-12, and that removing day-5 rows changes none of them. The second command recomputes all 33 headline numbers in this README and the reports directly from `evaluation/*.csv`, at the precision each is quoted to, and fails if any of them drifts from the evidence.

## How the evaluation was kept honest

- **Batches and chemicals are both held out.** Every fold leaves out one experiment date *and* purges that date's chemicals from training, so no chemical is ever seen at another dose on another plate.
- **The reserve was sealed before modelling.** Three later dates, 32 chemical identities purged from development everywhere, scored once, every date reported.
- **The criteria were hashed before scoring.** [docs/neuroforecast_gate_protocol.md](docs/neuroforecast_gate_protocol.md) records what counts as success, the operating points, and a dated amendment made before unsealing.
- **Selection optimism is measured.** A nested leave-one-date-out estimate of the whole selection procedure gives **1.133**, worse than persistence — reported beside the post-selection 0.937.
- **Leakage is tested, not asserted.** The audit removes every observation after day 7, then mutates every future value and every future control, and requires the model inputs to be bit-identical.
- **Failures are kept.** The worst batch, the collapsed control plate, and the failed criterion are in the reports; nothing was excluded to improve a number.

## Data and licences

Source: EPA network formation assay, [DOI 10.23719/1503191](https://doi.org/10.23719/1503191), accompanying [Shafer et al., 2019](https://doi.org/10.1093/toxsci/kfz052). Rat cortical cultures on microelectrode arrays, measured at days 5, 7, 9 and 12 under continuing chemical exposure. EPA-produced data are in the U.S. public domain under the [ScienceHub licence statement](https://pasteur.epa.gov/license/sciencehub-license.html) unless otherwise specified; citing the data and the original study does not imply EPA endorsement. Archive SHA-256 `fd92c1339bb764ee9b96c935bf31867c12640505e5c3f065ad5956a7eea08cfb`.

After removing 96 duplicate physical rows: 17,128 recordings, 4,320 physical wells, 17 experiment dates, 136 identified substances. Three conflicting identities and three unmapped names are excluded; silent wells and missing burst measurements are retained. Development: 819 dose-condition cases, 101 substances, 12 dates. Reserve: 224 conditions, 32 substances, 3 dates.

Our code is MIT licensed ([LICENSE](LICENSE)). The shipped models are trained only on the public EPA development data.

## What this does not establish

Rat cortical cultures in multi-well plates are not perfused organ chips, human cells, or clinical toxicity. Nothing here validates saved experimental days or wells, irreversibility, recovery after washout, or chip performance. The reserve is three later batches from one public study, not broad external validity. Forecast magnitudes are not calibrated probabilities. Early neural forecasting, MEA chemical screening, and toxicological tipping-point analysis are established prior work — see [docs/round3_topic_selection.md](docs/round3_topic_selection.md) for the prior-art assessment that bounds our claim. Extra Trees, control normalization and split-conformal prediction are standard methods used as implementation choices; the contribution is the triage workflow and the evaluation protocol around it.

## Repository map

| Path | What it is |
| --- | --- |
| `tools/neuroforecast_triage.py` | The tool: day-7 wells in, quality verdict, forecasts with comparators, trust verdicts and ranked lists out |
| `tools/neuroforecast_gate.py` | The validation gate: prepare, develop, reserve (once), audit |
| `tools/neuroforecast_data.py` | Frozen data core: audited loading, case construction, purged splits |
| `tools/neuroforecast_benchmark.py`, `tools/neuroforecast_representation.py` | Frozen selection benchmark and the control-relative representation |
| `experiments/*.json` | The three hashed protocols |
| `models/` | Final models trained on the 819 development cases, the development lock, and the reserve summary |
| `examples/` | Two real day-7 inputs from the sealed reserve |
| `docs/technical_report.md` | The technical report |
| `docs/` | Protocol, development results, reserve results, project brief, prior-art assessment |
| `evaluation/` | Audited per-case predictions, intervals and flags for both stages, so every number recomputes without the source archive |
| `tools/verify_published_numbers.py` | Recomputes every published number from those tables |
| `app/` | The browser tool: page, the ported pipeline, and the forests as typed arrays |
| `notebooks/` | A nine-cell walkthrough that clones, verifies and runs everything |
| `docs/history/` | Two earlier approaches from this campaign that failed their own gates, kept for disclosure |

## Disclosure

Earlier rounds of this campaign tested two different approaches — adaptive dose selection from cell-image profiles, and vessel-patch selection for an oxygen-transport model — and both failed their pre-registered gates. Their reports are in `docs/history/`. Claude (Anthropic) was used as a coding and drafting assistant throughout; all numbers in this repository come from the scripts here, run on public data, and every claim above is checked against a saved evaluation output.
