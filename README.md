# NeuroForecast: a validation and trust layer for AI on neural MEA assays

**Submission category: Tool & Platform.** AI4S Open Innovation: AI for Life Science (5th Pazhou Algorithm Competition). · [中文说明](README.zh.md) · **[Open the tool](https://saltytaro.github.io/neuroforecast/app/)** · [Technical report (PDF)](docs/technical_report.pdf)

**Before a lab trusts an AI forecast on a new culture batch, NeuroForecast measures how good the forecast really is on batches it has never seen, and says which of today's forecasts to distrust.**

Neuronal networks grown on microelectrode arrays (MEAs) are the functional assay of the OECD developmental-neurotoxicity in-vitro battery, and neural organ chips record the same kind of data. Every culture batch behaves differently, and each new chip run is a new batch. On public EPA data (41 batches, 2,243 conditions, 243 chemicals):

- **Weak validation flatters forecasters.** Holding out chemicals while sharing culture batches understates error by 12%, and random splits by 15%. Even the batch-held-out split is about 10% optimistic against a split by period.
- **On 24 never-seen batches the shipped forecast beat carrying day 7 forward** (error 0.735 against 0.980, 19 of 24 batches). The 15% of conditions it refused carried 38% of all error.
- **A lab's own day-12 checks repair under-covering intervals.** Replayed in date order, coverage of 80% intervals rose from 0.69 to 0.85.

![Error and interval coverage of the same model under four validation designs](docs/figures/split_study.png)

## 1. Validation: hold out culture batches, and measure what weaker splits hide

The harness ([docs/validation_harness.md](docs/validation_harness.md)) fixes folds for four split designs. It scores any forecaster against carry-forward and a calibrated day-7 curve on the same conditions, on the command line or **[in the browser](https://saltytaro.github.io/neuroforecast/app/score.html)** with nothing uploaded:

```bash
python -X utf8 -B tools/score_predictions.py --predictions my_forecasts.csv    # or --example
```

Its labels are public, so scores are self-reported: it is a shared yardstick, not a blind leaderboard.

A locked study compared the designs with five models and five fold repetitions. Its protocol, harness and code were [committed publicly](https://github.com/SaltyTaro/neuroforecast/commit/9a90926) before the first run. A one-model probe had already shown the effect; this is disclosed in the protocol.

| Validation design | Extra Trees error | Lower than the reference design by | 80% intervals validated this way, on new batches |
| --- | ---: | ---: | ---: |
| Random conditions | 0.598 | 15% | 0.75 |
| Chemicals held out | 0.615 | 12% | 0.76 |
| Batches held out | 0.688 | 2% | 0.80 |
| **Batches and chemicals held out (reference)** | **0.702** | — | 0.80 (by construction) |

- **Sharing culture batches, not familiar chemicals, is what flatters a forecaster.**
- **Flexible models are flattered most.** Extra Trees beats a calibrated day-7 curve by 29–30% under random or chemical splits, but by 21% under batch and chemical hold-out.
- **Mixing years within a fold flatters too** (post hoc). Trained only on the other period, as in the sealed external test, the same recipe errs 10% more on the same 24 batches (0.731 against 0.659), and its edge over the calibrated curve falls to 11%.

## 2. Trust at the bench: a day-7 tool that knows when not to forecast

From the day-5 and day-7 wells of a 12-day assay:
1. **Batch check.** Is the control reference usable?
2. **Provisional read.** A day-12 forecast with an 80% interval, always beside "carry day 7 forward".
3. **Refusal.** Which conditions day 12 must decide.
4. **Day-12 self-check and admission record.** The day-7 report is scored with the sealed test's own definition, and a record is written with decisions, reasons and model and input hashes.
5. **Recalibration.** The intervals are refitted from the lab's own checked records.

It runs in your browser with nothing uploaded, or from the command line. The two give bit-identical forecasts and verdicts, and day-12 checks that agree to within 1e-13.

| Sealed test (data sealed before modelling, criteria hashed, scored once) | Batches | Conditions | Forecast MAE | Carry day 7 forward |
| --- | ---: | ---: | ---: | ---: |
| External: earlier batches (2014–2016), 105 chemicals never seen (**v2, shipped**) | 24 | 927 | **0.735**: the forecast **won**, 19 of 24 batches | 0.980 |
| Reserve: three later batches (v1, retired) | 3 | 224 | 0.7716 | **0.6088**: the forecast **lost** |

What holds, and what does not:
- **Refusal works, but simply.** The 15% the tool refused carried 38% of all error; declining them cut error on the rest by 27%, and by 30% for carry-forward. Every refused condition had already collapsed at day 7, and declining the most extreme forecasts does as well.
- **On the conditions it keeps, the forecast only ties a calibrated day-7 curve** (0.541 against 0.554), and half of its gain over carry-forward (49%) sits in the conditions it refuses. For the call "no 2-fold change by day 12" it is no better than reading day 7 (negative predictive value 0.864 against 0.876). When its interval excludes "no change" it was right 33 times out of 33, but that happens for only 4% of kept conditions. At day 7 the trust layer is worth more than the forecaster.
- **Intervals under-covered on new batches** (0.70 at a nominal 0.80). Recalibrating from the lab's own day-12 checks was replayed in date order:
  - refitting on every earlier batch (the primary rule) gave 0.85, with intervals 62% wider;
  - refitting on the last six batches gave 0.80, 42% wider;
  - the worst fifth of batches stayed near 0.53–0.55.

  A constant widening chosen with hindsight does as well, so what the lab learns is one inflation factor.

## 3. A path for chips (ingestion only)

`tools/chip_ingest.py` turns spike times into 16 of the tool's 17 readouts and compares them with EPA's control wells. We ran it on a public four-compartment hippocampal chip ([Zenodo 4679064](https://doi.org/10.5281/zenodo.4679064), 10 recordings from 9 untreated cultures, days 21–37):
- 59% of readout values fall inside EPA's day-12 control-well range, with network readouts worst. The gap mixes tissue, culture age and our approximate readout definitions.
- The archive has no vehicle chips, so no real reference check was possible.
- The output cannot enter the tool yet: it has no `mi`, and no plate, treatment or concentration columns.

This shows the data path, not chip validation.

**What is new** is not the learners: Extra Trees, conformal intervals and abstention are standard. What is new:
- a public validation harness and scorer with batch-held-out splits, and the measured cost of weaker splits;
- a closed day-7-to-day-12 trust loop (refusal, self-check, admission record, recalibration) tested in the browser and in Python;
- the evidence: two sealed tests (one lost), a locked 19-model study (trees win; v2 stays shipped), and the publicly pre-registered split study.

**Failures are published:** early warning did not replicate; a day-7 chemical call reached 89.1% against a 90% bar; the learned difficulty score ties a zero-parameter rule.

**Full results.**
- [Technical report](docs/technical_report.md) ([PDF](docs/technical_report.pdf), 18 pages).
- [Validation harness and split study](docs/validation_harness.md).
- [Model study](docs/neuroforecast_v3_results.md).
- The two sealed tests: [test 1](docs/neuroforecast_reserve_results.md) and [test 2](docs/neuroforecast_external_results.md).

## Use it in a browser

**<https://saltytaro.github.io/neuroforecast/app/>**. There is no install, no account and no upload.

**Held-out batches**, none seen by v2, each with its public day-12 recordings, chosen by rule:

| Batch | Rule | v2 | Carry day 7 forward |
| --- | --- | ---: | ---: |
| 2016-01-20 | best of the 24 | 0.286 | 0.622 |
| 2014-12-31 | median | 0.678 | 1.092 |
| 2014-09-24 | v2's largest loss | 0.959 | 0.725 |

**On the page:**
- **Reveal measured day 12** runs the check.
- **Use example lab history** recalibrates the intervals from two earlier checked batches.
- **Download record** saves the admission record.
- Click any concentration–response panel to enlarge it.
- You can also open your own CSV of day-5 and day-7 wells ([input schema and template](docs/input_schema.md)).

**Direct links:**
- [#best,history,check](https://saltytaro.github.io/neuroforecast/app/#best,history,check);
- [#lost,check](https://saltytaro.github.io/neuroforecast/app/#lost,check);
- [#twin](https://saltytaro.github.io/neuroforecast/app/#twin), the day-12 functional profile panel (research preview).

To run the page locally, serve the repository (`python -m http.server 8000`) and open <http://localhost:8000/app/>.

## Run it from the command line

It needs Python 3.11 or newer (verified on 3.14) and a CPU: no GPU, paid service or account. Use a fresh virtual environment, because the pins replace numpy, pandas and scikit-learn. The pickled models need scikit-learn 1.9.0 exactly.

```bash
pip install -r tools/requirements-neuroforecast.txt
python -X utf8 -B tools/neuroforecast_triage_v2.py --input examples/v2/external_20160120_day5_day7.csv \
    --day12 examples/v2/external_20160120_day12.csv --record record.json
python -X utf8 -B tools/neuroforecast_triage_v2.py --input examples/v2/external_20160120_day5_day7.csv \
    --history app/examples/v2/record_20140924_checked.json app/examples/v2/record_20141231_checked.json
python -X utf8 -B tools/neuroforecast_triage_v2.py --input examples/v2/reserve_20171004_low_quality_day5_day7.csv
```

The first command adds the day-12 check:

```
Day-12 check (scored after day 12 was recorded; nothing here changes the day-7 report):
  Batch 20160120: 42 of 42 conditions scored (0 with an unusable day-12 reference)
    error: forecast 0.286 vs carry day 7 forward 0.622; forecast closer on 27 of 42
    80% intervals held 0.929 of outcomes
    provisional (39): forecast 0.266 vs carry-forward 0.423; declined (3): forecast 0.550 vs carry-forward 3.200
```

The second refits the intervals from two earlier checked batches: `80% scale 1.506 (shipped) -> 2.887`. The third is a batch whose cultures had not started firing by day 7, so every condition is marked "batch reference unusable".

The forecast **rejects** day-9 and day-12 rows. The check reads day-12 rows only from its own file, and never feeds back. Bad input (a missing column, a day-9 or day-12 row, a duplicate well, too few replicates) stops with a one-line message. `-X utf8` makes Windows read files as UTF-8, and `-B` keeps bytecode out of the repository.

## Verify and reproduce

```bash
python -X utf8 -B tools/verify_published_numbers.py        # recomputes all 436 published numbers from the evidence tables
python -X utf8 -B -m unittest discover -s tests -v         # sealed-test replays, harness, check, recalibration, chip readouts
node tests/test_web_triage_v2.mjs                          # browser port against Python on 1,228 conditions
node tests/test_web_day12_check.mjs                        # browser check, record and recalibration against Python
node tests/test_web_score.mjs                              # browser harness scorer against Python
node tests/test_web_twin.mjs                               # browser transformer against PyTorch on 203 conditions
node tests/test_web_triage.mjs                             # retired v1 browser port against the audited test-1 outputs
```

**Unit tests.** 77 pass when torch is installed; a CPU build is enough (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). Without it, the 3 transformer tests are skipped and 74 pass.

**Re-running.**
- **The split study:** see [docs/validation_harness.md](docs/validation_harness.md). It takes about 8 minutes on a CPU, and its shipped lock refuses changed code.
- **The post hoc analyses:** `tools/posthoc_temporal_split.py` (about 2 minutes), `posthoc_decision_value.py`, `posthoc_lab_recalibration.py` and `posthoc_calibrated_day7.py`.
- **Test 2 from source:** run `tools/fetch_epa_refinement.py` (pinned EPA files, about 3 MB, checksums verified), then `tools/reproduce_external.py` with `prepare`, `develop`, `freeze`, `external --unseal` and `audit`. `develop` is byte-reproducible on Windows; elsewhere, compare contents, because CSV line endings differ.
- **The model study:** run `tools/run_neuroforecast_v3.sh` (CUDA GPU, about 2.5 hours). Its predictions ship in `evaluation/v3/`, and its analysis reproduces byte for byte on CPU.
- **Test 1:** run `tools/fetch_epa_data.py`, then `tools/neuroforecast_gate.py`.
- **Notebook:** [`notebooks/neuroforecast_v2_walkthrough.ipynb`](notebooks/neuroforecast_v2_walkthrough.ipynb) runs the verifier, the tool, the harness scorer and the re-scoring of both sealed tests in about a minute, with outputs saved.

## How the evaluation was kept honest

- **Batches and chemicals are both held out**, in both sealed tests, the model study and the harness's reference split.
- **Sealed twice, scored once each.** The criteria were hashed before development and the thresholds locked before unsealing. Those timestamps are our own records; the split study's protocol was public before it ran.
- **Selection optimism is measured.** A nested estimate of the whole selection procedure is reported beside every selected model.
- **Leakage is tested, not asserted.** Future values and controls are mutated, and model inputs must stay bit-identical.
- **Failures stay in.** The lost test, the withdrawn claims, the under-covering intervals and the calibrated-curve tie are all reported.

## Data and licences

**Test 1 and v1 development** use the EPA network formation assay archive, [DOI 10.23719/1503191](https://doi.org/10.23719/1503191), with [Shafer et al., 2019](https://doi.org/10.1093/toxsci/kfz052). It is US public domain ([ScienceHub licence statement](https://pasteur.epa.gov/license/sciencehub-license.html)).

**Test 2, v2 development, the model study and the harness** use the EPA refinement release, [USEPA/CompTox-DNT-NFA-Refinement](https://github.com/USEPA/CompTox-DNT-NFA-Refinement) at `01adf3e`. It has no licence file; we treat it as a US-Government work and cite it to EPA.

**The chip example** is [Zenodo 4679064](https://doi.org/10.5281/zenodo.4679064) (CC BY 4.0; Vakilna, Tang, Wheeler and Brewer, 2021). Only derived per-compartment readouts are shipped.

Our code is MIT licensed ([LICENSE](LICENSE)).

## What this does not establish

- **Scope.** It does not establish performance on organ chips, human cells or clinical toxicity, and validates no saved recording days, wells or cost.
- **Independence.** The two tests come from one EPA program and one laboratory, and dates are not verified independent donors.
- **Decisions.** No lab decision is shown to change: for a 2-fold-change call the tool is no better than reading day 7, and day 12 is still recorded.
- **Post hoc analyses.** The period split, the calibrated-curve comparison, the decision-value tables, the recalibration replay and the chip ingestion were not pre-registered. The split study was, after a disclosed probe.
- **Forecast magnitudes** are not probabilities.

## Repository map

| Path | What it is |
| --- | --- |
| `tools/neuroforecast_harness.py`, `tools/score_predictions.py`, `app/score.html`, `evaluation/harness/` | The validation harness: splits, scorer (command line and browser), folds, reference forecasts, the split study's per-case results and lock |
| `tools/neuroforecast_split_study.py`, `experiments/neuroforecast_split_study_protocol.json` | The pre-registered split-optimism study |
| `tools/neuroforecast_triage_v2.py` | The v2 tool: batch verdict, forecast with interval and comparator, refusal; `--day12`, `--record`, `--history` |
| `tools/posthoc_temporal_split.py`, `tools/posthoc_decision_value.py`, `tools/posthoc_lab_recalibration.py`, `tools/posthoc_calibrated_day7.py`, `tools/posthoc_signal_analysis.py`, `tools/control_subsampling.py` | Post hoc analyses: the period split, where the forecast helps and the binary call, the recalibration replay, the calibrated-curve baseline, input importance, controls needed |
| `tools/chip_ingest.py`, `evaluation/chip_ingestion/` | Spike times to readouts and a domain check against EPA's controls; results on Zenodo 4679064 |
| `tools/neuroforecast_v2.py`, `tools/neuroforecast_v2_external.py`, `tools/reproduce_external.py` | Test 2: prepare, develop, freeze, score once, audit (locked bytes) |
| `tools/neuroforecast_triage.py`, `tools/neuroforecast_gate.py` | The v1 tool and the v1 gate (test 1), unchanged |
| `tools/neuroforecast_data.py`, `tools/neuroforecast_benchmark.py`, `tools/neuroforecast_representation.py` | Frozen cores, imported unchanged by everything |
| `tools/neuroforecast_v3_*.py`, `tools/neuroforecast_v31_intervals.py` | The model study and the per-batch interval study (locked bytes) |
| `experiments/*.json` | The hashed protocols |
| `models/v2/`, `models/` | The v2 models (test 2) and the v1 bundle (test 1) |
| `evaluation/` | Per-case evidence for every test and study, lock files and prepared inputs |
| `tools/verify_published_numbers.py` | Recomputes every published number from those tables |
| `app/` | The browser tool (`triage.js` forecast, `check.js` day-12 check, record and recalibration, `twin.js` profile panel) and the harness scorer (`score.html`, `score.js`) |
| `examples/`, `app/examples/` | Real inputs, day-12 files for the held-out examples, checked example records, and harness predictions |
| `docs/` | Technical report, harness, protocols and results; `docs/history/` keeps topic-selection records and two earlier approaches that failed their gates |

## Disclosure

Earlier rounds of this campaign tested two other approaches, adaptive dose selection from cell-image profiles and vessel-patch selection for an oxygen-transport model; both failed their pre-registered gates (`docs/history/`). Claude (Anthropic) was used as a coding, analysis and drafting assistant throughout. Every number in this repository comes from the scripts here, run on public data, and is checked against saved evidence.
