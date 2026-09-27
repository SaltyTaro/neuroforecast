# NeuroForecast: day-7 forecasting and trust triage for neuronal network assays

**Submission category: Tool & Platform.** AI4S Open Innovation: AI for Life Science (5th Pazhou Algorithm Competition).

Neuronal networks grown on microelectrode arrays (MEAs) are recorded at days 5, 7, 9 and 12 while a chemical acts on them. This is the functional assay of the OECD developmental-neurotoxicity in-vitro battery, and the readout that neural organ chips produce.

At the **day-7** recording, this tool answers three questions from day-5 and day-7 wells alone:

1. **Is this batch's reference usable?** A batch quality verdict from the day-7 untreated controls.
2. **What will day 12 show?** A forecast with an 80% interval, always printed beside the "carry day 7 forward" value.
3. **Can that forecast be trusted?** A per-condition verdict that declines the conditions it cannot predict.

## Two sealed tests

Each test was sealed before modelling, with the criteria hashed in advance, and scored once:

| Sealed test | Batches | Conditions | Forecast MAE | Carry day 7 forward |
| --- | ---: | ---: | ---: | ---: |
| 1. Three later batches, v1 model | 3 | 224 | 0.7716 | **0.6088**: the forecast **lost** |
| 2. Earlier batches (2014–2016), 105 chemicals never seen, v2 model | 24 | 927 | **0.735**: the forecast **won** | 0.980 |

In test 2, the v2 forecast had 25% lower error than carrying day 7 forward, and was better on 19 of 24 batches. The paired batch interval runs from −0.40 to −0.12.

| Component | Held-out evidence |
| --- | --- |
| **Trust verdict** | Declining the least predictable 15% of conditions cut error on the rest by **27%** for the forecast and **30%** for carry-forward. Declined conditions had 3.5 times the error of kept ones. In test 1 it declined 14% of conditions, which carried 51% of the error. |
| **Batch quality verdict** | In test 1, every plate of one batch had silent day-7 controls. A threshold fixed on development flagged all 84 of that batch's conditions, and no others, before any day-12 data existed. No batch in test 2 had silent controls, so the flag rests on this one event. |
| **Intervals** | Coverage was 70% at a nominal 80% in test 2 (71% in test 1). This is a stated shortfall under batch shift. |
| **Withdrawn** | Test 1 showed an early-warning ranking for conditions still quiet at day 7; test 2 did not replicate it. A day-7 chemical-level call reached 89.1% agreement with EPA's final call, short of its pre-registered 90%. |

**Technical report: [docs/technical_report.md](docs/technical_report.md).** Results in full:
- test 1: [docs/neuroforecast_reserve_results.md](docs/neuroforecast_reserve_results.md)
- test 2: [docs/neuroforecast_external_results.md](docs/neuroforecast_external_results.md)
- test 2 protocol: [experiments/neuroforecast_v2_protocol.json](experiments/neuroforecast_v2_protocol.json)

## Use it in a browser

**<https://saltytaro.github.io/neuroforecast/app/>**. There is no install, no account and no upload. The forests are shipped as typed arrays and run on your machine.

Open a held-out batch from 2016, or the collapsed batch, or drop in your own CSV of day-5 and day-7 wells. `tests/test_web_triage_v2.mjs` replays every sealed condition through the JavaScript port and fails if any output differs from Python by more than 1e-6.

## Run it from the command line

Python 3.14 on CPU, with no GPU, paid service or account. Pinned packages are in [tools/requirements-neuroforecast.txt](tools/requirements-neuroforecast.txt).

```bash
pip install -r tools/requirements-neuroforecast.txt
python -X utf8 -B tools/neuroforecast_triage_v2.py --input examples/v2/external_20160120_day5_day7.csv
python -X utf8 -B tools/neuroforecast_triage_v2.py --input examples/v2/reserve_20171004_low_quality_day5_day7.csv
```

The first example is a batch from test 2 that the model never saw. The second is the batch whose cultures had not started firing by day 7:

```
Batch 20171004: 84 conditions, day-7 reference activity 0.0 network spikes
  measurement quality: day-7 reference activity is too low for this batch;
  forecasts here are unreliable and a day-12 measurement is recommended
```

Day-9 and day-12 rows are **rejected**, so a forecast cannot use information the researcher would not have at day 7. Write results with `--output table.csv --summary batch.json`.

The v1 tool that was scored in test 1 is kept unchanged as `tools/neuroforecast_triage.py`, with its bundle in `models/`.

## Verify and reproduce

```bash
python -X utf8 -B tools/verify_published_numbers.py        # recomputes all 148 published numbers from the evidence tables
python -X utf8 -B -m unittest discover -s tests -v         # exact replays of both sealed tests, input rules, cohort separation
node tests/test_web_triage_v2.mjs                          # browser port against Python on every sealed condition
```

To re-run test 2 from source, start with `tools/fetch_epa_refinement.py`, which downloads the pinned EPA files (about 3 MB) and verifies their checksums. Then run `tools/reproduce_external.py` with the stages `prepare`, `develop`, `freeze`, `external --unseal` and `audit`.

`develop` takes about three minutes. It is **byte-reproducible**: a fresh run from this repository reproduced every prepared file, every out-of-fold prediction and all five models exactly. That was verified on Windows with Python 3.14; on other systems the CSV line endings differ, so compare contents there.

To re-run test 1, use `tools/fetch_epa_data.py` and `tools/neuroforecast_gate.py`. The stages are `prepare`, `develop`, `reserve --unseal` and `audit`.

[`notebooks/neuroforecast_reproduction.ipynb`](notebooks/neuroforecast_reproduction.ipynb) walks through the v1 result in nine cells.

## How the evaluation was kept honest

- **Batches and chemicals are both held out.** Every fold leaves out one batch *and* purges that batch's chemicals from training. In test 2, the 105 primary chemicals appear nowhere in development.
- **Sealed twice, scored once each.** The criteria were hashed before development, the thresholds were locked before unsealing, and the scoring code was frozen with the hashes of the unchanged v1 tool.
- **Selection optimism is measured.** A nested estimate of the whole selection procedure is reported beside the selected model: 1.133 for v1, 0.770 for v2.
- **Leakage is tested, not asserted.** Future values and controls are mutated and model inputs must stay bit-identical. No external condition, date or chemical reached any fit.
- **Failures stay in.** The lost test, the withdrawn claims and the under-covering intervals are in the reports. Nothing was excluded to improve a number.

## Data and licences

**Test 1 and v1 development** use the EPA network formation assay archive, [DOI 10.23719/1503191](https://doi.org/10.23719/1503191), with [Shafer et al., 2019](https://doi.org/10.1093/toxsci/kfz052). It is in the US public domain under the [ScienceHub licence statement](https://pasteur.epa.gov/license/sciencehub-license.html).

**Test 2 and v2 development** use the EPA refinement release, [USEPA/CompTox-DNT-NFA-Refinement](https://github.com/USEPA/CompTox-DNT-NFA-Refinement) at commit `01adf3e`. It re-processes the same recordings and adds 28 dates from 2014–2016. The repository has no licence file; we treat it as a US-Government work and cite it to EPA.

Both releases are rat cortical cultures on 48-well MEA plates. Our code is MIT licensed ([LICENSE](LICENSE)).

## What this does not establish

This work does not establish performance on perfused organ chips, human cells, or clinical toxicity. It validates no saved recording days, wells or cost. The two tests come from one EPA program and one laboratory, and dates are not verified independent donors.

The v2 gain over v1 comes from refitting on more batches, not from a new model class: the v1 architecture refit on the same data did as well. Forecast magnitudes are not probabilities. Extra Trees, control normalization, conformal intervals and abstention are standard methods; the contribution is the day-7 triage workflow and the evaluation around it.

## Repository map

| Path | What it is |
| --- | --- |
| `tools/neuroforecast_triage_v2.py` | The v2 tool: day-5/7 wells in; batch verdict, forecast with interval and comparator, trust verdict out |
| `tools/neuroforecast_v2.py`, `tools/neuroforecast_v2_external.py` | Test 2: prepare and develop; then freeze, score once, audit (locked bytes) |
| `tools/external_nfa_intake.py`, `tools/external_nfa_equivalence.py` | The adapter for the EPA refinement release, and its check on known data |
| `tools/reproduce_external.py`, `tools/fetch_epa_refinement.py` | Re-run test 2 from a fresh clone |
| `tools/neuroforecast_triage.py`, `tools/neuroforecast_gate.py` | The v1 tool and the v1 gate (test 1), unchanged |
| `tools/neuroforecast_data.py`, `tools/neuroforecast_benchmark.py`, `tools/neuroforecast_representation.py` | Frozen cores, imported unchanged by everything |
| `experiments/*.json` | The four hashed protocols |
| `models/v2/`, `models/` | The v2 models (test 2), and the v1 bundle (test 1) |
| `evaluation/` | Per-case evidence for both tests, the v2 lock files, and the prepared inputs, so every number recomputes without a download |
| `tools/verify_published_numbers.py` | Recomputes every published number from those tables |
| `app/` | The browser tool |
| `examples/v2/`, `examples/` | Real day-5/7 inputs for the v2 and v1 tools |
| `docs/` | Technical report, protocols and results for both tests, prior-art assessment |
| `docs/history/` | Two earlier approaches from this campaign that failed their own gates, kept for disclosure |

## Disclosure

Earlier rounds of this campaign tested two other approaches, adaptive dose selection from cell-image profiles and vessel-patch selection for an oxygen-transport model. Both failed their pre-registered gates, and their reports are in `docs/history/`.

Claude (Anthropic) was used as a coding, analysis and drafting assistant throughout. Every number in this repository comes from the scripts here, run on public data, and is checked against saved evidence.
