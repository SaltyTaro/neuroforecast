**NeuroForecast has been selected as the working topic; its predictive advantage still needs independent validation.** This report records the initial frozen comparison and one explicit development follow-up completed September 20, 2026 IST. The selected variant uses day-7 control-relative measurements. It has the lowest development mean error, with an uncertain 3.7% advantage over the strongest smaller baseline. All three reserved validation dates remain unscored.

The [selection decision](round3_topic_selection.md) explains the alternatives and prior work. The [project brief](neuroforecast_project.md) defines the intended researcher decision and next gate. Machine-readable values are in [the compact summary](neuroforecast_selection_summary.json) and `report_summary.json` in the run directory.

**Data and independent units.** The [EPA dataset, DOI 10.23719/1503191](https://doi.org/10.23719/1503191), accompanies [Shafer et al., 2019](https://doi.org/10.1093/toxsci/kfz052). Rat cortical neuronal cultures were exposed to chemicals during network development and measured on microelectrode arrays at days in vitro (DIV) 5, 7, 9 and 12. The exposure continued; these are not post-washout recovery experiments. The target column, `ns.n`, is the authors' number of network spikes, a collective network-activity readout.

| Audit step | Result |
| --- | ---: |
| Rows in the two pinned measurement CSVs | 17,224 |
| Repeated physical measurement rows removed | 96 |
| Distinct recordings after deduplication | 17,128 |
| Physical wells, using date + plate serial + well | 4,320 |
| Experiment-date groups in the source | 17 |
| Identified study substances, including named mixtures | 136 |
| Eligible substances after identity-conflict exclusions | 133 |
| Development substances after global reserve-identity purging | 101 |
| Development experiment-date groups | 12 |
| Development dose-condition cases | 819 |

The 96 duplicate rows are the same 24 Valinomycin wells across four dates-in-vitro, included in both source collections under aliases. All measurement fields agreed. Physical identity includes experiment date because plate serials recur. Chemical identities use the experimental-summary CAS fields or a named NOCAS identifier, with three explicit aliases. They are not 136 independently verified molecular structures.

The summary workbook and experimental metadata disagree about the identities of tris(2-chloroisopropyl) phosphate, acenaphthylene, and acenaphthene. All three were excluded rather than silently resolving the conflict. Three additional unmapped PFAS names were excluded; the authors' analysis code also excluded them. Zero-dose wells remain available as plate controls even when their assigned treatment identity is excluded, because their administered chemical dose is zero. Silent cultures (`nAE == 0`) remain in the assay. Missing burst metrics are represented as missing, with missingness features.

The source's latest three experiment dates, **20170920, 20171004 and 20171011**, were reserved from metadata before model fitting. They contain 33 identified substances, of which 32 are eligible. Every reserved chemical identity is also removed from earlier development dates. A dose-condition case is an experiment date, chemical identity, and positive concentration, aggregated across at least two usable replicate wells. The cohort requires eligible day-5, day-7 and day-12 summaries. There were no missing eligible targets among its 819 development input cases. Model replays, doses, wells and electrodes are not additional independent culture preparations. Date groups are the strongest available batch units used here; donor/litter independence is not established.

**Endpoint and available inputs.** Within each well and DIV, form `(ns.n + 1) / (same-plate zero-dose median ns.n + 1)`. Average these ratios across a case's replicate wells, then take log2. The day-12 value is the target. This is a pseudocount-regularized relative readout; a twofold change in it need not be a twofold change in the raw count, particularly near zero controls.

The initial model uses 111 features: day-5/day-7 summaries of 18 functional columns, same-day controls, missing fractions, the two normalized network-spike values, and log10 concentration. Raw and control means use `arcsinh`. The follow-up replaces raw/control pairs with bounded contrasts `(raw_mean - control_mean) / (abs(raw_mean) + abs(control_mean))`. Two zeros give zero; unavailable values stay missing. The selected variant uses 38 features from day 7 and dose. The smaller activity/coordination baseline uses 13 features from firing rate, correlation, active electrodes, actively bursting electrodes, and network-spike readout across days 5 and 7, plus normalized network-spike values and dose.

No chemical names/identities, plate/date IDs, day-9/day-12 measurements, future control values, viability, full-trajectory labels, or published potency estimates enter prediction features. The fitted imputer and scaler see only training folds. The day-7 variant was compared on the original cohort, which requires day 5; its current score therefore does not establish the result for all cultures with only a day-7 recording or prove measurement savings.

**Evaluation and selection chronology.** Every outer fold holds out one development date and purges its test chemical identities from all other training dates. Error is averaged across doses within each chemical, then chemicals within the held-out date, then equally across the 12 dates. It is mean absolute error (MAE) in log2 relative-readout units, not accuracy. The target-only models see normalized day-5/day-7 network-spike values and dose. Persistence predicts day 7 unchanged; the control reference predicts zero in log2 space; the trend rule extrapolates the day-5/day-7 slope to day 12.

Extra Trees uses 256 trees, depth 12, minimum leaf size 4, all features at each split, and seed 20260920. Ridge uses alpha 10. Hyperparameters were fixed, including through the follow-up. The [initial protocol](../experiments/neuroforecast_selection_protocol.json) was frozen before predictive evaluation. Its primary full-feature Extra Trees model achieved MAE **1.04736**, a **5.08%** reduction versus the strongest initial comparator, the control reference. Its descriptive paired-date interval was **-47.63% to +40.58%**; it improved seven of 12 dates. That result was inconclusive.

After inspecting the initial result, [one separately frozen follow-up](../experiments/neuroforecast_representation_protocol.json) tested four control-relative variants on the same data. The primary two-day relative model reached 1.00725, but the day-7 ablation reached 0.93711. The working prototype was chosen from these development comparisons. There was no nested model-selection estimate, so these scores must not be described as final unbiased performance.

| Method | Date-macro MAE | Role |
| --- | ---: | --- |
| Control-relative Extra Trees, day 7 | **0.93711** | Selected working prototype; follow-up ablation |
| Activity/coordination Extra Trees, days 5 + 7 | 0.97339 | Strongest smaller baseline |
| Control-relative Extra Trees, days 5 + 7 | 1.00725 | Follow-up primary hypothesis |
| Original full-feature Extra Trees | 1.04736 | Initial primary hypothesis |
| Control-relative Ridge | 1.09569 | Follow-up comparator |
| Control reference | 1.10345 | Constant baseline |
| Carry day 7 forward | 1.10480 | Persistence baseline |
| Target-only Extra Trees | 1.11721 | Initial learned comparator |
| Target-only Ridge | 1.14832 | Initial learned comparator |
| Original full-feature Ridge | 2.03921 | Initial learned comparator |
| Linear extrapolation | 2.91720 | Initial trend comparator |

![All development model comparisons](figures/model_comparison.png)

| Selected day-7 model compared with | Relative error reduction | Descriptive 95% paired-date interval | Dates improved |
| --- | ---: | ---: | ---: |
| Persistence | 15.18% | -22.31% to +39.90% | 9/12 |
| Control reference | 15.07% | -14.49% to +37.26% | 7/12 |
| Activity/coordination baseline | 3.73% | -3.52% to +9.76% | 9/12 |
| Two-day relative model | 6.96% | +1.90% to +11.23% | 9/12 |

Intervals use 10,000 paired resamples of the 12 date losses. They are descriptive development uncertainty, with few groups and overlapping training folds, and do not adjust for model selection. The original two-day follow-up primary is worse than its day-7 ablation. These data do not support a claim that observing two early days improves the proposed mechanism.

![Error on every development date](figures/batch_comparison.png)

**Early-warning diagnostics.** The initial protocol predefined a later large change as `abs(day12) >= 1`, and little early change as `abs(day7) < 0.5`. The point-forecast alert uses `abs(prediction) >= 1`. These operational cutoffs do not define clinical toxicity. Changes in either direction count. Classification results are diagnostics of a regression model selected on development data; predicted magnitudes are not calibrated probabilities.

Across all 819 cases, 175 have large later changes. The selected model detects 107, misses 68, and produces 47 false alerts; 597 cases are correctly below the alert threshold. In the predefined early-quiet subgroup, 45 of 427 cases have large later changes:

| Method | Detected / 45 | Missed / 45 | False alerts / 382 | Precision among alerts |
| --- | ---: | ---: | ---: | ---: |
| Selected day-7 model | 24 | 21 | 13 | 64.9% |
| Activity/coordination baseline | 23 | 22 | 8 | 74.2% |
| Original full-feature model | 24 | 21 | 40 | 37.5% |
| Persistence | 0 | 45 | 0 | No alerts |

The selected variant flags 37 of 427 cases, with recall 53.3%. Its extra detection over the small baseline comes with five additional false alerts. The smaller model also has slightly higher average precision in this subgroup (0.545 versus 0.535). The best forecast-error model is not an established best alert policy. Persistence's zero recall in this subgroup follows from its definition; the learned small baseline is the more informative comparison.

![Detected changes and false alerts](figures/early_warning.png)

**Failures and biological limits.** The largest per-date MAE is **2.56128** on 20160921. For plate MW1160-23 on that date, the day-12 zero-dose median network-spike count is zero, with controls spanning zero to 208. Forty-two development cases include this plate; all have large normalized day-12 changes. Three are in the early-quiet/later-change subgroup, and none of those three are detected. A small denominator can amplify relative targets, but this association does not prove an assay failure or a chemical mechanism. No cases were removed or scores recalculated to improve the result. `failure_audit.json` in the run directory and `development_control_summary.csv` in the run directory preserve the evidence.

Control/replicate reliability therefore belongs in the next gate. Any rule used at day 7 must use only information available then; later control quality can be analyzed as an outcome-quality sensitivity check under a separately locked protocol. Additional limits include correlated doses, only 12 development dates, identity uncertainty, standard assay-specific feature extraction, and a single public rat-culture source. A same-source three-date reserve provides a useful validation step, not broad external generalization. Clinical effects, irreversibility, recovery after washout, living-chip operation, and saved experimental days or wells have not been validated.

**Reproduction and executable examples.** All data, models, predictions and figures are under `E:\kaggle\AI4S\experiments\neuroforecast_selection_v1`; sources are under `E:\kaggle\AI4S\raw\research\round3`. Workspace junctions preserve the links in this report. Python 3.14.2 and [pinned CPU packages](../tools/requirements-neuroforecast.txt) were used. The initial run took 11.76 seconds and the follow-up 15.61 seconds as recorded by their scripts; these are local run timings, not cross-machine performance claims. Preparation, separate audits, documentation and downloads are additional work.

To run the saved representative example:

```powershell
python -X utf8 -B tools/neuroforecast_demo.py --input E:/kaggle/AI4S/experiments/neuroforecast_selection_v1/demo/representative_day7.csv --model E:/kaggle/AI4S/experiments/neuroforecast_selection_v1/representation_followup/models/20161109_extra_trees_relative_day7.joblib --output E:/kaggle/AI4S/experiments/neuroforecast_selection_v1/demo/cli_forecast.csv
```

The input has only day-7 observations and matching zero-dose controls. It includes preaudited identity/eligibility fields; this is not a general import tool for arbitrary instrument exports. The model was trained excluding this date and all of its test identities. It reproduces 84 saved forecasts. The second example is the worst development date, also with 84 cases. Their selection rule was upper-median and maximum per-date error, not a search for attractive predictions. The entry point rejects other DIVs. Maximum discrepancy between direct day-7 inference and saved cross-validation predictions is below `2e-15`.

![Representative and worst examples](figures/demo_examples.png)

For a fresh reproduction, use a new output directory. Completed prediction files are protected from accidental overwrite:

```powershell
python -X utf8 -B tools/neuroforecast_benchmark.py --stage prepare --raw E:/kaggle/AI4S/raw/research/round3/epa_nfa_raw.zip --out E:/kaggle/AI4S/experiments/neuroforecast_reproduction
python -X utf8 -B tools/neuroforecast_benchmark.py --stage evaluate --out E:/kaggle/AI4S/experiments/neuroforecast_reproduction
python -X utf8 -B tools/neuroforecast_benchmark.py --stage audit --raw E:/kaggle/AI4S/raw/research/round3/epa_nfa_raw.zip --out E:/kaggle/AI4S/experiments/neuroforecast_reproduction
python -X utf8 -B tools/neuroforecast_representation.py --source E:/kaggle/AI4S/experiments/neuroforecast_reproduction
python -X utf8 -B tools/neuroforecast_demo.py --export-examples --raw E:/kaggle/AI4S/raw/research/round3/epa_nfa_raw.zip --source E:/kaggle/AI4S/experiments/neuroforecast_reproduction
python -X utf8 -B tools/neuroforecast_failure_audit.py --raw E:/kaggle/AI4S/raw/research/round3/epa_nfa_raw.zip --source E:/kaggle/AI4S/experiments/neuroforecast_reproduction
python -X utf8 -B tools/report_neuroforecast.py --source E:/kaggle/AI4S/experiments/neuroforecast_reproduction
```

Each script accepts explicit paths for another machine. Obtain the source archive from its [record](https://doi.org/10.23719/1503191) or [pinned public URL](https://pasteur.epa.gov/uploads/10.23719/1503191/NTP_TC_Analysis.zip) and verify the recorded SHA-256. The existing local download helper uses E: by default. The benchmark needs neither the DeePhys MAT file nor the TUNI downloads.

The completed scientific checks removed every future observation, then separately mutated every future functional value and future control. Both operations left the input features exactly unchanged. Train/test dates and chemical identities are disjoint; reserve dates and identities are absent. Forty-eight saved initial models and 48 follow-up models reproduced their predictions, with maximum difference `1.78e-15`. Changing firing-rate units and matching controls together left the relative representation unchanged within numerical tolerance. Both day-7 examples reproduced all predictions; the standalone CLI also matched its saved output. The two auxiliary dataset audits reproduce with `python -X utf8 -B tools/audit_round3_sources.py`.

The source and artifact [storage summary](round3_storage_summary.json) records per-file hashes. Frozen data, benchmark, and representation code and both protocols retain their recorded hashes. Report, example and failure-audit scripts are separate from these numerical cores. The next gate should lock the measurement-quality checks, warning decision, operating point and comparison before unsealing the reserved dates.
