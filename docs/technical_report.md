# NeuroForecast: day-7 trust triage for neuronal network assays

**Submission category: Tool & Platform.**
AI4S Open Innovation: AI for Life Science — AI + Organ-on-a-Chip, 5th Pazhou Algorithm Competition.
Code and artifacts: <https://github.com/SaltyTaro/neuroforecast>

---

## Summary

Neuronal cultures on microelectrode arrays (MEAs) are measured repeatedly while a network forms. A researcher dosing such cultures sees a day-7 reading days before the endpoint, and has to decide what to do with it: trust it, act on it, or spend a measurement. We built a tool that answers three day-7 questions — whether today's measurement is usable, whether a given condition's day-12 outcome can be forecast at all, and where a limited review budget should go — and we tested it the hard way.

Three experiment dates (224 dose conditions, 32 chemicals) were sealed before any modelling. The success criteria, operating points and thresholds were written into a hashed protocol. The reserve was scored **once**.

**The learned forecast lost.** Simply carrying the day-7 reading forward was more accurate than our model: date-macro mean absolute error **0.6088 versus 0.7716**. We withdraw the forecasting claim, and the tool prints the simple rule beside every forecast.

Three things did transfer to those unseen batches and chemicals. A day-7 difficulty model declines 14% of conditions that carry 51% of all forecast error, halving the error of the conditions it keeps — and it improves the simple rule as well, so it is not a crutch for a weak model. A day-7 control-quality flag, thresholded on development data, identified every condition of the one batch whose cultures had not begun firing when the input was taken, before any day-12 data existed; that is precisely the batch where the forecast collapsed. And among conditions that had barely moved by day 7, ranking by forecast magnitude at a 20% review budget found 6 of 10 later large changes where ranking by the day-7 change found none.

The contribution is therefore not a better predictor. It is a triage workflow that knows when to abstain, and an evaluation protocol strict enough to discover that its own headline model was not worth deploying.

---

## 1. The problem and the decision

### 1.1 What the user is doing

In a developmental neurotoxicity screen, rat cortical cells are plated on multi-well MEAs, dosed with a chemical at several concentrations, and recorded at days *in vitro* (DIV) 5, 7, 9 and 12 while exposure continues. Network activity emerges over this window: at day 5 there is little coordinated firing, and by day 12 a healthy plate produces hundreds of network spikes per recording. The endpoint of interest is how far a dosed condition's day-12 network activity departs from its plate's untreated controls.

The day-7 recording exists, and is used mainly as a waypoint. The decisions available at that moment are real:

- **Is this plate's reading usable at all?** If the untreated controls are silent, the reference that every normalized readout depends on is degenerate, and downstream numbers from that plate are unreliable.
- **Which conditions deserve attention now?** Follow-up capacity — an extra recording, a repeat well, an imaging check, an analyst's eye — is finite. Conditions that have already collapsed by day 7 are visible without any model. The hard cases are those that look ordinary at day 7 and change later.
- **Can I rely on a prediction for this condition?** A point forecast with no statement of reliability is not usable for a decision with a cost attached.

### 1.2 What we deliberately do not claim

The obvious framing — "forecast day 12 from day 7 and save the later measurement" — is the one our evaluation rejected. Our forecast is less accurate than assuming nothing changes. We make no claim about saved days, saved wells, or saved cost, and none about toxicity, irreversibility, or recovery after washout. The data are rat cortical cultures in multi-well plates, not perfused organ chips or human tissue.

### 1.3 Relevance to organ-on-a-chip

The challenge is framed around AI for *in vitro* life systems, with organ-on-a-chip prominent and the supporting organization focused on neural chips. MEA electrophysiology is the functional readout neural chips produce, and the problem we address — deciding, mid-experiment, whether a functional measurement is trustworthy and whether a later state can be anticipated — is common to plate-based assays and perfused neural devices. We use public plate data because it is the largest measured, longitudinal, chemically perturbed functional dataset we could obtain under a public-data constraint, and because it supports a sealed multi-batch holdout. Transfer of these specific models to chips is untested and we do not assert it.

---

## 2. Prior work and the boundary of our claim

A bounded literature search (archived queries, data-record searches, source papers, cut-off 19 September 2026) established that the generic components of this problem are solved and published.

| Prior work | What it establishes | What it leaves for us |
| --- | --- | --- |
| [Early prediction of developing spontaneous activity in cultured neuronal networks (2021)](https://doi.org/10.1038/s41598-021-99538-9) | Later firing, bursting and synchrony can be predicted from early recordings with standard learners; 47 MEA dishes, 12 cultures. | Generic early neural forecasting is not novel. Our setting adds chemical exposure, unseen substances and unseen batches, and a decision rather than a number. |
| [Frank et al., 2017, 86-compound DNT screen](https://doi.org/10.1093/toxsci/kfx169) | Longitudinal MEA chemical screening; learned combinations of activity and coordination features. | Combining MEA features for toxicity is not novel. This work is why our strongest comparator is a small activity/coordination model rather than a trivial rule. |
| [Toxicological tipping points in neuronal network development (2018)](https://doi.org/10.1016/j.taap.2018.01.017) | Tipping-point and recovery/non-recovery trajectory analysis across neuronal development. | We must not claim a first tipping-point, irreversibility or recovery result. We predict one later measured readout under continuing exposure. |
| [Shafer et al., 2019](https://doi.org/10.1093/toxsci/kfz052) | The screen that produced our data, and its dose-response analyses. | The data and the observed chemical effects are prior work. Ours is the forecast benchmark, the triage decision, and the evaluation. |
| Conformal prediction; difficulty-scaled nonconformity | Distribution-free intervals with finite-sample validity under exchangeability; locally weighted variants. | Standard method. Our contribution is calibrating it under *batch and chemical* separation and reporting observed per-batch coverage rather than nominal. |

Extra Trees, control normalization, split-conformal prediction and selective prediction are established techniques used here as implementation choices. **We claim no new learning algorithm.** What we claim is the triage workflow — quality gate, abstention, budgeted ranking, each with held-out evidence and each reported including where it fails — and an evaluation protocol that was strict enough to reject our own primary hypothesis.

---

## 3. Data

### 3.1 Source and licence

EPA network formation assay, [DOI 10.23719/1503191](https://doi.org/10.23719/1503191), the dataset accompanying Shafer et al. (2019). Archive SHA-256 `fd92c1339bb764ee9b96c935bf31867c12640505e5c3f065ad5956a7eea08cfb` (~160 MB), verified on every load; the pipeline refuses to run against a different archive. EPA-produced data are in the U.S. public domain under the [ScienceHub licence statement](https://pasteur.epa.gov/license/sciencehub-license.html) unless otherwise specified. We cite the data and the original study; EPA endorsement is not implied. Our code is MIT licensed.

### 3.2 Audit, and what we chose not to hide

Loading is auditable and refuses to proceed on surprises.

| Audit step | Result |
| --- | ---: |
| Rows in the two pinned measurement files | 17,224 |
| Duplicate physical rows removed (all fields agreed) | 96 |
| Distinct recordings | 17,128 |
| Physical wells (date + plate serial + well) | 4,320 |
| Experiment dates | 17 |
| Identified substances | 136 |
| Eligible after identity-conflict exclusions | 133 |

The 96 duplicates are the same 24 Valinomycin wells appearing under aliases in both source collections; all measurement fields agreed, and they are removed as duplicates rather than treated as extra samples. Physical identity must include the experiment date because plate serial numbers recur. Three substances whose identities disagree between the summary workbook and the experimental metadata were **excluded rather than silently resolved**, and three unmapped PFAS names were excluded as the original analysis code also did. Zero-dose wells remain available as plate controls even when their nominal treatment identity is excluded, because they received no chemical. Silent wells (`nAE == 0`) are kept: a dead culture is a real outcome, not missing data. Missing burst metrics are represented as missing, with explicit missingness features.

### 3.3 Units of independence

A **case** is an experiment date, a chemical identity and a positive concentration, aggregated over at least two usable replicate wells (791 of 819 development cases span three plates, 28 span two). Cases within a date are correlated through shared plates, shared controls and shared culture preparation; cases sharing a chemical are correlated across dates.

We therefore treat **experiment dates as the grouping unit** and count dates and chemicals, never wells, electrodes, doses or model replays, as evaluation units. Dates are the strongest grouping available in the public files. They are **not** verified independent donors or litters: donor and litter identity is not published, so we do not claim biological independence between batches, only that a held-out batch was prepared, plated and recorded on a different day.

### 3.4 Cohort

| Split | Cases | Chemicals | Dates |
| --- | ---: | ---: | ---: |
| Development | 819 | 101 | 12 |
| Sealed reserve | 224 | 32 | 3 |

The reserve is the three latest experiment dates (20170920, 20171004, 20171011), chosen from metadata **before any model was fitted**. Every reserved chemical identity was additionally purged from the development set everywhere it appeared, including earlier repeats, so the reserve tests unseen chemicals and unseen batches simultaneously.

---

## 4. Method

### 4.1 Endpoint

For each well and DIV, form the control-relative ratio `(ns.n + 1) / (same-plate same-DIV zero-dose median ns.n + 1)`, where `ns.n` is the authors' network-spike count. Average over a case's replicate wells, then take log2. The day-12 value is the target; the day-7 value is both a feature and the persistence baseline.

The pseudocount keeps silent wells finite, at a cost we state plainly: near-zero controls inflate the ratio, so a twofold change in this readout is not necessarily a twofold change in raw counts. A predefined sensitivity subset (cases whose day-12 control median is at least 5 network spikes) quantifies this. That subset uses a future observation, so it is a reporting device only and never enters a feature, a model or a decision.

Two operational cutoffs, fixed in the first protocol: a **later large change** is `|target| ≥ 1`, and a condition is **quiet at day 7** if `|day-7 value| < 0.5`. These are analysis thresholds, not clinical definitions of toxicity.

### 4.2 Information available at the decision

Only day-5 and day-7 measurements, same-plate same-day zero-dose controls, and log10 concentration. Excluded by construction: chemical identity, plate and date identifiers, any day-9 or day-12 measurement or control, viability, published potency estimates, and author trajectory labels. The deployed model and its reliability layer use day 7 alone; day 5 is needed only by the small comparator.

Features are **control-relative contrasts**: for each of 18 readouts, `(raw − control) / (|raw| + |control|)`, bounded in [−1, 1], zero when both are zero, missing when unavailable, plus missingness fractions, the normalized network-spike value and dose. This representation is invariant to a change of measurement units applied to a readout and its controls together — a property the audit verifies numerically rather than assuming.

### 4.3 The three components

**(a) Measurement-quality flags.** Four fixed day-7 rules: control median network spikes below 1; fewer than four usable control wells; replicate spread or control spread above the development 90th percentile. These are display elements with no gating power over the metrics, and they are reported with the model's error on flagged versus unflagged conditions so a reader can judge them.

**(b) Forecast, with comparators.** Extra Trees (256 trees, depth 12, minimum leaf 4, all features per split, fixed seed) on the day-7 contrast features. Hyperparameters were fixed in the first protocol and never tuned afterwards. Comparators are always computed and always displayed: persistence, the control reference (predict no change), a day-5-to-day-7 trend extrapolation, target-only learners, Ridge variants, and the strongest small model — a 13-feature Extra Trees on firing rate, correlation, active electrodes, actively bursting electrodes and network spikes across days 5 and 7.

**(c) Trust verdict and intervals.** A difficulty model σ(x) — the same learner, on the day-7 features plus seven day-7 reliability indicators (replicate count, replicate spread, and five plate-control statistics) — is fitted to absolute out-of-fold residuals. Split-conformal intervals use difficulty-scaled nonconformity scores `|residual| / σ_oof`, with the finite-sample corrected quantile. **Every calibration residual comes from a model that never saw that case's date or chemical**, which is what makes the coverage claim meaningful under shift rather than under an i.i.d. assumption that does not hold here. Conditions whose predicted difficulty exceeds the development 80th percentile are marked *declined*.

**(d) Budgeted ranking.** Within each batch, rank by forecast magnitude and flag the top 20% (10% and 30% also reported). Two lists are produced: over all conditions, and restricted to conditions quiet at day 7. The second is the one the tool leads with, because the first is dominated by conditions the day-7 column already reveals.

---

### 4.4 Implementation

The code separates three concerns deliberately, because mixing them is how retrospective studies drift.

**Frozen numerical cores.** `neuroforecast_data.py` (audited loading, case construction, purged splits), `neuroforecast_benchmark.py` and `neuroforecast_representation.py` are hash-recorded and imported unchanged by everything downstream. The gate never edits them; its own manifests record their SHA-256 so that a changed core cannot be silently mixed into a completed run.

**The gate.** `neuroforecast_gate.py` has four stages — `prepare`, `develop`, `reserve`, `audit`. `prepare` rebuilds the feature matrix and checks it against the earlier frozen run (zero difference, verified). `develop` runs the nested folds, fits the difficulty model, calibrates the intervals, locks the thresholds and writes a **development lock**: a JSON file recording code hashes, protocol hashes, per-file hashes and every locked operating point. `reserve` refuses to run unless that lock verifies byte-for-byte, an explicit `--unseal` flag is present, and no reserve output already exists. `audit` re-checks leakage, fold purity, model replays and, after unsealing, every reserve output hash.

**The tool.** `neuroforecast_triage.py` loads only the shipped bundle — final models, lock, calibration constants — and never imports the evaluation machinery's state. It resolves its bundle repo-locally first so a fresh clone runs without any experiment directory.

Runtime on one CPU: `develop` 503 s single-threaded (12 outer and 131 inner folds, 12 candidates each, plus the difficulty models), `reserve` 11.8 s, the tool 0.2 s per batch, the full test suite 2.6 s. No GPU, no cloud, no paid service at any point.

## 5. Evaluation protocol

### 5.1 Splits

Twelve outer folds: leave out one development date, and purge that date's chemicals from training everywhere. Inside each outer training set, an inner leave-one-date-out loop over the remaining 11 dates, with the same identity purge, produces out-of-fold predictions for conformal calibration and for the difficulty model — so no calibration quantity is contaminated by the fold it will be applied to.

The primary metric is **date-macro MAE**: absolute error averaged over doses within a chemical, then over chemicals within a date, then equally over dates. This prevents a date with more conditions, or a chemical tested at more doses, from dominating.

### 5.2 Pre-registration, and one disclosed amendment

Three protocols were hashed before the work they govern: the selection protocol (before any predictive evaluation), a representation follow-up, and the validation gate. The gate fixed the cohort, the inputs, the candidate list, the product-model rule, the 20% budget, the 80% nominal level, the flag definitions, the reserve procedure and the pass/fail criteria **before the development run**, and authorized exactly one reserve scoring.

One amendment (v1.1) was made after the development run and **before** unsealing, and is disclosed as such in the protocol. The original criteria bundled "beats trivial baselines" together with "beats our own smaller model" into single pass/fail rules. The development run showed the second comparison sat inside selection noise. Bundled, a failure on the less interesting comparison would have concealed the outcome of the more important one. The criteria were therefore split into (a) versus trivial rules and (b) versus the small model. No data, model, comparator, threshold or operating point changed; every comparison is still computed and reported.

A second, purely technical amendment followed: the first re-run was not bit-reproducible, because floating-point summation order in multi-threaded tree prediction perturbed the difficulty model's targets by ~1e-15, moving the abstention threshold from 1.5445 to 1.5433. Forecasts were unaffected. The gate now predicts single-threaded and rounds residuals before fitting σ; two independent development runs then produced identical hashes for every artifact.

### 5.3 What the audit tests

Leakage is tested, not asserted. The audit removes every observation after day 7 and requires the model inputs to be bit-identical; then mutates every future measurement *and every future control* to an absurd value and requires the inputs to be bit-identical again. It verifies that the reliability indicators rebuild exactly, that train and test dates and chemicals are disjoint in all 12 outer and 131 inner folds, that reserve dates and identities appear nowhere in development, and that saved models replay their recorded predictions (maximum difference 4.4e-16). After the reserve, it re-verifies every output hash.

### 5.4 Criteria, fixed before scoring

| Criterion | Rule |
| --- | --- |
| C1a forecast vs trivial rules | Mean-of-three-dates MAE below persistence and the control reference, and below persistence on ≥2 of 3 dates |
| C1b forecast vs the small model | Below the activity/coordination model overall and on ≥2 of 3 dates |
| C2a warning vs the naive rule | At the 20% budget among quiet conditions, recall above persistence ranking and above chance |
| C2b warning vs the small model | Recall at least the small model's |
| C3 reliability | Observed coverage ≥ 0.70 at nominal 0.80, and retained-case MAE below all-case MAE |

---

## 6. Development results

### 6.1 Forecast comparison

| Candidate | Date-macro MAE |
| --- | ---: |
| Control-relative Extra Trees, day 7 (**product model**) | **0.93711** |
| Day-7 model plus reliability indicators | 0.94465 |
| Activity/coordination Extra Trees, days 5+7 | 0.97339 |
| Control-relative, days 5+7 | 1.00725 |
| Original full-feature Extra Trees | 1.04736 |
| Control-relative Ridge | 1.09569 |
| Control reference | 1.10345 |
| Persistence | 1.10480 |
| Target-only Extra Trees / Ridge | 1.11721 / 1.14832 |
| Full-feature Ridge; linear trend | 2.03921; 2.91720 |
| **Nested estimate of the selection procedure** | **1.13295** |

The product model is 15.2% better than persistence and 3.7% better than the small model on development, with paired-date intervals of −22.3% to +39.9% and −3.5% to +9.8% — both spanning no gain. Adding the reliability indicators to the forecaster does not help (0.8%, interval −7.0% to +6.2%).

Per date, against the two comparators that matter:

| Held-out date | Product | Persistence | Small model | Conditions |
| --- | ---: | ---: | ---: | ---: |
| 20160720 | 0.651 | 0.969 | 0.679 | 77 |
| 20160803 | 0.454 | 1.232 | 0.586 | 84 |
| 20160907 | 1.017 | 1.112 | 0.864 | 56 |
| 20160921 | **2.561** | 1.011 | 2.824 | 84 |
| 20161013 | 0.654 | 1.211 | 0.703 | 56 |
| 20161109 | 0.831 | 1.661 | 0.990 | 84 |
| 20170412 | 0.393 | 0.468 | 0.423 | 70 |
| 20170628 | 1.028 | 1.992 | 1.135 | 42 |
| 20170712 | 1.648 | 1.748 | 1.547 | 84 |
| 20170809 | 1.005 | **0.698** | 0.856 | 84 |
| 20170816 | 0.617 | **0.541** | 0.687 | 35 |
| 20170913 | 0.385 | 0.615 | 0.386 | 63 |

Persistence already beat the product model on two development dates. One of them, 20170809, is a low-activity batch — median day-7 reference activity 0.8 network spikes, half its conditions carrying the low-activity flag — which is exactly the failure mode that would later dominate the reserve. The other, 20170816, is the smallest development batch (35 conditions) with the *highest* day-7 activity and a narrow margin, so the two are not one story. The first signal was present before the reserve was opened; we did not act on it, because acting on it would have meant changing the method after seeing which dates were hard. The reserve then made the point decisively.

![Development comparison of every candidate, with the nested selection estimate in green](figures/gate_model_comparison.png)

**The nested estimate is the most important number in this section.** Running the entire selection procedure inside each outer fold — inner leave-one-date-out chooses a candidate, that candidate is scored on the held-out date — gives 1.133, *worse than persistence*. Inner selection chose the day-7 model on six dates, the reliability variant on three and the full-feature model on three; the full-feature choices produced two catastrophic outer losses (2.808 and 2.500). With 11 inner batches, choosing among the leading candidates is noise. The 0.937 headline is a post-selection number, and we report it as one.

### 6.2 Warning, reliability and quality on development

At the 20% budget among the 427 quiet conditions (45 later large changes): product 22, small model 23, persistence ranking 10, chance 9.5. The two learned policies are indistinguishable; both roughly double the naive rule.

Conformal coverage is on target pooled (0.807 at nominal 0.80; 0.899 at 0.90) but varies from 0.50 to 0.96 across held-out dates — batch shift is visible directly in the coverage. Intervals are wide: ~3.0 log2 units at 80%. Abstention at the locked threshold retains 655 of 819 conditions with MAE 0.678 against 2.210 for those declined.

![Development: recall against review budget, all conditions and quiet conditions](figures/gate_warning_development.png)

![Development: observed coverage of nominal 80% intervals on each held-out date](figures/gate_coverage_development.png)

![Development: predicted difficulty against realized error, with the locked abstention threshold](figures/gate_abstention_development.png)

The control-dispersion flag fired on exactly the 42 conditions of the one plate group whose day-12 controls had collapsed (flagged MAE 4.64 versus 0.79). The replicate-spread flag showed *no* association with error (0.76 flagged versus 1.01 unflagged) — reported because it is evidence against one of our own display elements.

---

## 7. The sealed evaluation

Scored once, on 224 conditions and 32 unseen chemicals across three batches. Every date is reported.

### 7.1 Forecast: the primary criterion failed

| Method | 20170920 | 20171004 | 20171011 | Mean |
| --- | ---: | ---: | ---: | ---: |
| **Carry day 7 forward** | **0.424** | 0.680 | 0.722 | **0.6088** |
| Target-only Ridge | | | | 0.6948 |
| Day-7 model + reliability indicators | 0.655 | 0.917 | 0.724 | 0.7657 |
| **Product model** | 0.583 | 1.057 | **0.674** | 0.7716 |
| Activity/coordination | 0.600 | 1.107 | 0.694 | 0.8005 |
| Control reference | 0.863 | **0.604** | 1.278 | 0.9150 |

![Single reserve scoring: per-date and mean date-macro MAE](figures/gate_reserve_comparison.png)

**C1a FAILED. C1b passed** (better than the small model on all three dates).

Why persistence won is not mysterious, and it is a lesson about the covariate the field should report. The reserved batches are far more *persistent* than development: the day-7-to-day-12 correlation of the relative readout is 0.87 on the reserve against 0.54 on development, and 13.8% of reserve conditions show a later large change against 21.4% on development. A model trained where early readings are weakly informative was deployed where they are strongly informative, and a rule that copies the early reading forward won. This is distribution shift in the difficulty of the task itself, not in the inputs alone, and no amount of input-space validation would have caught it.

### 7.2 Warning: passed, with an honest caveat

| Policy | All conditions (31 positives) | Quiet conditions (10 positives) |
| --- | ---: | ---: |
| Product forecast | 22 detected, 24 false alerts | **6 detected**, 33 false alerts |
| Activity/coordination | 22, 24 | 6, 33 |
| Rank by day-7 change | 20, 26 | **0**, 39 |
| Chance | 6.4 | 2.1 |

![Reserve: recall against review budget](figures/gate_warning_reserve.png)

**C2a and C2b passed.** The caveat matters: all ten quiet-condition positives lie on 20171004, the batch whose day-7 controls were zero — which makes every day-7 relative value identically zero there, so the naive rule is *definitionally* blind and its 0/10 is not a fair defeat so much as an impossibility. The learned ranking did find 6, against 2.1 expected by chance, on the batch where its own point forecasts were worst. Ranking worked where magnitudes did not. This is one batch and ten positives, and we say so.

### 7.3 Reliability: passed, with a stated shortfall

| Interval | Coverage | Width | 20170920 | 20171004 | 20171011 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scaled, nominal 0.80 | **0.714** | 2.32 | 0.61 | 0.82 | 0.71 |
| Symmetric, nominal 0.80 | 0.884 | 2.75 | 0.98 | 0.82 | 0.84 |
| Scaled, nominal 0.90 | 0.893 | 3.13 | 0.85 | 0.95 | 0.88 |
| Symmetric, nominal 0.90 | 0.942 | 5.81 | 0.98 | 0.88 | 0.98 |

C3's 0.70 bar is met, but 0.714 against a nominal 0.80 is a **shortfall** under batch and chemical shift, and we report it as one rather than as a pass. The symmetric intervals over-cover because they carry the harder development error scale onto easier batches; the scaled intervals are better centred and narrower at 90%.

![Reserve: observed coverage on each reserved date](figures/gate_coverage_reserve.png)

Abstention is the component that transferred most cleanly:

| | Product model | Persistence |
| --- | ---: | ---: |
| All 224 conditions | 0.784 | 0.595 |
| Kept (192) | **0.446** | 0.433 |
| Declined (32) | 2.813 | 1.566 |

Declining 14.3% of conditions removes 51% of total forecast error. Critically, it improves the *simple rule* as well (0.433 versus 0.595) — the difficulty model is estimating whether the outcome is predictable, not merely where our particular model is weak. It does not reverse the ranking: persistence remains slightly better on kept conditions too.

![Reserve: predicted difficulty against realized error; the declined conditions carry half the error](figures/gate_abstention_reserve.png)

### 7.4 Measurement quality: the result we would most like a reviewer to check

On 20171004, all six plates had a day-7 zero-dose control median of exactly zero network spikes (ranges 0–3, 0–0, 0–1, 0–1, 0–3, 0–0). By day 12 the same plates had recovered to medians of 23.5–53.5. The cultures had simply not begun firing when the input measurement was taken, so every control-relative day-7 feature is degenerate; the model extrapolated to forecasts near −5.0 for conditions whose outcomes were near +1, and eight of its ten largest reserve errors are on this date.

The `low_reference_activity` flag — threshold fixed on development, computed from day-7 controls alone — fired on **all 84 conditions of that batch and no others** (flagged MAE 1.057 versus 0.620 unflagged). The batch-level verdict the tool prints is *"day-7 reference activity is too low for this batch; forecasts here are unreliable and a day-12 measurement is recommended."*

The other three flags: `replicates_disagree` fired on 3 conditions, `controls_disagree` and `few_controls` on none. No reserve plate had a day-12 control median below 5, so the outcome-quality sensitivity subset is the whole reserve and that analysis is identical to the primary result.

---

## 8. The tool

`neuroforecast_triage.py` takes an audited well-level CSV of day-7 observations and their zero-dose controls. Day-5 rows are optional; **day-9 and day-12 rows are rejected**, so a forecast cannot be produced with information the researcher would not have. It emits:

1. A **batch verdict** from the day-7 controls, with the reference activity level.
2. A **per-condition table**: forecast, 80% interval, persistence and the small model side by side, predicted difficulty, and a verdict of *forecast usable* or *declined: measure day 12 directly*.
3. **Two ranked lists** at the validated budget, the early-warning list first.
4. The **held-out evidence**, printed every run, including the sentence that the simple rule was more accurate.

### 8.1 A batch that passes, and one that does not

Real output, from the two sealed-reserve inputs that ship with the tool. First, the batch whose cultures were developing normally:

```
Batch 20171011: 56 conditions, day-7 reference activity 33.3 network spikes
  measurement quality: day-7 measurement quality is usable
  2 conditions carry a quality flag; 8 forecasts declined; 39 still quiet at day 7

  Early warning: quiet at day 7, forecast to change (8 of 39 conditions)
     #  chemical                         uM    day 7  forecast       80% interval  persistence  verdict
     1  Fenitrothion                     10    -0.05     +0.22     [-0.36, +0.80]        -0.05  forecast usable
     2  Rotenone - ToxCast G-8       0.0003    +0.20     +0.17     [-0.36, +0.71]        +0.20  forecast usable
     3  Flufenacet                        3    -0.15     +0.17     [-0.35, +0.69]        -0.15  forecast usable

  Largest forecast change overall (12 of 56 conditions)
     1  Valinomycin                   0.001    -4.90     -5.15     [-8.96, -1.34]        -4.90  declined: measure day 12 directly
     2  Valinomycin                   0.003    -4.90     -5.15     [-8.96, -1.34]        -4.90  declined: measure day 12 directly
```

The contrast between the two lists is the design argument. The second is filled with Valinomycin and Rotenone at doses that had already silenced the network by day 7 — the day-7 column reads `-4.90` and no model is needed to see it. The first list holds conditions sitting at `-0.05`, `+0.20`, `-0.15`, which a researcher scanning the plate would pass over.

Now the batch whose cultures had not started firing:

```
Batch 20171004: 84 conditions, day-7 reference activity 0.0 network spikes
  measurement quality: day-7 reference activity is too low for this batch;
                       forecasts here are unreliable and a day-12 measurement is recommended
  84 conditions carry a quality flag; 18 forecasts declined; 76 still quiet at day 7

  Early warning: quiet at day 7, forecast to change (16 of 76 conditions)
     #  chemical                         uM    day 7  forecast       80% interval  persistence  verdict
     1  Tamoxifen                        20    +0.00     -5.00     [-9.96, -0.03]        +0.00  declined: measure day 12 directly
     2  Diphenhydramine                  20    +0.00     -5.00     [-9.94, -0.06]        +0.00  declined: measure day 12 directly
     3  Bisphenol AF                     20    +0.00     -5.00     [-9.96, -0.03]        +0.00  declined: measure day 12 directly
```

Every day-7 value is exactly `+0.00`, because the plate controls were zero and the ratio is degenerate. The model, as expected, produces extreme forecasts. Every surfaced condition is marked **declined**, and the batch verdict tells the researcher to measure day 12 rather than rely on any of it. The tool reaches the correct operational conclusion on the batch where its own forecast is worst — which is the behaviour we were trying to buy.

### 8.2 Provenance printed at every run

The footer of every run states the held-out evidence, including the sentence a vendor would omit:

```
Held-out evidence (3 experiment dates, 224 conditions, chemicals unseen in training):
  forecast error 0.772 versus persistence 0.609 - the simple rule was more accurate
  keeping only trusted forecasts: 0.446 versus 0.784 for all conditions
  intervals observed 71.4% coverage at a nominal 80% level
```

Two real inputs from the sealed reserve ship with it. Building the tool changed its design once: the naive "largest forecast change" list filled with conditions already dead at day 7 — which the day-7 column shows directly — so the early-warning list was promoted to the primary output, matching the subgroup where the held-out evidence actually exists.

---

## 9. What we would do differently, and what comes next

**Shift in task difficulty, not just inputs.** The single most useful diagnostic we lacked was a day-7-computable estimate of *how persistent a batch is*. The reserve's day-7-to-day-12 correlation (0.87) against development's (0.54) explains the outcome entirely, and a batch-level persistence estimate would let the tool choose between the model and the simple rule per batch rather than globally. That selector would itself need a sealed test, and our reserve is spent.

**The difficulty model deserved to be the primary hypothesis.** It is the component that transferred, it helps both predictors, and we specified it as supporting apparatus. A protocol built around selective prediction — retained-case error at a fixed retention rate as the primary metric — would have been a better-aimed experiment, and is what we would pre-register next.

**One event is not evidence for a flag.** Each quality flag rests on a single batch. Flags of this kind need many batches with independently known outcome quality, which means either a larger public corpus or a prospective collaboration.

**What would make this deployable.** A second independent study, ideally on human iPSC-derived neurons and on a perfused device, to test whether the day-7 quality rules and the difficulty model survive a change of preparation and platform; and a cost model for the actual laboratory decision, so a review budget follows from consequences rather than from a round 20%.

## 10. Limitations

- **The forecast is not better than a trivial rule** on the sealed batches. Any deployment should run persistence alongside and prefer it absent evidence otherwise.
- **Three batches from one study** is a weak holdout for generalization, and dates are not verified independent donors or litters.
- **The warning evidence rests on ten positives in a single batch**, and that batch is the degenerate one.
- **Intervals under-covered** (0.714 at nominal 0.80) and are wide (~2.3–3.1 log2 units).
- **Each quality flag has at most one supporting event.** The control-dispersion flag's development success and the low-activity flag's reserve success are each a single batch; the replicate-spread flag showed no association with error at all.
- **The pseudocount-regularized target** can be inflated by near-zero controls; we quantify but do not eliminate this.
- **Rat cortical cultures in plates** are not organ chips, human cells, or clinical toxicity. Transfer is untested.
- **No claim of saved days, wells or cost** is validated.
- **Forecast magnitudes are not calibrated probabilities.**

---

## 11. Reproduction

Everything is CPU-only, needs no account or paid service, and runs in minutes.

```bash
git clone https://github.com/SaltyTaro/neuroforecast && cd neuroforecast
pip install -r tools/requirements-neuroforecast.txt
python -X utf8 -B tools/neuroforecast_triage.py --input examples/reserve_20171004_low_quality_day5_day7.csv
python -X utf8 -B tools/verify_published_numbers.py
python -X utf8 -B -m unittest discover -s tests -v
```

`verify_published_numbers.py` recomputes all 33 headline numbers in the README and reports directly from the per-case evidence tables — never from a summary — at the precision each is quoted to, and fails if any claim has drifted. Writing it caught two real inconsistencies in our own drafts, both corrected and both recorded in the git history. The 26 tests include end-to-end checks that the shipped tool reproduces the audited reserve forecasts, intervals, trust verdicts and quality flags to 1e-12, and that removing day-5 rows changes none of them.

Full reproduction from the public archive: `fetch_epa_data.py` (checksum-verified) then the four gate stages. `develop` takes ~8.4 minutes single-threaded and is bit-reproducible; `reserve` takes 12 seconds, requires the unchanged lock and an explicit `--unseal`, and refuses to run twice. A nine-cell notebook walks through the whole result and was validated by executing every cell from an empty directory.

The repository ships the three hashed protocols, the final models, the two example inputs, and the audited per-case predictions, intervals and flags for both stages.

---

## 12. Disclosure

Data: EPA network formation assay (U.S. public domain, cited above). Code: MIT. Models trained only on the public development split. No private, proprietary or request-only data was used. No cloud training or paid compute.

Prior campaign work: two earlier approaches — adaptive dose selection from cell-image profiles, and vessel-patch selection for an oxygen-transport model — were built, evaluated against pre-registered gates, and **failed them**. Their reports are shipped in `docs/history/`. Nothing from those was reused as a contribution here.

AI assistance: Claude (Anthropic) was used throughout as a coding and drafting assistant. All numerical results were produced by the scripts in this repository, run on public data; every number in this report is checked by `verify_published_numbers.py` against a saved evaluation output.

---

## What we would tell another team

The most valuable thing this project produced is not a model. It is the sequence that made a negative result legible: hash the criteria before looking, hold out batches *and* identities, measure the optimism of your own selection procedure, score the holdout once, and write a script that recomputes your paper's numbers from your evidence tables so that prose cannot drift from data.

We started intending to claim a better forecast. The protocol we wrote to defend that claim is what disproved it, and what found the three components that survived. A tool that declines to answer 14% of the time, and says so on the batch where it would otherwise have been confidently wrong, is more useful in a laboratory than one that is 3.7% better on a development average.
