# NeuroForecast v2: sealed external test, scored once on September 27, 2026

This is the one-time scoring of 28 EPA experiment dates from 2014–2016 that no NeuroForecast model had seen. It follows the hashed protocol [`experiments/neuroforecast_v2_protocol.json`](../experiments/neuroforecast_v2_protocol.json).

## Timeline and audit trail

| Time (UTC, September 27) | Event |
| --- | --- |
| 16:15:58 | Candidates, rules, criteria and wording fixed. Protocol SHA-256 `fe046966…0e0d` recorded in `candidates_fixed.json`. |
| 16:22:20 | Development run complete. Lock written (`development_lock.json`, SHA-256 `5317dfc5…ea4b`) with code, data, model and operating-point hashes. |
| 16:23:57 | Scoring code frozen (`external_code_lock.json`), with the hashes of the unchanged v1 tool and bundle. These match the public release byte for byte. |
| after 16:24 | `external --unseal` ran once. The audit found no external condition or date in any fit, and saved v2 forecasts replay to 8.9e-16. |

Before unsealing, the scoring code was dry-run on a pseudo-external set built from the spent reserve dates, which is development data. That dry run wrote to a separate scratch directory, not to the output directory.

## Cohort

- 983 input-eligible conditions on 24 of the 28 dates. Four dates contributed no eligible condition.
- Every condition had a usable day-12 reference, so the target-validity rule excluded nothing.
- **Primary cohort:** 927 conditions, 105 chemicals absent from every development date.
- **Secondary cohort:** 56 conditions on the 6 shared chemicals.
- Every date has at least 10 scored conditions.

## Results, primary cohort (date-macro MAE, log2 units)

| Predictor | Date-macro MAE | Dates better than persistence (of 24) | Paired date difference vs persistence, 95% |
| --- | ---: | ---: | --- |
| Carry day 7 forward (persistence) | 0.980 | — | — |
| Assume no change from control | 1.403 | — | — |
| **Arm A: frozen v1 tool, unchanged** (two inputs missing) | 0.941 | 14 | −0.130 to +0.052 |
| **Arm B: v2 product** (small activity/coordination model) | **0.735** | **19** | **−0.396 to −0.123** |
| v1 architecture refit on the same inputs (a v2 comparator) | 0.706 | — | — |

**Abstention.** Each rule declines a fixed share of conditions. "Kept" is the MAE of the conditions it keeps; the reduction is relative to all conditions.

| Arm | Declined | Kept / all, model | Reduction | Kept / all, persistence | Reduction |
| --- | ---: | --- | ---: | --- | ---: |
| A (v1 difficulty) | 120 | 0.640 / 0.907 | 29% | 0.717 / 0.932 | 23% |
| A (decline the most extreme, same count) | 120 | 0.641 / 0.907 | 29% | — | — |
| B (v2 difficulty) | 135 | 0.541 / 0.742 | 27% | 0.655 / 0.932 | 30% |

Area under the risk–coverage curve for Arm B: locked difficulty model 0.542, extremity rule 0.562, random 0.742.

**Early warning** (conditions still quiet at day 7, 20% budget, 65 later large changes):

| Ranking | Expected detections |
| --- | ---: |
| Arm A (v1 forecast) | 16.0 |
| Arm B (locked score, v1-refit forecast magnitude) | 14.0 |
| Highest dose first | 16.6 |
| Day-7 magnitude, random ties | 21.0 |
| Chance | 14.8 |

**Intervals.** Arm A observed 0.666 coverage at nominal 0.80. Arm B observed 0.702 at 0.80 and 0.812 at 0.90.

**Day-7 chemical call (D).** There are 106 EPA samples, of which 72 are active by EPA's final area-under-curve call.
- EPA's own day-7 call, applied to every sample, agrees 87.7% of the time.
- The locked policy (h = 2, c = 0.75) decides 95.3% of samples at day 7 with 89.1% agreement. Against the secondary reference (three or more active endpoints) it agrees 96.0% of the time.
- The forecast gate changed no decision: the ablation without it is identical.

**Quality flag.** No external batch had silent day-7 controls, so the day-7 flag had nothing to fire on.

## Criteria, as pre-written

| Criterion | Result | Pre-written statement |
| --- | --- | --- |
| A1 v1 vs persistence | **PASS** | The frozen v1 forecast beat carrying day 7 forward on earlier batches despite two missing inputs. The margin is small, and the paired interval includes zero. |
| A2 abstention | **PASS** | Abstention on predicted difficulty transferred, for both predictors. |
| A2b difficulty model vs extremity rule | **PASS** | By 0.001 (0.640 against 0.641), which is effectively a tie. |
| A3 quality flag | NOT TESTABLE | No external condition was flagged. |
| A4 v1 early warning | **FAIL** | The v1 early-warning ranking did not beat the dose rule (16.0 against 16.6). |
| A5 v1 interval coverage | **FAIL** | Intervals under-covered below the v1 bar (0.666 against a bar of 0.70). |
| B1 v2 vs persistence | **PASS** | A v2 forecaster beat carrying day 7 forward on earlier batches with unseen chemicals: 25% lower error, 19 of 24 dates, interval excludes zero. |
| B1b v2 vs v1 | Reported | Below Arm A's v1 (0.735 against 0.941). **Not** below v1 refit on the same inputs (0.706). The gain comes from refitting on processing-consistent data, not from a new algorithm. |
| B2 trust | **PASS** | The locked difficulty model ranks predictability better than the extremity rule (area 0.542 against 0.562). |
| B3 v2 early warning | **FAIL** | Early warning did not beat the dose rule, and it fell below chance (14.0 against 16.6 and 14.8). |
| B4 day-7 chemical call | **FAIL** | Agreement was 0.891, below the 0.90 bar. No recording-time saving is claimed. |
| B5 v2 interval coverage | **FAIL** | Intervals under-covered (0.702 against a bar of 0.75 at nominal 0.80). |

The protocol's statements say "28 earlier batches". Eligible conditions exist on 24 of the 28, and every number here is computed on those 24.

## What this changes

1. **The forecasting claim is supported externally, with the right comparators.** On 24 batches from an earlier period and 105 unseen chemicals, the locked v2 forecaster cut error by a quarter relative to carrying day 7 forward, and its interval excludes zero. The unchanged v1 tool, missing two inputs, also edged persistence out.
   - The reserve result stands and is still reported: v1 lost on three later, atypically behaving batches.
   - Two sealed tests now point in different directions for v1, and the larger one favours the model.
2. **Abstention is the most reproducible component.** It held on the v1 reserve and on both external arms, and it cut the error of kept conditions by 23–30% for every predictor. The learned difficulty model beat the extremity rule in Arm B and tied it in Arm A.
3. **Early warning in quiet conditions did not replicate.** The reserve's 6-of-10 result, on one batch, was not a general property. This claim is withdrawn.
4. **Intervals under-cover under batch shift**, at 0.67–0.70 against a nominal 0.80. This is consistent with the reserve (0.714).
5. **The day-7 chemical call is mostly EPA's own day-7 hit count.** Requiring two or more active endpoints and deferring ambiguous samples raises agreement from 87.7% to 89.1% while still deciding 95% of samples at day 7. That is short of the pre-registered 90%, and our forecast gate added nothing.
6. **The quality flag remains supported by one event** (20171004). No external batch tested it.

## Post-hoc, descriptive observation

External batches were about as persistent overall as the reserve: pooled day-7/day-12 correlation 0.84, against 0.87 on the reserve and 0.66 on development. Yet the v2 model beat persistence on 19 of 24.

The per-batch v2 advantage correlates only weakly with batch persistence (r = −0.27). So "the reserve was persistent" does not by itself explain the v1 loss. Two other factors on the reserve are more plausible:
- v1's under-reaction on the persistent batches (forecast slope 0.87–0.94 against 1.3).
- The degenerate 20171004 batch.

Large changes were also twice as common externally (27.5% of conditions against 13.8% on the reserve).

## Post hoc addendum (September 28, 2026)

After scoring, a separate script analysed the locked tool: `tools/posthoc_signal_analysis.py`, with outputs in `evaluation/v2_posthoc/`. It was not pre-registered, refits nothing on external outcomes and selects nothing. Details are in the technical report, §7.4 and §7.7–7.8.
- **Point 2 above, qualified.** At the shipped threshold, declining the 135 most extreme forecasts keeps an MAE of 0.5405, against 0.5413 for the trust verdict. On the 56-condition secondary cohort the extremity rule won on the whole curve (0.634 against 0.668). Abstention transfers; the learned score's edge over that simple rule does not.
- **Point 6 above, extended.** The tool's batch rule fires on 3 of the 41 batches v2 has seen, all in development. On two of them the forecast beat carrying day 7 forward (0.88 against 1.85, and 0.53 against 1.99). The rule marks a degenerate reference, not a failed forecast, and the tool now says so.
- **Where v2 wins.** On the 60 conditions with raised activity at day 7, v2's error was 0.47 against 1.14. Near-control conditions tie (0.476 against 0.485).
- **Intervals.** Recalibrating on the first 4 batches in time order raised coverage on the other 20 from 0.711 to 0.876, with wider intervals.

## Files

In this repository:
- `evaluation/v2_prepared/`: the prepared inputs.
- `evaluation/v2_development/`: `development_results.json`, `development_lock.json`, `dev_oof_predictions.csv`, `candidates_fixed.json` and `external_code_lock.json`.
- `models/v2/`: the final models.
- `evaluation/external/`: `results.json`, `arm_a_cases.csv`, `arm_b_cases.csv`, `decision_units.csv` and `audit.json`.
- `evaluation/v2_intake/`: the intake and equivalence reports.

Code: `tools/neuroforecast_v2.py` (prepare and develop), `tools/neuroforecast_v2_external.py` (freeze, external and audit), and `tools/reproduce_external.py` to re-run from a fresh clone.
