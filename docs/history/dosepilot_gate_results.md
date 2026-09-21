> Historical record from an earlier round of this campaign, kept for disclosure. Its run artifacts are not shipped in this repository; the links below name files in that run's output directory.

**DosePilot gate result: FAILED.** Evaluated September 19, 2026 using the protocol frozen before the first adaptive-policy run.

At four concentrations/eight treatment wells, DosePilot's image-assisted batch policy had full-grid MAE **0.02814**, versus **0.02813** for **Assay-only selection, image prediction**. The relative error reduction was **-0.02%** (negative means worse), with a descriptive paired-scaffold 95% interval of **-1.6% to 1.7%**. The predeclared requirement was at least 15% reduction with positive paired evidence and decision guardrails.

**Decision: do not advance this version to a competition product.** The strict comparison includes the predictor/acquisition ablations. Independently of that choice, the gain over the optimized fixed image-assisted design was only **1.7%**, with a descriptive 95% interval of **-3.2% to 6.9%**. It does not meet the 15% gate. The fixed design also needs one assay round instead of two.

The proposed policy selected **the same dose-rank set for all 399 compounds**: `[0, 5, 6, 7]` (399 compounds). These are zero-based ranks, so the four-concentration schedule is the lowest concentration and the three highest. Images therefore did not produce useful compound-specific dose selection in this implementation.

A separate post-gate diagnostic purchased those same concentrations in one batch and reproduced **all 399 saved predictions**. This operational simplification does not alter the frozen gate result. The complete acquisition and calculation audit verified 38,304 non-stopping replays.

The fixed-budget error path did not pass. The alternative 20% treatment-well savings path did not pass. The complete numeric decision and comparisons are in `gate_decision.json`.

This evaluates one small learned acquisition approach and its ablations, not every possible adaptive-design algorithm. The data already informed topic selection. It is a grouped follow-up on one laboratory's 2D primary-hepatocyte assays, not an untouched final test, organ-chip validation, or a prospective experiment.

| Method | Full-grid MAE | Missed-response rate | False-response rate | Rounds |
| --- | ---: | ---: | ---: | ---: |
| Assay-only selection, image prediction | 0.02813 | 1.7% | 0.0% | 2 |
| DosePilot image-assisted batch selection | 0.02814 | 1.7% | 0.0% | 2 |
| Image-assisted variance sampling | 0.02814 | 1.7% | 0.0% | 2 |
| Optimized fixed design, with images | 0.02862 | 2.5% | 0.0% | 1 |
| Assay-only variance sampling | 0.02886 | 1.7% | 0.0% | 2 |
| Image selection, assay-only prediction | 0.02886 | 1.7% | 0.0% | 2 |
| Two-round bracket heuristic | 0.02890 | 1.7% | 0.0% | 2 |
| Optimized fixed design, assay only | 0.02907 | 1.7% | 0.0% | 1 |
| Assay-only batch variance reduction | 0.02922 | 1.7% | 0.0% | 2 |
| Random second batch, with images | 0.03171 | 3.2% | 0.1% | 2 |
| Evenly spaced, with images | 0.03206 | 2.5% | 0.4% | 1 |
| Evenly spaced, assay only | 0.03263 | 3.3% | 0.0% | 1 |
| Random second batch, assay only | 0.03276 | 3.5% | 0.0% | 2 |
| Evenly spaced + Hill fit | 0.03283 | 3.3% | 0.0% | 1 |
| Bayesian Hill + space filling | 0.03440 | 3.3% | 0.0% | 2 |
| Evenly spaced + PCHIP | 0.03448 | 3.3% | 0.0% | 1 |

All rows use 399 test-compound predictions, aggregated over five scaffold folds. Random acquisition averages five fixed seeds within each compound. A response is a predeclared decrease below 0.8 of the DMSO-control signal; it is not a clinical toxicity label. Fixed designs optimized their concentration masks on development compounds only and could choose any four of eight concentrations.

!`figures/error_vs_wells.png`

!`figures/primary_budget_ablations.png`

**Stopping and uncertainty.** Calibration used separate scaffold groups and the maximum standardized error over both potential assay stages within each group. Predictions at purchased concentrations equal the measured two-well mean; confidence statements concern reconstruction of this finite measured reference, not unknown clinical or population outcomes.

| Stopping policy | Mean treatment wells | Early stops | Unresolved at cap | Both-stage bound coverage | Full-curve fallback wells |
| --- | ---: | ---: | ---: | ---: | ---: |
| DosePilot image-assisted batch selection | 8.00 | 0.0% | 100.0% | 90.0% | 16.00 |
| Assay-only batch variance reduction | 8.00 | 0.0% | 100.0% | 92.2% | 16.00 |
| Assay-only selection, image prediction | 8.00 | 0.0% | 100.0% | 90.5% | 16.00 |
| Image-assisted variance sampling | 8.00 | 0.0% | 100.0% | 90.0% | 16.00 |

The stopping target was full-grid MAE <= 0.03 at nominal 90% coverage. Unresolved cases remain in the error table and do not qualify as solved cases for the two-round savings gate. The fallback column charges enough additional treatment wells to measure all eight concentrations; doing so after the cap would require an extra assay round. Controls and imaging are separate costs, and the replay does not measure monetary or laboratory-time savings.

**Data and robustness.** The cohort contains 399 compounds in 295 scaffold groups, with eight doses and two distinct wells per dose. It uses 6,384 treatment wells from the public source archive. Duplicate physical-well rows were removed before joining the 672 image features. Normalization uses the known source-plate DMSO medians; the source contains 4,849 such control wells across 65 plates, inventoried separately.

The first threshold-crossing rank differed between the two source wells for **66 of 399 compounds**. Agreement was **83.5%**, and the average absolute difference between individual-well responses was **0.1033**. These differences include plate/batch effects. The full report therefore includes the subset with concordant first-crossing ranks as a secondary analysis rather than treating the mean curve as noise-free biological truth.

There are only **2 source-batch combinations** and **33 plate-pair groups**. Source/plate slices report variation within this study; they do not constitute independent-laboratory validation. Whole scaffold families were kept apart in fitting, design, calibration and each outer test partition.

**Verification and reproduction.** Eight behavioral tests checked immutable masked snapshots, atomic acquisition, the two-round cap, scaffold separation, image-free predictor isolation, and calibration invariance to duplicate rows. Each real-data fold additionally passed 72 hidden-data perturbation decisions, for 360 checks across the five folds. Model and protocol hashes, split manifests, per-case predictions, and complete acquisition logs are retained.

The five model-fit/calibration/test folds took **101.6 seconds** in total after data preparation. Peak process working set was **546.0 MiB**, measured by the operating system. CPU numerical libraries were limited to one thread. All data, models, logs, predictions and figures are physically under `E:\kaggle\AI4S\`; no paid compute was used.

Dependencies are recorded in `tools/requirements-dosepilot-gate.txt`; the run environment and per-fold source hashes are saved with the artifacts.

```powershell
python -X utf8 -B -m unittest discover -s tests -v
python -X utf8 -B tools/run_dosepilot_gate.py
python -X utf8 -B tools/audit_dosepilot_gate.py
python -X utf8 -B tools/report_dosepilot_gate.py
```

Completed folds are reused only when their code and protocol hashes match. Reproduce one compound with `python -X utf8 -B tools/replay_dosepilot_gate.py --compound NAME --budget 4`; choose an exact name from the saved cohort manifest.

Artifacts: [protocol](dosepilot_gate_protocol.md), `policy_summary.csv`, `paired_ablations.json`, `replicate_audit.json`, `concordant_replicate_summary.json`, and [storage migration](storage.md).
