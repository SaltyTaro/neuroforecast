> Historical record from an earlier round of this campaign, kept for disclosure. Its run artifacts are not shipped in this repository; the links below name files in that run's output directory.

**Round 2 result, September 20, 2026 IST: OxyGuard failed its selection gate.** Adding transport-model sensitivity improved the primary error by only **0.76%** over the strongest simpler review rule. The predefined internal target was 20%. Do not advance this version as the main competition entry.

The hypothesis was to select vessel-segmentation patches for manual correction according to their effect on a downstream oxygen model. The experiment used **36 public murine tissue images across six tissue types**, real automatic segmentation outputs, and curated reference masks. Each tissue type was held out in turn. This is an exploratory software/method benchmark. Its oxygen fields are **model-derived**, and it is **not validation on living organ chips**.

At four corrected patches, or **6.25% of image area**, the proposed policy produced full-field MAE **0.014344**, compared with **0.014454** for selection by predicted mask change. Their absolute difference was only **0.000110** in normalized surrogate units. A descriptive paired bootstrap over the six tissue groups gave a relative-improvement interval of **-5.40% to +9.09%**. Only **3/6** tissues improved.

| Four-patch policy | Field MAE | Absolute error of mean field |
| --- | ---: | ---: |
| OxyGuard: transport + learned error | 0.014344 | 0.011260 |
| Predicted mask change | 0.014454 | 0.010742 |
| Expected pixel error | 0.014653 | 0.010418 |
| Entropy | 0.014835 | 0.010654 |
| Algorithm disagreement | 0.015658 | 0.011288 |
| Spatial coverage | 0.016195 | 0.011719 |
| random 0 | 0.016308 | 0.011726 |
| random 3 | 0.016471 | 0.011806 |
| Transport sensitivity only | 0.016489 | 0.012097 |
| random 4 | 0.016792 | 0.012033 |
| random 2 | 0.016813 | 0.012194 |
| random 1 | 0.016859 | 0.012225 |

These are errors of a computed field against that same model evaluated on a curated mask. They are not clinical accuracy, measured oxygen error, blood-flow accuracy, or cell-survival error. Random rows are five seeds on the same 36 images, not independent cohorts.

!`figures/review_budget.png`

**What did work.** The held-out learned hard mask had zero-review field MAE **0.017694**, versus **0.022802** for the supplied REAVER output. Four proposed corrections reduced error by **18.9%** relative to the initial learned mask. A simpler mask-change rule achieved nearly all of that gain. The soft-probability zero-review baseline was **0.020614**, so the proposed review did beat that alternative. The failed claim is the additional benefit of transport-aware selection.

| Registered criterion | Observed result | Decision |
| --- | --- | --- |
| At least 20% lower error than strongest review baseline | 0.76% | Fail |
| Absolute error reduction at least 0.005 | 0.000110 | Fail |
| Improvement in at least five tissue groups | 3/6 | Fail |
| Positive lower paired-bootstrap endpoint | -5.40% | Fail |
| Beat zero-review soft probabilities | 0.014344 versus 0.020614 | Pass |

!`figures/tissue_comparison.png`

**Sensitivity checks do not rescue the result.** Keeping the selected correction patches fixed and changing the surrogate's diffusion length gave:

| Diffusion length (micrometers) | Strongest simpler rule | Proposed relative improvement |
| ---: | --- | ---: |
| 15 | expected_mask_change | +3.00% |
| 30 | expected_mask_change | +0.76% |
| 60 | expected_mask_change | -4.35% |

The proposed rule is worse under the 60-micrometer scenario. A separate numerical-resolution check on the alphabetically first image of each tissue compared 64- and 128-grid calculations. Results are in `grid_sensitivity.csv`. This checks numerical behavior; it does not validate the assumed physiology.

**Audit and runtime.** Five scientific contract tests passed, covering exact constant solutions, physical bounds/monotonicity, finite-difference sensitivity, hidden-reference invariance, and full-review recovery. The saved-run audit reproduced **432** primary replays, with a maximum metric difference of 1e-16, and passed **864** real-image hidden-reference decision checks. It verified tissue separation, unique test membership, distinct patch purchases, and artifact hashes. There are still only 36 images and six tissue groups. Animal identifiers are absent; images and pixels must not be presented as independent animals.

The six folds took **134.5 seconds** in total, excluding data preparation and the audit. The audit took **21.5 seconds**. Computation used the local CPU, with four forest workers and one BLAS thread. No paid service or GPU was used.

!`figures/representative_replay.png`

**Interpretation and limits.** The model assumes a maintained source in every segmented vessel and solves steady diffusion/reaction. It does not test hydraulic connectivity, flow, patency, dynamic exposure, or biological oxygen delivery. The images are downsampled to 256 pixels for segmentation and a 64-grid transport calculation. Reference correction is assumed perfect, and patch area is not measured human labor. The reference masks were curated from an automatic initialization. These constraints are explicit in [the protocol](oxyguard_pilot_protocol.md).

This result rejects the current proposed mechanism. It does not rule out all function-aware review, graph-based flow models, or vascular-chip research. It also does not justify changing the gate threshold or promoting the strongest baseline as a novel winner. The [second-round topic assessment](round2_topic_selection.md) records the broader search and data-access findings.

Reproduce from the archived public inputs with:

```powershell
python -X utf8 -B -m unittest discover -s tests -p test_oxyguard_pilot.py -v
python -X utf8 -B tools/oxyguard_pilot.py
python -X utf8 -B tools/audit_oxyguard_pilot.py
python -X utf8 -B tools/report_oxyguard_pilot.py
```

Pinned packages are in [requirements-oxyguard-pilot.txt](../tools/requirements-oxyguard-pilot.txt). The [source catalog](round2_sources.json), `data_audit.json`, `gate.json`, and `replay_audit.json` preserve provenance. Protocol SHA-256: `f0a77e9988bfd4f7e60e341e5329587acf67b466bb5a9e8ef61025a0f491c6e6`.

All raw downloads are on `E:\kaggle\AI4S\raw\research\round2`. Data, models, predictions, traces, audits, and PNG/SVG figures are on `E:\kaggle\AI4S\experiments\oxyguard_pilot_v1`.
