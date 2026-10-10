# Validation harness and split-optimism study

A public validation harness for forecasting the day-12 state of a neuronal MEA culture from its day-5 and day-7 recordings. It has fixed splits that hold out culture batches, a scorer (command line and [in the browser](https://saltytaro.github.io/neuroforecast/app/score.html)), and a pre-registered study of how much weaker splits overstate accuracy.

## What it contains

| Path | What it is |
| --- | --- |
| `evaluation/v3/prepared/cases.csv`, `features.csv` | Every eligible condition of the 41 public EPA batches: 2,285 conditions, 2,243 with a usable day-12 target, 243 chemicals. Features use days 5 and 7 only; the rows are aligned. |
| `evaluation/harness/folds.csv` | One fold column per design (`fold__random_condition`, `fold__chemical`, `fold__batch`, `fold__batch_chemical`). The reference split is `batch_chemical`, which equals the v3 study's outer folds. |
| `evaluation/harness/reference_predictions.csv` | Carry-forward, the calibrated day-7 curve, v2 and the v2 + transformer average on the reference split. |
| `tools/neuroforecast_harness.py` | Loading, splits (`splits`, `inner_splits`), baselines and metrics (`score`). |
| `tools/score_predictions.py`, `app/score.html` | The scorer, on the command line and in the browser (no upload). They agree exactly except for the bootstrap's random stream. |
| `examples/harness/v2_reference_predictions.csv` | A valid predictions file: v2's forecasts with their 80% intervals. |

**What it is not.** Both sealed tests are spent, so this is a **development harness, not a sealed test**. Its labels are public, so every score is self-reported. Use it to compare methods and validation designs under a fixed split, not to claim a new sealed result.

## The four designs

| Design | Test and training share | Use it for |
| --- | --- | --- |
| `random_condition` | culture batches and chemicals | nothing you will report |
| `chemical` | culture batches | measuring how much batch sharing flatters you |
| `batch` | chemicals tested on other batches | separating batch shift from chemical novelty |
| `batch_chemical` | neither, but each fold mixes years (see below) | **the reference estimate for a new run with new chemicals** |

Training must use only the training positions of each fold, and only rows with `usable == True`. In `batch_chemical`, every chemical tested on a test batch is removed from training.

## Scoring

```bash
python -X utf8 -B tools/score_predictions.py --predictions my_forecasts.csv           # reference split
python -X utf8 -B tools/score_predictions.py --predictions my_forecasts.csv --design chemical
python -X utf8 -B tools/score_predictions.py --example
```

Or open [`app/score.html`](https://saltytaro.github.io/neuroforecast/app/score.html) and load the same file.

A predictions CSV needs `case_id` and `prediction` (log2 day-12 network spikes relative to same-plate controls) for every usable condition. You can add `low` and `high` for an interval at `--level` (default 0.8), and `decline` (1 where your method refuses to forecast).

The scorer reports:
- date-macro MAE;
- the paired 95% batch interval and batch wins against each baseline;
- for intervals, pooled, batch-mean and worst-fifth coverage, and the share of batches covering at least 0.1 below nominal (70% at 80%);
- for refusals, the error of kept and declined conditions.

On a design other than the reference, the baselines are refitted under that design, so the comparison is fair. The absolute errors are still optimistic.

## Every model scored so far on the reference split

These are the models of the locked v3 study (`docs/neuroforecast_v3_results.md`), all on the same folds. The study chose among them on these folds, so the best of them is optimistic as a group; its nested estimate, which counts the selection, is 0.695.

| Model | Date-macro MAE |
| --- | ---: |
| v2 + batch-context transformer, averaged | 0.668 |
| Extra Trees on the dose series | 0.679 |
| **v2 recipe (shipped)** | **0.690** |
| Batch-context transformer | 0.711 |
| LightGBM, all features | 0.718 |
| TabPFN v2, all features | 0.723 |
| MLP | 0.862 |
| Ridge | 0.880 |
| Calibrated day-7 curve | 0.881 |
| Carry day 7 forward | 1.016 |

## The split-optimism study

The protocol is `experiments/neuroforecast_split_study_protocol.json`, and the code is `tools/neuroforecast_split_study.py`. The protocol, the harness and the study code were committed publicly (commit `9a90926`, pushed 2026-10-08 06:41:57 UTC by GitHub's record) before the first run, and locked at 06:42:12 UTC.

Two caveats:
- **The protocol's `written` field is wrong.** It says 06:45 UTC, three minutes after the push. It was typed by hand. The commit and the push record are the times to trust; the file cannot be corrected without breaking its lock.
- **A one-model probe preceded the protocol** and is disclosed in it. It had already shown about 15% optimism, so H1's 5% bar tested whether the effect survives the locked design and repetitions, not whether it exists. With constant-width intervals, H3 largely follows from H1.

**Design.**
- The four designs, with 9 outer and 4 inner folds and 5 random fold repetitions each.
- Five models on v2's 13 inputs with fixed hyperparameters: carry-forward, a calibrated day-7 curve, ridge, LightGBM and the v2 Extra Trees recipe.
- Constant-width split-conformal intervals from inner out-of-fold residuals, calibrated under the same design.

**Outputs** (`evaluation/harness/split_study/`): per-case out-of-fold predictions and interval half-widths (`oof.csv.gz`), `summary.json` and `lock.json`. `analyze` recomputes the summary from `oof.csv.gz` alone.

| Date-macro MAE | Random conditions | Chemicals held out | Batches held out | Batches + chemicals |
| --- | ---: | ---: | ---: | ---: |
| Extra Trees (v2 recipe) | 0.598 | 0.615 | 0.688 | 0.702 |
| Calibrated day-7 curve | 0.855 | — | — | 0.887 |
| Optimism, Extra Trees | 15% | 12% | 2% | — |
| Transfer coverage at 0.80, Extra Trees | 0.747 | 0.761 | 0.796 | 0.80 (by construction) |

All three pre-registered hypotheses passed:
- **H1:** the chemical design understates error by at least 5%. Measured 12%, with a paired interval of −0.139 to −0.045.
- **H2:** random conditions do the same. Measured 15%, interval −0.157 to −0.058.
- **H3:** intervals validated under either design cover at most 0.77 on new batches. Measured 0.747 and 0.761.

**Further findings:**
- Sharing culture batches is what flatters a model; holding out batches alone leaves 2% optimism.
- Flexible models are flattered most: 11–15%, against 4–5% for ridge and the calibrated curve. The Extra Trees edge over the calibrated curve is 29–30% under random or chemical splits and 21% under batch and chemical hold-out.
- The ranking barely moves. Extra Trees and LightGBM trade first place by 0.001 under chemical hold-out and 0.012 under batch and chemical hold-out; otherwise the order is the same in every design.

To re-run the study, copy `evaluation/harness/split_study/lock.json` into an empty folder and run `python -X utf8 -B tools/neuroforecast_split_study.py run --out <that folder>`. It takes about 8 minutes on a CPU, needs LightGBM (pinned in `tools/requirements-neuroforecast.txt`), and refuses to run if the protocol, the harness or the study script differs from the locked hashes. The results are deterministic: compare the new `oof.csv.gz` and `summary.json` with the shipped ones.

## Post hoc: the reference split is optimistic too, because folds mix years

Added after the study, in response to a reviewer; not pre-registered (`tools/posthoc_temporal_split.py`, predictions in `evaluation/posthoc_baselines/temporal_split_predictions.csv.gz`). The reference folds mix batches from 2014 to 2017, so a model is often trained on batches recorded after its test batch. A lab only ever has earlier ones.

| v2 recipe, date-macro MAE | Interleaved (reference folds) | Same number of training batches, any date | Other period or earlier batches only |
| --- | ---: | ---: | ---: |
| 24 batches of 2014–16, trained on the 17 of 2016–17 (the sealed external design) | 0.659 | 0.678 | **0.731** |
| 31 batches, each trained on every earlier batch (at least 10) | 0.647 | 0.658 | **0.686** |

- **Across periods the reference split is 10% optimistic** (paired interval −0.113 to −0.034), and 7 points remain with the number of training batches matched. The two periods also differ in EPA's processing, so "period" means time and pipeline together.
- **The learned model's edge over the calibrated curve falls from 20% to 11%.** The curve barely changes across periods (0.823 to 0.817); the trees do.
- **Forward chaining within the data costs 6%,** and the interval includes zero.

If you report a harness score, report the period design beside it.
