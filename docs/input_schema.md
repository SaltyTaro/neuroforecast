# Input schema: what the v2 tool reads

The tool reads one row per well per recording day, for days 5 and 7 only. The readouts are EPA's per-well summaries from the network formation assay, computed as in the EPA refinement release that v2 was trained on. A runnable template is [`examples/v2/template_day5_day7.csv`](../examples/v2/template_day5_day7.csv): one real held-out batch, 3 plates × 48 wells × 2 days.

## Rows and layout

| Column | Meaning | Rule |
| --- | --- | --- |
| `date` | Batch ID: the cultures plated and recorded together | integer, e.g. `20160120` |
| `Plate.SN` | Plate (or chip carrier) ID | text |
| `well` | Well ID | text; one row per well per day |
| `DIV` | Day in vitro of the recording | **5 or 7 only**. Both days are required, and days 9 and 12 are rejected. |
| `trt` | Treatment name | text; the vehicle controls may share one name |
| `dose` | Concentration | µM; **0 marks an untreated (vehicle) control** |
| `units` | Concentration unit | must be `uM` |
| `identity` | Optional chemical ID | if missing, `trt` is used |

**Replicates.** A condition is a batch, a chemical and a positive concentration. It needs at least 2 replicate wells with both day-5 and day-7 rows. The replicates may be on different plates of the same batch, as in EPA's layout. A single plate therefore usually yields no condition, and the tool says so.

**Controls.** Every plate needs zero-dose wells on both days. Conditions on plates with fewer than 4 controls are flagged `few_controls`. In a post hoc subsampling analysis, a single control per plate raised false batch verdicts; two gave none (report §10).

**Duplicates.** A repeated (date, plate, well, day) is rejected and must be audited first.

## Which component reads what

| Component | Readouts used |
| --- | --- |
| Batch verdict | `ns.n` of the day-7 controls: half or more of a batch's conditions below one control network spike |
| Forecast | 5 readouts, `ns.n`, `nAE`, `nABE`, `meanfiringrate` and `r`, at days 5 and 7, relative to same-plate controls, plus concentration |
| Trust verdict and intervals | all 17 readouts at day 7, relative to controls, with their missingness, plus control statistics and three batch summaries |

All 17 columns must be present. Eight may be blank (see below). **Leave them blank; do not fill them with zeros**, because missingness is itself an input.

## The 17 readouts

These are EPA's definitions, taken from EPA's processing code ([USEPA/CCTE_Shafer_MEA_dev_pre-processing_scripts](https://github.com/USEPA/CCTE_Shafer_MEA_dev_pre-processing_scripts)) and the `sjemea` and `meadq` R packages they build on.

**Recording.** Each recording is 15 minutes on a 16-electrode well, with spikes detected at 8× rms noise.

**Electrode classes.**
- An **active electrode (AE)** fires at least 5 spikes/min.
- An **actively bursting electrode (ABE)** has at least 0.5 bursts/min.

**Bursts** use the Maximum Interval method:
- a burst starts at an ISI below 0.1 s and ends at an ISI above 0.25 s;
- bursts closer than 0.8 s are merged;
- a burst needs at least 5 spikes and at least 0.05 s.

**Network spikes** are counted in 0.05 s bins. A bin qualifies when at least 5 electrodes fire in it and at least 4 of them fire twice or more.

**Status against Axion's Neural Metric Tool**, assuming matched spike detection and criteria:
- *equivalent*: same construct;
- *approximate*: same construct, different detector or averaging, so recalibrate;
- *different*: a related quantity, so recompute with EPA's code;
- *none*: no counterpart.

| EPA column | Definition (well value) | Units | Blank when | Nearest Axion metric | Status |
| --- | --- | --- | --- | --- | --- |
| `meanfiringrate` | spikes ÷ recording span, mean over AE | Hz | never (0 if silent) | Weighted Mean Firing Rate | equivalent |
| `burst.per.min` | bursts per minute, mean over AE | bursts/min | never | Burst Frequency × 60 | approximate |
| `mean.isis` | mean ISI within bursts, mean over ABE | s | no ABE | Mean ISI within Burst | approximate |
| `per.spikes.in.burst` | % of spikes in bursts, mean over ABE | % | never | Burst Percentage | approximate |
| `mean.dur` | mean burst duration, first to last spike, mean over ABE | s | no ABE | Burst Duration – Avg | approximate |
| `mean.IBIs` | mean gap from one burst's end to the next burst's start, mean over ABE | s | no ABE | Inter-Burst Interval (start to start) | different |
| `nAE` | number of active electrodes (of 16) | count | never | Number of Active Electrodes | equivalent |
| `nABE` | number of electrodes with at least 0.5 bursts/min | count | never | Number of Bursting Electrodes | approximate |
| `ns.n` | number of network spikes in the recording | count | never | Number of Network Bursts | different |
| `ns.peak.m` | mean number of electrodes in a network spike's peak bin | electrodes | no network spikes | Electrodes Participating in Burst | different |
| `ns.durn.m` | mean network-spike duration, full width at half maximum | s | no network spikes | Network Burst Duration – Avg | different |
| `ns.percent.of.spikes.in.ns` | % of the well's spikes that fall in network-spike peaks | % | never | Network Burst Percentage | different |
| `ns.mean.insis` | mean interval between network spikes | s | ≤ 1 network spike | none exported | none |
| `ns.durn.sd` | standard deviation of network-spike durations | s | ≤ 1 network spike | Network Burst Duration – Std | different |
| `ns.mean.spikes.in.ns` | spikes per network spike, in the peak window | spikes | no network spikes | Spikes per Network Burst | different |
| `r` | mean pairwise Pearson correlation of 10 ms binary spike trains, **over AE only**; 0 if at most one AE | unitless | never | Area Under Normalized Cross-Correlation | different |
| `mi` | normalized multi-information of ~3 ms binary trains (Ball et al. 2017) | bits | never (0 if silent) | none | none |

## Using data from other MEA software

Two readouts map directly: `meanfiringrate` and `nAE`, with Axion's activity criterion set to 5 spikes/min. The burst readouts are approximate. The nine network and coordination readouts (the `ns.*` columns, `r` and `mi`) cannot be mapped from Axion's summary metrics. They must be computed from the spike list with EPA's code.

For a chip or another platform, the practical route is:
1. export spike lists;
2. compute these 17 readouts with EPA's code;
3. run the tool;
4. recalibrate the intervals and thresholds on the lab's own first batches;
5. validate prospectively in a sealed test before trusting the forecast.

Accuracy off EPA's platform is **not** established.

## Caveats

- **`r` changed between EPA releases.** The v1 archive (2018) correlated all spiking electrodes. The refinement release, and so v2, uses active electrodes only. Supply the active-electrode version.
- **Recording length.** `ns.n` is a count per recording, not a rate. Inputs should come from recordings of about 15 minutes.
- **Detection threshold.** All readouts shift with the spike-detection threshold: EPA used 8× rms.
- **Unconfirmed points.**
  - The published papers' own parameter tables (Brown et al. 2016, Frank et al. 2017) were not accessible to us, so the definitions above follow EPA's current code.
  - Axion's default criteria are not stated in the user guides we read (Neural Metric Tool v4.1, AxIS Navigator v3.10).
