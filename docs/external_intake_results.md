# External intake, September 27, 2026

Code: `tools/external_nfa_intake.py` and `tools/external_nfa_equivalence.py`.
Outputs: `evaluation/v2_intake/`.

## Source

- **Repository:** [USEPA/CompTox-DNT-NFA-Refinement](https://github.com/USEPA/CompTox-DNT-NFA-Refinement), commit `01adf3e1a0068c87fe221d60df36b9f96c4b4b1d`.
- **File:** `source_files/All_DIV_Data.Rdata`, SHA-256 `fe8015c7…2af6`. It holds 31,756 well-DIV rows across 45 experiment dates. Removing duplicate physical rows leaves 31,600.
- **Licence:** the repository has no LICENSE file. The data were produced by the EPA, so we treat them as a US-Government work and report this limitation.
- **Reader:** the file was read with `rdata` 1.1.0, which is pinned in `tools/requirements-neuroforecast.txt`.

## Relation to our archive (17 shared dates)

This comparison used our own development and reserve data only.

- **It is a re-processing of the same recordings, not new data.**
  - 16,983 physical rows match on date, plate, well and DIV. 145 rows are only in our archive, mostly the excluded PFAS names. 877 are only in the release; 576 of those are three extra plates on 20170201.
  - Doses agree on every matched row.
  - Per-readout Pearson r is 0.988 (`r`) to 0.9997 (`ns.n`).
  - Network-spike counts differ in 12% of rows, typically by one spike. The median relative difference among rows that differ is 3.7%.
- **Readouts:** the release lacks `cv.time` and `cv.network`, which v1 uses, and adds `mi`.
- **Identity:** all 267 EPA sample IDs resolve to a DSSTox ID from the release's own tables. 191 resolve through the reference-chemical annotation and 76 through names. Our 136 CAS identities link one-to-one through shared wells.

## Equivalence of the frozen v1 tool on known data (spent reserve)

| Input | v1 date-macro MAE | Persistence | Declined | Low-activity flags |
| --- | ---: | ---: | ---: | ---: |
| Our archive (audited) | 0.7716 | 0.6088 | 32 | 84 |
| Archive with `cv.*` removed | 0.7904 | 0.6088 | 32 | 84 |
| Release copy | 0.8200 | 0.6086 | 33 | 84 |

The archive run reproduces the audited forecasts to 4e-16.

The two missing readouts move the median forecast by 0.019. They move the 95th percentile by 1.07, so a minority of conditions depend on them heavily. The quality flag and the abstention count transfer unchanged.

Arm A therefore carries a disclosed handicap of about 0.05 date-macro MAE on the forecast. Persistence is unaffected.

## Sealed external cohort (counts only; no outcome read)

- 28 dates from 2014-02-05 to 2016-06-01, 73 plate-dates, from EPA sources Brown2014 (6 dates), Frank2017 (17) and OPP2015 (5).
- 983 conditions are eligible by the input rules, on **24 dates**; four dates contribute no eligible condition. Each eligible condition has at least two treated wells with a day-12 record.
- 111 chemicals. **105 are absent from all 17 development dates**; they cover 927 conditions, and the match was checked by DSSTox ID and, independently, by CAS. Six chemicals are shared and form the secondary cohort.
- Controls per plate at day 7: 4 to 14.
- SHA-256 of the external rows: `879ca0cc…9568`.

## Disclosure

The author has seen the external dates, plate and chemical counts, chemical names and IDs, and day-7 control counts. EPA's plate-exclusion script names one external plate, MW1045-09_20141231, as having zero-valued day-12 control endpoints on 14 rows. No other day-9 or day-12 value, and no hit call, for an external chemical has been read.

Two EPA outcome tables were downloaded but not read for external chemicals: `Potency_Table_for_All_Methods_for_KC.csv`, and the AUC and DIV-12 tcplfit2 results. `Bioactivity_bin_tbl_comp_methods.csv` is read only through `neuroforecast_v2.bioactivity`, which skips unpermitted rows at read time.
