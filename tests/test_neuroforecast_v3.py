"""Leakage and purge checks for the NeuroForecast v3 study (run with the v3 environment).

They use the prepared v3 benchmark when it exists and synthetic wells otherwise.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import neuroforecast_v3_data as v3d  # noqa: E402

# The copy shipped in the public repository, so every machine tests what judges download; the author's study
# folder is used only when the shipped copy is absent.
EVIDENCE = next((d for d in [Path(__file__).resolve().parents[1] / "evaluation/v3", v3d.OUT]
                 if (d / "prepared/manifest.json").exists()), None)
PREPARED = EVIDENCE is not None


def synthetic_wells(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for plate in ["P1", "P2"]:
        for i in range(12):
            well = f"{'ABCDEF'[i // 2]}{1 + i % 2}"
            control = i < 4
            row = {"date": 20200101, "Plate.SN": plate, "well": well, "dose": 0.0 if control else [0.1, 1.0, 10.0, 30.0][i % 4],
                   "identity": None if control else ("CHEM_A" if i < 8 else "CHEM_B"), "eligible": not control,
                   "trt": "DMSO" if control else ("A" if i < 8 else "B"), "spid": "x"}
            for day in v3d.DAYS:
                for f in v3d.READOUTS:
                    row[f"{f}__d{day}"] = float(rng.gamma(2.0, 10.0))
            rows.append(row)
    return pd.DataFrame(rows)


class SplitTests(unittest.TestCase):
    @unittest.skipUnless(PREPARED, "v3 benchmark not prepared")
    def test_outer_and_inner_purge(self):
        cases = v3d.load(EVIDENCE)["cases"]
        seen_test = set()
        for k, dates, train, test in v3d.outer_splits(cases):
            self.assertFalse(set(cases.identity.iloc[train]) & set(cases.identity.iloc[test]))
            self.assertEqual(set(cases.date.iloc[test]), set(dates))
            self.assertFalse(set(cases.date.iloc[train]) & set(dates))
            seen_test |= set(test.tolist())
            covered = set()
            for j, idates, itrain, itest in v3d.inner_splits(cases, train):
                self.assertTrue(set(itrain) <= set(train) and set(itest) <= set(train))
                self.assertFalse(set(cases.identity.iloc[itrain]) & set(cases.identity.iloc[itest]))
                covered |= set(itest.tolist())
            self.assertEqual(covered, set(train.tolist()))
        self.assertEqual(seen_test, set(range(len(cases))))

    @unittest.skipUnless(PREPARED, "v3 benchmark not prepared")
    def test_context_matches_dates(self):
        loaded = v3d.load(EVIDENCE)
        x, cases = loaded["features"], loaded["cases"]
        for col in v3d.CONTEXT:
            if col == "ctx_plate_ctrl_growth_ns":
                continue
            per_date = x[col].groupby(cases.date).nunique(dropna=False)
            self.assertTrue((per_date <= 1).all(), col)


class PCTTests(unittest.TestCase):
    def setUp(self):
        try:
            import neuroforecast_v3_pct as pct  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_inputs_ignore_later_days(self):
        import neuroforecast_v3_pct as pct
        wells = synthetic_wells()
        x1, _ = pct.well_inputs(wells)
        mutated = wells.copy()
        for f in v3d.READOUTS:
            for day in (9, 12):
                mutated[f"{f}__d{day}"] = mutated[f"{f}__d{day}"] * 7 + 3
        x2, _ = pct.well_inputs(mutated)
        np.testing.assert_array_equal(x1, x2)

    def test_purged_labels_are_masked(self):
        import torch
        import neuroforecast_v3_pct as pct
        wells = synthetic_wells()
        keys = wells.loc[wells.dose > 0, ["date", "identity", "dose"]].drop_duplicates().reset_index(drop=True)
        cases = keys.assign(target=0.5, usable=True)
        features = pd.DataFrame({c: 0.1 for c in pct.COND_COLUMNS}, index=range(len(cases)))
        features["log10_dose"] = np.log10(cases.dose)
        twin = pd.DataFrame(np.ones((len(cases), 3)), columns=["a", "b", "c"])
        batch = pct.build_batches(wells, cases, features, twin)[20200101]
        staged = pct.Staged([batch], pct.Normalizer([batch]), {"CHEM_A"}, torch.device("cpu"))
        c_n = len(batch.case_pos)
        weight, tw, aux = staged.w[0, :c_n].numpy(), staged.twin[0, :c_n].numpy(), staged.aux[0].numpy()
        purged = batch.cond_identity == "CHEM_A"
        self.assertTrue(purged.any() and (~purged).any())
        self.assertTrue((weight[purged] == 0).all() and (weight[~purged] == 1).all())
        self.assertTrue(np.isnan(tw[purged]).all() and np.isfinite(tw[~purged]).all())
        self.assertTrue(np.isnan(aux[: len(batch.plate)][batch.identity == "CHEM_A"]).all())
        self.assertTrue(np.isfinite(aux[: len(batch.plate)][batch.identity == ""]).any())

    def test_trains_and_predicts_every_condition(self):
        import torch
        import neuroforecast_v3_pct as pct
        batches = []
        for seed in range(3):
            w = synthetic_wells(seed)
            w["date"] = 20200101 + seed
            keys = w.loc[w.dose > 0, ["date", "identity", "dose"]].drop_duplicates().reset_index(drop=True)
            cases = keys.assign(target=0.3, usable=True)
            features = pd.DataFrame({c: 0.1 for c in pct.COND_COLUMNS}, index=range(len(cases)))
            twin = pd.DataFrame(np.ones((len(cases), 3)), columns=["a", "b", "c"])
            batches.append(pct.build_batches(w, cases, features, twin)[20200101 + seed])
        for flags in [{}, {"context": False}, {"relational": False}, {"cond_features": "small"}]:
            cfg = pct.Config(epochs=3, batch_dates=2, d=16, **flags)
            out = pct.fit_predict(batches[:2], batches[2:], {"CHEM_B"}, cfg, [1, 2], torch.device("cpu"))
            self.assertEqual(len(out), len(batches[2].case_pos))
            self.assertTrue(all(np.isfinite(v).all() for v in out.values()))


if __name__ == "__main__":
    unittest.main()
