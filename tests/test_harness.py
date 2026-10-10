"""Tests for the validation harness, the scorer and the split-optimism study.

Each design must separate exactly what it claims to separate, the scorer must reproduce the v3 study's reference
numbers, and the split study's shipped summary must recompute from its per-case table with its lock intact.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import neuroforecast_harness as h  # noqa: E402
import neuroforecast_split_study as study  # noqa: E402

STUDY = ROOT / "evaluation/harness/split_study"


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = h.load_benchmark()

    def shared(self, design: str, repetition: int | None) -> tuple[int, int]:
        """Folds whose training set shares a batch / a chemical with the test set."""
        cases = self.bench.cases
        batches = chemicals = 0
        for _, train, test in h.splits(cases, design, repetition):
            batches += bool(set(cases.date.iloc[train]) & set(cases.date.iloc[test]))
            chemicals += bool(set(cases.identity.iloc[train]) & set(cases.identity.iloc[test]))
        return batches, chemicals

    def test_each_design_separates_what_it_claims(self):
        for repetition in (None, 3):
            self.assertEqual(self.shared("batch_chemical", repetition), (0, 0))
            self.assertEqual(self.shared("batch", repetition)[0], 0)
            self.assertGreater(self.shared("batch", repetition)[1], 0)
            self.assertEqual(self.shared("chemical", repetition)[1], 0)
            self.assertGreater(self.shared("chemical", repetition)[0], 0)
            self.assertEqual(self.shared("random_condition", repetition), (h.FOLDS, h.FOLDS))

    def test_every_condition_is_tested_once(self):
        for design in h.DESIGNS:
            tested = np.concatenate([test for _, _, test in h.splits(self.bench.cases, design, 1)])
            self.assertEqual(sorted(tested.tolist()), list(range(len(self.bench.cases))))

    def test_reference_split_is_the_v3_outer_folds_and_folds_csv(self):
        np.testing.assert_array_equal(h.fold_assignment(self.bench.cases, "batch_chemical"), self.bench.cases.outer_fold)
        folds = pd.read_csv(ROOT / "evaluation/harness/folds.csv", dtype={"identity": str})
        self.assertEqual(folds.case_id.tolist(), self.bench.cases.case_id.tolist())
        for design in h.DESIGNS:
            np.testing.assert_array_equal(folds[f"fold__{design}"], h.fold_assignment(self.bench.cases, design))

    def test_inner_splits_stay_inside_the_outer_training_set(self):
        cases = self.bench.cases
        _, train, test = next(h.splits(cases, "batch_chemical", 0))
        for _, itrain, itest in h.inner_splits(cases, "batch_chemical", train, 0, 0):
            self.assertTrue(set(itrain) <= set(train) and set(itest) <= set(train))
            self.assertFalse(set(cases.identity.iloc[itrain]) & set(cases.identity.iloc[itest]))


class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = h.load_benchmark()
        cls.example = pd.read_csv(ROOT / "examples/harness/v2_reference_predictions.csv")

    def test_reference_numbers(self):
        result = h.score(self.bench, self.example)
        self.assertEqual((result["conditions"], result["batches"], result["chemicals"]), (2243, 41, 243))
        self.assertAlmostEqual(result["date_macro_mae"], 0.6903273818784471, places=12)
        self.assertAlmostEqual(result["against"]["carry_forward"]["date_macro_mae"], 1.0160398080840376, places=12)
        self.assertEqual(result["against"]["carry_forward"]["batches_better"], 37)
        self.assertEqual(result["against"]["v2"]["batches_better"], 0)
        self.assertAlmostEqual(result["intervals"]["pooled"], 0.801, places=3)

    def test_calibrated_curve_matches_the_posthoc_baseline(self):
        import posthoc_calibrated_day7 as calibrated
        posthoc = calibrated.study_cases().set_index("case_id").calibrated_day7
        ours = pd.Series(h.calibrated_day7(self.bench.cases, "batch_chemical"), index=self.bench.cases.case_id).reindex(posthoc.index)
        np.testing.assert_allclose(ours, posthoc, rtol=0, atol=1e-12)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "no finite prediction"):
            h.score(self.bench, self.example.iloc[:-5])
        with self.assertRaisesRegex(ValueError, "repeat a case_id"):
            h.score(self.bench, pd.concat([self.example, self.example.iloc[:1]]))

    def test_command_line(self):
        done = subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / "tools/score_predictions.py"), "--example"],
                              capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("date-macro MAE 0.690", done.stdout)
        done = subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / "tools/score_predictions.py"),
                               "--predictions", str(ROOT / "evaluation/harness/folds.csv")], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(done.returncode, 2)
        self.assertNotIn("Traceback", done.stderr)


class SplitStudyTests(unittest.TestCase):
    def test_lock_matches_the_files(self):
        lock = json.loads((STUDY / "lock.json").read_text(encoding="utf-8"))
        for relative, digest in lock["sha256"].items():
            self.assertEqual(study.sha256(ROOT / relative), digest, relative)

    def test_summary_recomputes_from_the_per_case_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copyfile(STUDY / "oof.csv.gz", Path(tmp) / "oof.csv.gz")
            again = study.analyze(Path(tmp))
        shipped = json.loads((STUDY / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(json.loads(json.dumps(again)), shipped)
        self.assertTrue(all(shipped["hypotheses"].values()))

    def test_table_covers_every_design_repetition_and_condition(self):
        table = pd.read_csv(STUDY / "oof.csv.gz", usecols=["design", "repetition", "case_id"])
        counts = table.groupby(["design", "repetition"]).case_id.nunique()
        self.assertEqual(len(counts), len(h.DESIGNS) * study.REPETITIONS)
        self.assertTrue((counts == 2243).all())


if __name__ == "__main__":
    unittest.main()
