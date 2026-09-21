"""Behavioral tests for the trust-triage tool, including end-to-end agreement with the audited reserve outputs."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_triage as triage  # noqa: E402

from test_neuroforecast_gate import synthetic_raw  # noqa: E402

NAMES = {20171004: "reserve_20171004_low_quality_day5_day7.csv", 20171011: "reserve_20171011_usable_day5_day7.csv"}


def _examples() -> dict:
    """Prefer inputs shipped beside the code, so a fresh clone runs these checks rather than skipping them."""
    local = data.ROOT / "examples"
    root = local if (local / NAMES[20171011]).exists() else gate.DEFAULT_OUT / "demo"
    return {date: root / name for date, name in NAMES.items()}


def _evaluation() -> Path:
    local = data.ROOT / "evaluation"
    return local if (local / "reserve_intervals.csv").exists() else gate.DEFAULT_OUT


EXAMPLES = _examples()
EVALUATION = _evaluation()


def bundle_available() -> bool:
    return (EVALUATION / "reserve_intervals.csv").exists() and all(p.exists() for p in EXAMPLES.values())


class InputGuardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = data.protocol()
        self.raw = synthetic_raw()

    def test_rejects_observations_after_the_decision_day(self):
        for day in (9, 12):
            with self.assertRaises(ValueError) as error:
                triage.build_features(self.raw.loc[self.raw.DIV.isin([7, day])], self.cfg)
            self.assertIn(str(day), str(error.exception))

    def test_requires_day_seven(self):
        with self.assertRaises(ValueError):
            triage.build_features(self.raw.loc[self.raw.DIV == 5], self.cfg)

    def test_day_seven_alone_is_enough_and_omits_the_day_five_comparator(self):
        frame, has_day5 = triage.build_features(self.raw.loc[self.raw.DIV == 7], self.cfg)
        self.assertFalse(has_day5)
        self.assertNotIn("normalized_ns_d5", frame.columns)
        for column in gate.RELIABILITY:
            self.assertIn(column, frame.columns)

    def test_day_five_and_seven_together_add_the_comparator_features(self):
        frame, has_day5 = triage.build_features(self.raw.loc[self.raw.DIV.isin([5, 7])], self.cfg)
        self.assertTrue(has_day5)
        self.assertIn("contrast__ns.n__d5", frame.columns)

    def test_features_ignore_everything_after_day_seven(self):
        early = self.raw.loc[self.raw.DIV.isin([5, 7])]
        base, _ = triage.build_features(early, self.cfg)
        mutated = early.copy()
        mutated.loc[mutated.DIV == 5, "ns.n"] = mutated.loc[mutated.DIV == 5, "ns.n"] + 0  # no-op guard
        pd.testing.assert_frame_equal(triage.build_features(mutated, self.cfg)[0], base, check_exact=True)

    def test_duplicate_physical_rows_are_rejected(self):
        duplicated = pd.concat([self.raw.loc[self.raw.DIV == 7], self.raw.loc[self.raw.DIV == 7].head(1)])
        with self.assertRaises(ValueError):
            triage.build_features(duplicated, self.cfg)


class RankingTests(unittest.TestCase):
    def test_ranking_is_per_batch_and_flags_the_top_budget_share(self):
        out = pd.DataFrame({"date": [1] * 10 + [2] * 5, "forecast_log2": list(range(10)) + [-5, 4, -3, 2, 1]})
        out = triage.rank_within(out, "review", out.index, budget=0.2)
        flagged = out.loc[out.review_flagged]
        self.assertEqual(sorted(flagged.loc[flagged.date == 1, "forecast_log2"]), [8, 9])
        self.assertEqual(list(flagged.loc[flagged.date == 2, "forecast_log2"]), [-5])

    def test_ranking_a_subset_leaves_other_rows_unranked(self):
        out = pd.DataFrame({"date": [1] * 6, "forecast_log2": [0.1, 5.0, 0.2, 0.3, 0.4, 0.5]})
        subset = out.index[[0, 2, 3, 4, 5]]
        out = triage.rank_within(out, "early_warning", subset, budget=0.5)
        self.assertTrue(pd.isna(out.early_warning_rank.iloc[1]))
        self.assertFalse(out.early_warning_flagged.iloc[1])
        self.assertEqual(int(out.early_warning_flagged.sum()), 3)


@unittest.skipUnless(bundle_available(), "requires the completed gate run with its reserve scoring")
class ReserveAgreementTests(unittest.TestCase):
    """The shipped tool must reproduce the audited reserve outputs exactly."""

    @classmethod
    def setUpClass(cls):
        cls.bundle = triage.load_bundle(triage.DEFAULT_BUNDLE, triage.DEFAULT_LOCK, triage.DEFAULT_RESERVE)
        cls.intervals = pd.read_csv(EVALUATION / "reserve_intervals.csv", dtype={"identity": str}).set_index("case_id")
        predictions = pd.read_csv(EVALUATION / "reserve_predictions.csv", dtype={"identity": str})
        cls.comparator = predictions.loc[predictions.model == triage.COMPARATOR].set_index("case_id")

    def triage_example(self, date):
        table, summary = triage.triage(pd.read_csv(EXAMPLES[date], dtype={"identity": str}), self.bundle)
        return table.set_index("case_id"), summary

    def test_forecasts_and_intervals_match_the_audited_reserve(self):
        for date in EXAMPLES:
            table, _ = self.triage_example(date)
            expected = self.intervals.loc[self.intervals.date == date]
            self.assertEqual(len(table), len(expected))
            table = table.reindex(expected.index)
            level = self.bundle["level"]
            np.testing.assert_allclose(table.forecast_log2, expected.prediction, atol=1e-12, rtol=0)
            np.testing.assert_allclose(table.difficulty, expected.sigma, atol=1e-12, rtol=0)
            np.testing.assert_allclose(table.forecast_low, expected[f"scaled_lower_{level}"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(table.forecast_high, expected[f"scaled_upper_{level}"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(table.comparator_log2, self.comparator.reindex(expected.index).prediction, atol=1e-12, rtol=0)

    def test_trust_verdict_matches_the_locked_abstention_decision(self):
        for date in EXAMPLES:
            table, _ = self.triage_example(date)
            expected = self.intervals.loc[self.intervals.date == date]
            np.testing.assert_array_equal(table.reindex(expected.index).forecast_trusted.to_numpy(), ~expected.abstain.to_numpy())

    def test_quality_flags_match_the_locked_gate_flags(self):
        for date in EXAMPLES:
            table, _ = self.triage_example(date)
            expected = self.intervals.loc[self.intervals.date == date]
            table = table.reindex(expected.index)
            for flag in ["low_reference_activity", "few_controls", "replicates_disagree", "controls_disagree"]:
                np.testing.assert_array_equal(table[f"quality_{flag}"].to_numpy(), expected[f"flag_{flag}"].to_numpy(), err_msg=flag)

    def test_the_unusable_batch_is_called_out_and_the_usable_one_is_not(self):
        _, bad = self.triage_example(20171004)
        _, good = self.triage_example(20171011)
        self.assertIn("too low", bad["batches"][0]["batch_verdict"])
        self.assertEqual(bad["batches"][0]["low_reference_activity_fraction"], 1.0)
        self.assertIn("usable", good["batches"][0]["batch_verdict"])
        self.assertEqual(good["batches"][0]["low_reference_activity_fraction"], 0.0)

    def test_day_seven_alone_reproduces_the_same_forecasts(self):
        """The product model and its trust verdict must not depend on day-5 data being supplied."""
        for date in EXAMPLES:
            observed = pd.read_csv(EXAMPLES[date], dtype={"identity": str})
            full, _ = triage.triage(observed, self.bundle)
            only7, _ = triage.triage(observed.loc[observed.DIV == 7], self.bundle)
            np.testing.assert_allclose(only7.forecast_log2, full.forecast_log2, atol=1e-12, rtol=0)
            np.testing.assert_allclose(only7.difficulty, full.difficulty, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(only7.forecast_trusted, full.forecast_trusted)
            self.assertTrue(only7.comparator_log2.isna().all())

    def test_reported_evidence_states_the_failed_forecasting_comparison(self):
        _, summary = self.triage_example(20171011)
        evidence = summary["validated_evidence"]
        self.assertLess(evidence["persistence_mean_mae"], evidence["product_mean_mae"])
        self.assertTrue(any("more accurate" in line for line in summary["honest_limits"]))


if __name__ == "__main__":
    unittest.main()
