"""Tests for the day-12 check, the admission record, and the calibrated day-7 baseline.

The check must score a day-7 report with the sealed test's own definitions, so on the shipped held-out examples it
reproduces the external evidence exactly. The record must be what the browser writes (tests/test_web_day12_check.mjs
compares the browser to the same fixtures). Everything runs from a fresh clone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import neuroforecast_triage_v2 as t2  # noqa: E402
import posthoc_calibrated_day7 as calibrated  # noqa: E402
import posthoc_lab_recalibration as recalibration  # noqa: E402

EXAMPLES = {20160120: "best", 20141231: "median", 20140924: "lost"}
read = lambda path: pd.read_csv(ROOT / path, dtype={"identity": str})


class Day12CheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = t2.load_bundle()
        cls.sealed = read("evaluation/external/arm_b_cases.csv")
        cls.runs = {}
        for date in EXAMPLES:
            early, late = f"examples/v2/external_{date}_day5_day7.csv", f"examples/v2/external_{date}_day12.csv"
            table, summary = t2.triage(read(early), cls.bundle)
            checked, check = t2.day12_check(table, read(late))
            cls.runs[date] = (early, late, table, summary, checked, check)

    def test_truth_is_the_sealed_target(self):
        for date, (_, _, _, _, checked, _) in self.runs.items():
            expected = self.sealed.loc[self.sealed.date == date].set_index(["identity", "dose"])
            got = checked.set_index(["identity", "dose"]).reindex(expected.index)
            self.assertEqual(len(got), 42)
            np.testing.assert_allclose(got.day12_measured_log2, expected.target, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(got.day12_reference_usable.to_numpy(), expected.usable.astype(bool).to_numpy())

    def test_batch_errors_are_the_published_per_batch_errors(self):
        for date, (_, _, _, _, checked, check) in self.runs.items():
            rows = self.sealed.loc[(self.sealed.date == date) & self.sealed.primary.astype(bool) & self.sealed.usable.astype(bool)]
            for column, key in [("prediction", "forecast_error"), ("day7", "carry_forward_error")]:
                published = (rows[column] - rows.target).abs().groupby(rows.identity).mean().mean()
                self.assertAlmostEqual(check["batches"][0][key], published, places=12)

    def test_example_roles_follow_their_rules(self):
        table = pd.DataFrame({date: {"v2": c["batches"][0]["forecast_error"], "carry": c["batches"][0]["carry_forward_error"]}
                              for date, (*_, c) in self.runs.items()}).T
        self.assertLess(table.loc[20160120, "v2"], table.loc[20141231, "v2"])
        self.assertGreater(table.loc[20140924, "v2"], table.loc[20140924, "carry"])   # the shipped loss is shown as a loss

    def test_record_matches_fixture_and_hashes_inputs(self):
        for date, (early, late, table, summary, checked, check) in self.runs.items():
            record = t2.admission_record(checked, summary, Path(early).name, (ROOT / early).read_bytes(),
                                         check, Path(late).name, (ROOT / late).read_bytes())
            record = json.loads(json.dumps(record, allow_nan=False))
            fixture = json.loads((ROOT / f"tests/fixtures/v2_day12_record_{date}.json").read_text(encoding="utf-8"))
            for r in (record, fixture):
                r["created_utc"] = None
                r["input"]["sha256"] = None
                r["day12_check"]["input"]["sha256"] = None
            self.assertEqual(record, fixture)
            again = t2.admission_record(checked, summary, "x", (ROOT / early).read_bytes())
            self.assertEqual(again["input"]["sha256"], hashlib.sha256((ROOT / early).read_bytes()).hexdigest())
            self.assertEqual(len(record["conditions"]), 42)
            decisions = {c["decision"] for c in record["conditions"]}
            self.assertLessEqual(decisions, {"provisional forecast", "measure day 12"})

    def test_lab_calibration_matches_fixture_and_refuses_bad_history(self):
        early, late, table, summary, _, _ = self.runs[20160120]
        history = [ROOT / "app/examples/v2/record_20140924_checked.json", ROOT / "app/examples/v2/record_20141231_checked.json"]
        calibration = t2.lab_calibration([(p.name, p.read_bytes()) for p in history], {20160120})
        scaled = t2.recalibrate(table, calibration)
        pd.testing.assert_frame_equal(scaled.drop(columns=["forecast_low", "forecast_high"]), table.drop(columns=["forecast_low", "forecast_high"]))
        checked, check = t2.day12_check(scaled, read(late))
        record = t2.admission_record(checked, summary, Path(early).name, (ROOT / early).read_bytes(), check,
                                     Path(late).name, (ROOT / late).read_bytes(), calibration)
        record = json.loads(json.dumps(record, allow_nan=False))
        fixture = json.loads((ROOT / "tests/fixtures/v2_day12_record_20160120_history.json").read_text(encoding="utf-8"))
        for r in (record, fixture):
            r["created_utc"] = None
            r["input"]["sha256"] = None
            r["day12_check"]["input"]["sha256"] = None
        self.assertEqual(record, fixture)
        # The scale is the split-conformal quantile of the history's |measured - forecast| / difficulty.
        scores = sorted(abs(c["day12"]["measured_log2"] - c["forecast_log2"]) / c["difficulty"]
                        for p in history for c in json.loads(p.read_text(encoding="utf-8"))["conditions"] if c["day12"]["reference_usable"])
        self.assertEqual(calibration["scale"], scores[int(np.ceil((len(scores) + 1) * 0.8)) - 1])
        with self.assertRaisesRegex(ValueError, "at least 2 batches"):
            t2.lab_calibration([(history[0].name, history[0].read_bytes())], {20160120})
        own = ROOT / "tests/fixtures/v2_day12_record_20160120.json"
        with self.assertRaisesRegex(ValueError, "cannot calibrate itself"):
            t2.lab_calibration([(p.name, p.read_bytes()) for p in [history[0], own]], {20160120})
        unchecked = json.loads(history[0].read_text(encoding="utf-8"))
        unchecked["day12_check"] = None
        with self.assertRaisesRegex(ValueError, "no day-12 check"):
            t2.lab_calibration([("u.json", json.dumps(unchecked).encode()), (history[1].name, history[1].read_bytes())], {20160120})

    def test_day12_input_rules(self):
        with self.assertRaisesRegex(ValueError, "no day-12 recordings"):
            t2.normalize_day12(read("examples/v2/external_20160120_day5_day7.csv"), t2.v2.v2_config()["feature_readouts"])
        with self.assertRaisesRegex(ValueError, "rejects observations from days"):
            t2.triage(read("examples/v2/external_20160120_day12.csv"), self.bundle)

    def test_command_line_reports_bad_input_without_a_traceback(self):
        early = str(ROOT / "examples/v2/external_20160120_day5_day7.csv")
        for argv, message in [(["--input", str(ROOT / "no_such_file.csv")], "No such file"),
                              (["--input", str(ROOT / "examples/v2/external_20160120_day12.csv")], "rejects observations from days"),
                              (["--input", early, "--day12", early], "no day-12 recordings")]:
            done = subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / "tools/neuroforecast_triage_v2.py"), *argv],
                                  capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(done.returncode, 2, argv)
            self.assertIn(message, done.stderr)
            self.assertNotIn("Traceback", done.stderr)


class CalibratedDay7Tests(unittest.TestCase):
    def test_shipped_summary_reproduces(self):
        shipped = json.loads((ROOT / "evaluation/posthoc_baselines/calibrated_day7.json").read_text(encoding="utf-8"))
        self.assertEqual(json.loads(json.dumps(calibrated.run())), shipped)

    def test_kept_conditions_and_batch_difficulty(self):
        external = calibrated.external_cases()
        kept = calibrated.kept_external(external)
        self.assertEqual(kept["conditions"], int((~external.declined.astype(bool)).sum()))
        self.assertEqual(kept["batches"], 24)
        # The pooled gain decomposition must add up: declined share plus kept share is the whole gain.
        gain = (external.day7 - external.target).abs() - (external.prediction - external.target).abs()
        kept_share = gain[~external.declined.astype(bool)].sum() / gain.sum()
        self.assertAlmostEqual(kept["share_of_pooled_gain_from_declined"] + kept_share, 1.0, places=12)
        difficulty = calibrated.batch_difficulty(external, "trust_score", "prediction")
        self.assertEqual(difficulty["batches"], 24)
        self.assertGreater(difficulty["spearman"], 0.5)


class LabRecalibrationReplayTests(unittest.TestCase):
    def test_shipped_replay_reproduces(self):
        shipped = json.loads((ROOT / "evaluation/posthoc_baselines/lab_recalibration.json").read_text(encoding="utf-8"))
        self.assertEqual(json.loads(json.dumps(recalibration.run())), shipped)

    def test_history_holds_only_earlier_batches(self):
        cases = recalibration.external()
        for window in (None, recalibration.WINDOW):
            frame = recalibration.replay(cases, 0.8, window).groupby("date").first()
            expected = [min(i, window or i) for i in range(len(frame))]
            self.assertEqual(frame.history_batches.tolist(), expected)
            self.assertTrue(frame.lab_scale.iloc[:recalibration.MIN_BATCHES].isna().all())
            self.assertTrue(frame.lab_scale.iloc[recalibration.MIN_BATCHES:].notna().all())


if __name__ == "__main__":
    unittest.main()
