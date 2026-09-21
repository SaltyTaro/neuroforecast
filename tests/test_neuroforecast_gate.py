"""Behavioral tests for the NeuroForecast validation gate; they use synthetic frames only."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402


def synthetic_raw(seed: int = 0) -> pd.DataFrame:
    """Two dates, two plates each, six controls and two chemicals x seven doses x three replicates per plate, DIV 5/7/9/12."""
    rng = np.random.default_rng(seed)
    cfg = data.protocol()
    rows = []
    chemical = 0
    for date in [20200101, 20200201]:
        for plate in ["P1", "P2"]:
            wells = []
            for k in range(6):
                wells.append(("C", 0.0, f"c{k}"))
            for _ in range(2):
                chemical += 1
                for dose in [0.03, 0.1, 0.3, 1, 3, 10, 30]:
                    for r in range(3):
                        wells.append((f"chem{chemical}", dose, f"w{chemical}_{dose}_{r}"))
            for div in [5, 7, 9, 12]:
                for identity, dose, well in wells:
                    row = {"date": date, "Plate.SN": f"{plate}-{date}", "well": well, "DIV": div, "dose": dose, "units": "uM", "trt": identity, "identity": None if identity == "C" else identity, "eligible_identity": identity != "C"}
                    for field in cfg["feature_readouts"]:
                        row[field] = float(rng.gamma(2.0, 5.0 * div / 5))
                    row["nAE"] = float(rng.integers(1, 17))
                    row["ns.n"] = float(rng.poisson(3 * div)) if identity == "C" else float(rng.poisson(max(0.5, 3 * div / (1 + dose))))
                    rows.append(row)
    return pd.DataFrame(rows)


class ReliabilityIndicatorTests(unittest.TestCase):
    def test_indicators_ignore_everything_after_day_seven(self):
        cfg = data.protocol()
        raw = synthetic_raw()
        base = gate.reliability_indicators(raw, cfg)
        mutated = raw.copy()
        later = mutated.DIV > 7
        for column in cfg["feature_readouts"]:
            mutated.loc[later, column] = -1e6
        pd.testing.assert_frame_equal(gate.reliability_indicators(mutated, cfg), base, check_exact=True)
        pd.testing.assert_frame_equal(gate.reliability_indicators(raw.loc[raw.DIV <= 7], cfg), base, check_exact=True)

    def test_indicators_have_expected_values(self):
        cfg = data.protocol()
        raw = synthetic_raw()
        out = gate.reliability_indicators(raw, cfg)
        self.assertTrue((out["rel__rep_count_d7"] == 3).all())
        self.assertTrue((out["rel__plate_ctrl_count_d7"] == 6).all())
        self.assertTrue((out["rel__plates_spanned"] == 1).all())
        self.assertTrue(out["rel__rep_sd_log2_ns_d7"].notna().all())
        day7 = raw.loc[(raw.DIV == 7) & (raw.dose == 0)]
        expected = np.arcsinh(day7.groupby(["date", "Plate.SN"])["ns.n"].median())
        first = out.index[0]
        plate = raw.loc[(raw.date == first[0]) & (raw.trt == first[1]), "Plate.SN"].iloc[0]
        self.assertAlmostEqual(out.loc[first, "rel__plate_ctrl_median_ns_d7"], expected.loc[(first[0], plate)])

    def test_outcome_quality_uses_day_twelve_controls_only(self):
        cfg = data.protocol()
        raw = synthetic_raw()
        quality = gate.outcome_quality(raw, cfg)
        day12 = raw.loc[(raw.DIV == 12) & (raw.dose == 0)].groupby(["date", "Plate.SN"])["ns.n"].median()
        first = quality.index[0]
        plate = raw.loc[(raw.date == first[0]) & (raw.trt == first[1]), "Plate.SN"].iloc[0]
        self.assertEqual(quality.loc[first], day12.loc[(first[0], plate)])


class MetricTests(unittest.TestCase):
    def test_conformal_quantile_is_finite_sample_corrected(self):
        scores = np.arange(1, 10, dtype=float)
        # n = 9, level 0.8 -> ceil(10 * 0.8) = 8th smallest
        self.assertEqual(gate.conformal_quantile(scores, 0.8), 8.0)
        # level 0.95 -> ceil(10 * 0.95) = 10 > n -> largest
        self.assertEqual(gate.conformal_quantile(scores, 0.95), 9.0)

    def test_conformal_symmetric_coverage_on_exchangeable_data(self):
        rng = np.random.default_rng(1)
        calibration = rng.normal(size=2000)
        test = rng.normal(size=20000)
        q = gate.conformal_quantile(abs(calibration), 0.8)
        self.assertAlmostEqual(float((abs(test) <= q).mean()), 0.8, delta=0.02)

    def test_warning_budget_ranking_flags_top_scores_within_each_date(self):
        frame = pd.DataFrame({
            "date": [1] * 10 + [2] * 5,
            "score": list(range(10)) + [5, 4, 3, 2, 1],
            "positive": [False] * 8 + [True, True] + [True, False, False, False, True],
            "quiet": [True] * 15,
        })
        records = gate.warning_metrics(frame, [0.2])
        pooled = next(r for r in records if r["subgroup"] == "all")["pooled"]
        # date 1: ceil(2) = 2 flagged, both positive; date 2: ceil(1) = 1 flagged (score 5), positive.
        self.assertEqual(pooled["flagged"], 3)
        self.assertEqual(pooled["detected"], 3)
        self.assertEqual(pooled["missed"], 1)
        self.assertEqual(pooled["false_alerts"], 0)
        self.assertAlmostEqual(pooled["recall"], 0.75)
        quiet = next(r for r in records if r["subgroup"] == "early_quiet")["pooled"]
        self.assertEqual(quiet, pooled)

    def test_warning_random_expectation_equals_budget_share(self):
        frame = pd.DataFrame({"date": [1] * 10, "score": np.zeros(10), "positive": [True] * 4 + [False] * 6, "quiet": [False] * 10})
        pooled = next(r for r in gate.warning_metrics(frame, [0.5]) if r["subgroup"] == "all")["pooled"]
        self.assertAlmostEqual(pooled["random_expected_recall"], 0.5)

    def test_macro_mae_averages_doses_then_chemicals_then_dates(self):
        frame = pd.DataFrame({"date": [1, 1, 1, 2], "identity": ["a", "a", "b", "c"], "prediction": [0, 0, 0, 0], "target": [1, 3, 4, 10]})
        # date 1: chemical a = 2, chemical b = 4 -> 3; date 2: 10 -> macro 6.5
        self.assertAlmostEqual(gate.macro_mae(frame), 6.5)


class ReserveGuardTests(unittest.TestCase):
    def test_reserve_refuses_without_unseal_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                gate.reserve(Path(tmp), data.DEFAULT_RAW, unseal=False)

    def test_reserve_refuses_without_development_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                gate.reserve(Path(tmp), data.DEFAULT_RAW, unseal=True)

    def test_lock_detects_code_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            hashes = gate.code_hashes()
            hashes["neuroforecast_gate.py"] = "0" * 64
            (out / "development_lock.json").write_text(json.dumps({"code_hashes": hashes, "file_sha256": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                gate.verify_lock(out)

    def test_protocol_authorizes_single_reserve_scoring_with_fixed_operating_points(self):
        plan = gate.plan()
        self.assertTrue(plan["reserve_scoring"]["authorized_by_this_protocol"])
        self.assertEqual(plan["warning_decision"]["primary_budget"], 0.2)
        self.assertEqual(plan["reliability"]["primary_level"], 0.8)
        self.assertEqual(plan["cohort"]["reserve"]["dates"], data.protocol()["reserved_dates"])


if __name__ == "__main__":
    unittest.main()
