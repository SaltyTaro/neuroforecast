"""Tests for the v2 external test: locked artifacts, exact replays, input rules, and cohort separation.

All of these run from a fresh clone: they use the shipped models, prepared features, per-case evidence and
fixtures, and never need the EPA download.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import neuroforecast_data as data  # noqa: E402
import neuroforecast_gate as gate  # noqa: E402
import neuroforecast_triage_v2 as t2  # noqa: E402
import neuroforecast_v2 as v2  # noqa: E402

LOCK = json.loads((ROOT / "evaluation/v2_development/development_lock.json").read_text(encoding="utf-8"))
RESULTS = json.loads((ROOT / "evaluation/external/results.json").read_text(encoding="utf-8"))


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, dtype={"identity": str, "spid": str})


class LockedArtifacts(unittest.TestCase):
    def test_release_code_and_models_are_the_locked_bytes(self):
        paths = {"neuroforecast_v2.py": "tools/neuroforecast_v2.py", "neuroforecast_v2_protocol.json": "experiments/neuroforecast_v2_protocol.json",
                 "external_nfa_intake.py": "tools/external_nfa_intake.py", "neuroforecast_data.py": "tools/neuroforecast_data.py",
                 "neuroforecast_benchmark.py": "tools/neuroforecast_benchmark.py", "neuroforecast_representation.py": "tools/neuroforecast_representation.py",
                 "neuroforecast_gate.py": "tools/neuroforecast_gate.py"}
        for name, path in paths.items():
            self.assertEqual(data.digest(ROOT / path), LOCK["code"][name], name)
        for name, digest in LOCK["models"].items():
            self.assertEqual(data.digest(ROOT / "models/v2" / name), digest, name)
        frozen = json.loads((ROOT / "evaluation/v2_development/external_code_lock.json").read_text(encoding="utf-8"))
        self.assertEqual(data.digest(ROOT / "tools/neuroforecast_v2_external.py"), frozen["external_code_sha256"])

    def test_candidates_were_fixed_before_the_lock(self):
        fixed = json.loads((ROOT / "evaluation/v2_development/candidates_fixed.json").read_text(encoding="utf-8"))
        self.assertEqual(fixed["sha256"], LOCK["protocol_sha256"])
        self.assertLess(fixed["utc"], LOCK["locked_utc"])


class Separation(unittest.TestCase):
    def test_no_external_row_or_chemical_reaches_development(self):
        dev, ext = read("evaluation/v2_prepared/dev_cases.csv"), read("evaluation/v2_prepared/ext_cases.csv")
        self.assertFalse(set(dev.date) & set(ext.date))
        self.assertFalse(set(dev.case_id) & set(ext.case_id))
        primary = ext.loc[ext.primary.astype(bool)]
        self.assertFalse(set(primary.identity) & set(dev.identity))
        self.assertEqual(len(primary), 927)
        self.assertEqual(primary.identity.nunique(), 105)


class Replays(unittest.TestCase):
    def test_saved_models_reproduce_the_scored_external_forecasts(self):
        features = pd.read_csv(ROOT / "evaluation/v2_prepared/ext_features.csv")
        locked = LOCK["locked"]
        base = {}
        for name in ["v1_refit", "small_refit", "g", "g_batch"]:
            saved = joblib.load(ROOT / "models/v2" / f"final_{name}.joblib")
            base[name] = saved["model"].predict(features[saved["columns"]])
        params = dict(locked["final_params"], blend=tuple(locked["final_params"]["blend"]))
        forecast = v2.combine(base, features, params)[locked["product"]]
        scored = read("evaluation/external/arm_b_cases.csv")
        np.testing.assert_allclose(forecast, scored.prediction.to_numpy(), atol=1e-12, rtol=0)

    def test_tool_reproduces_its_reference_outputs_from_raw_wells(self):
        table, summary = t2.triage(read("tests/fixtures/v2_external_day5_day7.csv"), t2.load_bundle())
        reference = read("evaluation/v2_reference/external_triage_v2.csv").set_index("case_id")
        table = table.set_index("case_id").loc[reference.index]
        for column in ["forecast_log2", "forecast_low", "forecast_high", "difficulty"]:
            np.testing.assert_allclose(table[column], reference[column], atol=1e-12, rtol=0)
        self.assertTrue((table.forecast_trusted == reference.forecast_trusted).all())
        self.assertEqual(len(table), 983)

    def test_tool_forecasts_equal_the_scored_arm_b_forecasts(self):
        reference = read("evaluation/v2_reference/external_triage_v2.csv").set_index("case_id")
        scored = read("evaluation/external/arm_b_cases.csv").set_index("case_id").loc[reference.index]
        np.testing.assert_allclose(reference.forecast_log2, scored.prediction, atol=1e-12, rtol=0)
        self.assertTrue((reference.forecast_trusted == ~scored.declined.astype(bool)).all())


class InputRules(unittest.TestCase):
    def setUp(self):
        self.wells = read("examples/v2/external_20160120_day5_day7.csv")

    def test_later_days_are_rejected(self):
        late = self.wells.loc[self.wells.DIV == 7].assign(DIV=12)
        with self.assertRaises(ValueError):
            t2.triage(pd.concat([self.wells, late]), t2.load_bundle())

    def test_day5_is_required(self):
        with self.assertRaises(ValueError):
            t2.triage(self.wells.loc[self.wells.DIV == 7], t2.load_bundle())

    def test_collapsed_batch_gets_the_measure_day12_verdict(self):
        _, summary = t2.triage(read("examples/v2/reserve_20171004_low_quality_day5_day7.csv"), t2.load_bundle())
        self.assertIn("too low", summary["batches"][0]["batch_verdict"])
        _, healthy = t2.triage(self.wells, t2.load_bundle())
        self.assertIn("usable", healthy["batches"][0]["batch_verdict"])

    def test_template_has_the_documented_columns_and_runs(self):
        template = pd.read_csv(ROOT / "examples/v2/template_day5_day7.csv")
        self.assertEqual(list(template.columns), t2.REQUIRED + v2.v2_config()["feature_readouts"])
        table, _ = t2.triage(template, t2.load_bundle())
        full, _ = t2.triage(read("examples/v2/external_20160120_day5_day7.csv"), t2.load_bundle())
        merged = table.merge(full, on=["name", "dose"], suffixes=("", "_full"))
        self.assertEqual(len(merged), len(full))
        np.testing.assert_array_equal(merged.forecast_log2, merged.forecast_log2_full)

    def test_single_replicate_input_gets_a_clear_error(self):
        one_plate = self.wells.loc[self.wells["Plate.SN"] == sorted(self.wells["Plate.SN"].unique())[0]]
        with self.assertRaisesRegex(ValueError, "replicate wells"):
            t2.triage(one_plate, t2.load_bundle())


class Results(unittest.TestCase):
    def test_headline_comparison_recomputes_from_the_case_tables(self):
        b = read("evaluation/external/arm_b_cases.csv")
        primary = b.loc[b.primary.astype(bool) & b.usable.astype(bool)]
        product = gate.macro_mae(primary)
        persistence = gate.macro_mae(primary.assign(prediction=primary.day7))
        stored = RESULTS["arm_B"]["primary"]["forecast_vs_persistence"]
        self.assertAlmostEqual(product, stored["a_date_macro"], places=12)
        self.assertAlmostEqual(persistence, stored["b_date_macro"], places=12)
        self.assertEqual(RESULTS["criteria"]["B1"]["result"], "PASS")


if __name__ == "__main__":
    unittest.main()
