"""Post hoc analyses added after the third mock review: each shipped JSON reproduces from the shipped evidence.

The temporal split's model refits take about two minutes, so these tests re-analyse its shipped per-condition
predictions instead and check the purge rule on a small table.
"""

import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import posthoc_decision_value as decision  # noqa: E402
import posthoc_lab_recalibration as recalibration  # noqa: E402
import posthoc_temporal_split as temporal  # noqa: E402


def shipped(name: str) -> dict:
    return json.loads((ROOT / "evaluation/posthoc_baselines" / name).read_text(encoding="utf-8"))


class TemporalSplit(unittest.TestCase):
    def test_analysis_reproduces_the_shipped_summary(self):
        table = pd.read_csv(temporal.PREDICTIONS, dtype={"identity": str})
        self.assertEqual(json.loads(json.dumps(temporal.analyze(table))), shipped("temporal_split.json"))

    def test_designs_test_the_batches_their_rules_name(self):
        table = pd.read_csv(temporal.PREDICTIONS, dtype={"identity": str})
        dates = sorted(table.date.unique())
        period = sorted(table.loc[table["period__extra_trees_v2"].notna()].date.unique())
        forward = sorted(table.loc[table["forward__extra_trees_v2"].notna()].date.unique())
        self.assertEqual(period, [d for d in dates if d < temporal.PERIOD_SPLIT])
        self.assertEqual((len(period), len(forward), forward[0]), (24, 31, dates[temporal.MIN_HISTORY]))
        self.assertTrue(table["interleaved__extra_trees_v2"].notna().all())

    def test_training_rows_drop_test_chemicals_and_other_batches(self):
        cases = pd.DataFrame({"date": [1, 1, 2, 2, 3, 3], "identity": ["a", "b", "a", "c", "c", "d"]})
        usable = np.array([True, True, True, True, True, False])
        rows = temporal.fit_rows(cases, usable, [1, 3], test=np.array([2, 3]))
        self.assertEqual(rows.tolist(), [1])   # batch 1 chemical b; a and c are tested, batch 3 row d is unusable


class DecisionValue(unittest.TestCase):
    def test_reproduces_the_shipped_summary(self):
        self.assertEqual(json.loads(json.dumps(decision.run())), shipped("decision_value.json"))

    def test_bands_partition_the_conditions(self):
        w = decision.run()["where_ml_helps"]
        self.assertEqual(sum(b["conditions"] for b in w["bands"].values()), w["all"]["conditions"])
        b = decision.run()["binary_call"]
        arm = pd.read_csv(ROOT / "evaluation/external/arm_b_cases.csv")
        arm = arm.loc[arm.primary.astype(bool) & arm.usable.astype(bool)]
        self.assertEqual(sum(b["large_changes"].values()), int((arm.target.abs() >= decision.LARGE).sum()))


class Recalibration(unittest.TestCase):
    def test_reproduces_the_shipped_replay_with_oracles(self):
        result = recalibration.run()
        self.assertEqual(json.loads(json.dumps(result)), shipped("lab_recalibration.json"))
        for level, entry in result["levels"].items():
            self.assertGreaterEqual(entry["oracle_constant_nominal"]["pooled"], float(level))
            for rule in ("expanding", "last_6_batches"):
                same = entry[rule]["oracle_constant_same_width"]
                self.assertAlmostEqual(same["mean_width"], entry[rule]["lab"]["mean_width"], places=9)


class ChipIngestOutput(unittest.TestCase):
    def test_new_spike_files_need_an_output_folder(self):
        run = subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / "tools/chip_ingest.py"), "--spikes", "x.csv"],
                             capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--spikes needs --out", run.stderr)


if __name__ == "__main__":
    unittest.main()
