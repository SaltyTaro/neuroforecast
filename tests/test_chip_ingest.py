"""Tests for chip ingestion: EPA-style readouts from spike times, on synthetic wells with known answers.

The example recordings (Zenodo 4679064) are not shipped, so the shipped summary is checked for internal consistency
only; the readout code is checked on spike trains built to have a known number of bursts and network spikes.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import chip_ingest as ci  # noqa: E402


def synchronous_well(electrodes: int = 16, events: int = 30, duration: float = 900.0) -> list[np.ndarray]:
    """Every electrode fires a 6-spike burst (20 ms apart) at the same 30 times, plus nothing else."""
    times = np.linspace(10, duration - 10, events)
    burst = np.arange(6) * 0.02
    return [np.sort((times[:, None] + burst[None, :] + 0.001 * k).ravel()) for k in range(electrodes)]


class ReadoutTests(unittest.TestCase):
    def test_counts_rates_and_network_spikes(self):
        r = ci.well_readouts(synchronous_well(), 900.0)
        self.assertEqual(r["nAE"], 16)                      # 180 spikes in 15 minutes = 12 per minute, above 5
        self.assertAlmostEqual(r["meanfiringrate"], 180 / 900)
        self.assertEqual(r["nABE"], 16)                     # 30 bursts in 15 minutes = 2 per minute, above 0.5
        self.assertAlmostEqual(r["burst.per.min"], 2.0)
        self.assertAlmostEqual(r["per.spikes.in.burst"], 100.0)
        self.assertAlmostEqual(r["mean.isis"], 0.02, places=6)
        self.assertAlmostEqual(r["mean.dur"], 0.10, places=6)
        self.assertEqual(r["ns.n"], 30)                     # every event is one network spike
        self.assertEqual(r["ns.peak.m"], 16)
        self.assertGreater(r["r"], 0.3)

    def test_silent_and_sparse_wells(self):
        silent = ci.well_readouts([np.array([]) for _ in range(16)], 900.0)
        self.assertEqual((silent["nAE"], silent["nABE"], silent["ns.n"]), (0, 0, 0))
        self.assertEqual((silent["meanfiringrate"], silent["r"]), (0.0, 0.0))
        sparse = ci.well_readouts([np.array([1.0, 500.0]) for _ in range(16)], 900.0)
        self.assertEqual(sparse["nAE"], 0)                  # 2 spikes in 15 minutes is far below 5 per minute

    def test_bursts_follow_the_maximum_interval_rules(self):
        too_short = np.array([0.0, 0.01, 0.02, 0.03])        # 4 spikes: below the 5-spike minimum
        self.assertEqual(ci.bursts(too_short), [])
        two_close = np.concatenate([np.arange(5) * 0.02, 0.5 + np.arange(5) * 0.02])   # gap 0.42 s < 0.8 s: merged
        self.assertEqual(ci.bursts(two_close), [(0, 9)])
        two_apart = np.concatenate([np.arange(5) * 0.02, 2.0 + np.arange(5) * 0.02])   # gap > 0.8 s: two bursts
        self.assertEqual(ci.bursts(two_apart), [(0, 4), (5, 9)])

    def test_rescaling_to_an_epa_well(self):
        scaled = ci.to_epa_scale({"ns.n": 10, "nAE": 19, "nABE": 9.5, "r": 0.3}, duration=300.0, electrodes=19)
        self.assertEqual(scaled["ns.n"], 30.0)
        self.assertEqual(scaled["nAE"], 16.0)
        self.assertEqual(scaled["nABE"], 8.0)
        self.assertEqual(scaled["r"], 0.3)


class ShippedSummaryTests(unittest.TestCase):
    def test_summary_matches_the_shipped_tables(self):
        base = ROOT / "evaluation/chip_ingestion"
        summary = json.loads((base / "zenodo_4679064_summary.json").read_text(encoding="utf-8"))
        table = pd.read_csv(base / "zenodo_4679064_readouts.csv")
        domain = pd.read_csv(base / "zenodo_4679064_domain.csv")
        self.assertEqual((summary["recordings"], summary["wells"]), (10, 40))
        self.assertEqual(summary["readouts_missing"], ["mi"])
        self.assertTrue((table.electrodes == 19).all())      # every compartment of the four-compartment chip has 19 electrodes
        reference = table.groupby("recording")["ns.n"].median()
        self.assertEqual(summary["reference_check"]["usable_recordings"], int((reference >= 1).sum()))
        self.assertAlmostEqual(summary["domain_check"]["share_inside_overall"], domain.inside_epa_1_99.dropna().astype(bool).mean())
        self.assertIn("ingestion only", summary["status"])


if __name__ == "__main__":
    unittest.main()
