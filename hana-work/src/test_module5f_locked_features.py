"""Focused tests for Module 5F outputs and leakage boundaries."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HANA = Path(__file__).resolve().parents[1]


class Module5FTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((HANA / "data/manifests/module5f_locked_features_manifest.json").read_text())
        cls.locked = set(cls.manifest["locked_calendar_indices"])

    def test_exact_locked_membership(self):
        self.assertEqual(self.manifest["locked_test_rows"], 241)
        self.assertEqual(self.locked, set(range(961, 1202)))

    def test_b2_is_explicitly_blocked_without_refit(self):
        self.assertEqual(self.manifest["b2_status"], "unavailable")

    def test_outputs_are_locked_only_and_finite_when_available(self):
        for item in self.manifest["summaries"]:
            frame = pd.read_csv(HANA / item["daily_path"])
            self.assertEqual(len(frame), 241)
            self.assertEqual(set(frame["calendar_index"]), self.locked)
            feature_cols = [c for c in frame if c.startswith(("word2vec_", "finbert_", "roberta_"))]
            self.assertTrue(feature_cols)
            available = frame["feature_available"].astype(bool)
            self.assertTrue(np.isfinite(frame.loc[available, feature_cols].to_numpy()).all())
            self.assertFalse(frame["calendar_index"].isin({0, 960}).any())

    def test_hashes_and_dimensions(self):
        for item in self.manifest["summaries"]:
            frame = pd.read_csv(HANA / item["daily_path"])
            self.assertEqual(item["daily_shape"][0], 241)
            self.assertEqual(item["daily_shape"][1], len(frame.columns))


if __name__ == "__main__": unittest.main()
