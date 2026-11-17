"""Focused Module 6C tests."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from module6c_final_evaluation import HANA, metric_dict


class Module6CTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((HANA / "data/manifests/module6c_final_evaluation_manifest.json").read_text())
        cls.results = pd.read_csv(HANA / cls.manifest["metrics_path"])

    def test_locked_test_isolation(self):
        self.assertEqual(self.manifest["locked_test_calendar_rows"], 241)
        self.assertTrue(self.manifest["no_refit"])
        for record in self.manifest["artifacts"]:
            pred = pd.read_csv(HANA / record["prediction_path"])
            self.assertTrue(pred["calendar_index"].between(961, 1200).all())

    def test_prediction_alignment_and_unique_rows(self):
        for path in self.results["prediction_path"]:
            pred = pd.read_csv(HANA / path)
            self.assertFalse(pred["calendar_index"].duplicated().any())
            self.assertEqual(len(pred), len(pred["calendar_index"].unique()))

    def test_metric_calculation(self):
        result = metric_dict(np.array([False, True, True, False]), np.array([False, True, False, False]), np.array([0.1, 0.9, 0.4, 0.2]))
        self.assertAlmostEqual(result["mcc"], 0.577350269, places=6)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (2, 0, 1, 1))

    def test_artifact_integrity_columns(self):
        required = {"experiment", "timing_mode", "model", "mcc", "macro_f1", "balanced_accuracy", "accuracy"}
        self.assertTrue(required <= set(self.results.columns))
        self.assertTrue(self.results["artifact_sha256"].str.len().eq(64).all())


if __name__ == "__main__": unittest.main()
