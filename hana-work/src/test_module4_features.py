"""Module 4 focused correctness, coverage, and leakage tests."""

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from module4_features import (
    AVAILABILITY_FEATURES,
    TECHNICAL_FEATURES,
    add_task2_daily_derived_features,
    aggregate_basic_nlp,
    assert_predictor_whitelist,
    build_availability,
    build_rate_features_exact,
    load_master_calendar,
)


ROOT = Path(__file__).resolve().parents[2]


def task2_reference_functions():
    source = (ROOT / "src/task2_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"compute_rsi", "build_rate_features"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"pd": pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "task2_reference", "exec"), namespace)
    return namespace


class Module4FeatureTests(unittest.TestCase):
    def test_technical_features_match_task2_reference(self):
        dates = pd.date_range("2021-01-01", periods=40, freq="D")
        rates = pd.DataFrame({"Date": dates, "USD_IDR": np.linspace(14000, 14100, 40)})
        reference = task2_reference_functions()["build_rate_features"](rates)
        actual = build_rate_features_exact(rates)
        for column in TECHNICAL_FEATURES:
            np.testing.assert_allclose(
                actual.loc[20:38, column].to_numpy(dtype=float),
                reference.loc[reference["Date"].astype(str) >= "2021-01-20", column].to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_lag_features_do_not_use_future_prices(self):
        dates = pd.date_range("2021-01-01", periods=30, freq="D")
        base = pd.DataFrame({"Date": dates, "USD_IDR": np.arange(14000, 14030)})
        changed = base.copy()
        changed.loc[20:, "USD_IDR"] += 10000
        first = build_rate_features_exact(base)
        second = build_rate_features_exact(changed)
        for column in TECHNICAL_FEATURES:
            np.testing.assert_allclose(first.loc[:18, column], second.loc[:18, column], equal_nan=True)

    def test_warmup_and_flat_rows_remain_in_calendar(self):
        calendar = load_master_calendar()
        self.assertEqual(len(calendar), 1202)
        self.assertEqual(int(calendar["is_warmup"].sum()), 20)
        self.assertEqual(int(calendar["is_flat_movement"].sum()), 13)
        self.assertEqual(int(calendar["supervised_eligible"].sum()), 1168)

    def test_availability_keeps_empty_news_days(self):
        calendar = pd.DataFrame({"Date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])})
        articles = pd.DataFrame({"Aligned_Date": pd.to_datetime(["2021-01-01", "2021-01-01"])})
        availability = build_availability(calendar, articles)
        self.assertEqual(availability.loc[0, "n_articles"], 2)
        self.assertEqual(availability.loc[1, "n_articles"], 0)
        self.assertEqual(availability.loc[1, "has_news"], 0)
        self.assertEqual(availability.loc[1, "log1p_n_articles"], 0)

    def test_basic_aggregation_and_decay_are_deterministic(self):
        articles = pd.DataFrame(
            {
                "Aligned_Date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02"]),
                "Title": ["a", "b", "c"],
                "vader_title_compound": [0.2, 0.4, -0.2],
                "vader_title_pos": [0.2, 0.3, 0.0], "vader_title_neg": [0.0, 0.0, 0.3],
                "vader_body_compound": [0.1, 0.2, -0.1], "vader_body_pos": [0.1, 0.2, 0.0], "vader_body_neg": [0.0, 0.0, 0.2],
                "lm_title_polarity": [0.1, 0.2, -0.1], "lm_title_subjectivity": [0.2, 0.3, 0.4],
                "lm_body_polarity": [0.1, 0.2, -0.1], "lm_body_subjectivity": [0.2, 0.3, 0.4],
            }
        )
        daily = aggregate_basic_nlp(articles)
        technical = pd.DataFrame({"Date": pd.to_datetime(["2021-01-01", "2021-01-02"]), "volatility_5d": [0.1, 0.2]})
        result = add_task2_daily_derived_features(daily, technical)
        self.assertEqual(daily.loc[daily["Date"] == "2021-01-01", "n_articles"].iloc[0], 2)
        self.assertAlmostEqual(result.loc[1, "title_sentiment_decay3"], 0.05)

    def test_predictor_whitelist_blocks_target_metadata(self):
        assert_predictor_whitelist(["return_t"])
        with self.assertRaises(AssertionError):
            assert_predictor_whitelist(["return_t", "target_direction"])

    def test_baseline_date_coverage_and_fold_compatibility(self):
        calendar = load_master_calendar()
        expected = set(calendar["calendar_index"])
        manifest = pd.read_json(ROOT / "hana-work/data/manifests/module4_feature_manifest.json") if False else None
        for timing in ("strict", "same_date"):
            for dataset in ("a2", "b0", "b1", "b2"):
                path = ROOT / f"hana-work/data/processed/{dataset}_{timing}_predictors.csv"
                self.assertEqual(len(pd.read_csv(path)), 1202)
            metadata = pd.read_csv(ROOT / f"hana-work/data/processed/baseline_{timing}_metadata.csv")
            self.assertEqual(set(metadata["calendar_index"]), expected)
            self.assertEqual(len(metadata), 1202)


if __name__ == "__main__":
    unittest.main()
