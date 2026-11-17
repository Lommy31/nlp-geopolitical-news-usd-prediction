"""Synthetic tests for Module 3 chronological validation safeguards."""

import unittest

import pandas as pd

from module3_validation import (
    assert_predictor_columns,
    assert_preprocessing_fit_scope,
    assert_target_date_boundary,
    construct_splits,
)


def synthetic_calendar(rows=12):
    dates = pd.date_range("2021-01-01", periods=rows, freq="D")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "next_Date": dates.shift(1) if hasattr(dates, "shift") else list(dates[1:]) + [pd.NaT],
            "next_USD_IDR": [10001 + i for i in range(rows - 1)] + [None],
            "next_return": [0.001] * (rows - 1) + [None],
            "target_available": [True] * (rows - 1) + [False],
            "is_flat_movement": [False] * rows,
            "target_direction": [True, False] * 5 + [True, None],
            "calendar_index": list(range(rows)),
            "is_warmup": [True, True] + [False] * (rows - 2),
            "supervised_eligible": [False, False] + [True] * (rows - 3) + [False],
        }
    )
    # pandas DatetimeIndex has no shift method in all supported versions.
    frame["next_Date"] = list(dates[1:]) + [pd.NaT]
    return frame


class Module3ValidationTests(unittest.TestCase):
    def test_expanding_folds_and_reproducibility(self):
        calendar = synthetic_calendar()
        first = construct_splits(calendar)
        second = construct_splits(calendar)
        self.assertEqual(first["folds"], second["folds"])
        self.assertEqual(len(first["locked_test"]["calendar_indices"]), 3)
        self.assertLess(first["folds"][0]["train_row_count"], first["folds"][1]["train_row_count"])
        self.assertLess(first["folds"][1]["train_row_count"], first["folds"][2]["train_row_count"])

    def test_boundary_purge_is_outside_train_and_test(self):
        result = construct_splits(synthetic_calendar())
        self.assertEqual(result["locked_test"]["calendar_indices"][0], 9)
        self.assertEqual(result["configuration"]["test_start_calendar_index"], 9)
        for fold in result["folds"]:
            self.assertTrue(set(fold["purged_calendar_indices"]).isdisjoint(fold["train_calendar_indices"]))
            self.assertTrue(set(fold["purged_calendar_indices"]).isdisjoint(fold["validation_calendar_indices"]))

    def test_target_date_boundary_is_rejected(self):
        calendar = synthetic_calendar()
        with self.assertRaises(AssertionError):
            assert_target_date_boundary(calendar, [3], [4])

    def test_flat_and_warmup_rows_are_not_eligible(self):
        calendar = synthetic_calendar()
        result = construct_splits(calendar)
        for fold in result["folds"]:
            all_members = fold["train_calendar_indices"] + fold["validation_calendar_indices"]
            self.assertTrue(calendar.loc[all_members, "supervised_eligible"].all())

    def test_forbidden_predictors_and_preprocessing_scope(self):
        assert_predictor_columns(["return_lag1", "volatility_5d"])
        with self.assertRaises(AssertionError):
            assert_predictor_columns(["return_lag1", "target_direction"])
        assert_preprocessing_fit_scope([1, 2], [1, 2, 3])
        with self.assertRaises(AssertionError):
            assert_preprocessing_fit_scope([1, 4], [1, 2, 3])

    def test_locked_test_isolated_from_development(self):
        result = construct_splits(synthetic_calendar())
        test = set(result["locked_test"]["calendar_indices"])
        development = set(result["development"]["calendar_indices"])
        purge = set(result["configuration"]["test_start_calendar_index"] - 1 for _ in [0])
        self.assertTrue(test.isdisjoint(development))
        self.assertTrue(test.isdisjoint(purge))


if __name__ == "__main__":
    unittest.main()
