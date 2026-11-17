"""Module 5A tests for fold-local TF-IDF/SVD extraction."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from module5a_tfidf_svd import (
    FORBIDDEN_TEXT_COLUMNS,
    N_COMPONENTS,
    build_fold_daily_output,
    fit_transform_fold,
    fold_indices,
    load_daily_inputs,
    load_fold_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


class Module5ATests(unittest.TestCase):
    def test_fold_membership_matches_module3_manifest(self):
        manifest = load_fold_manifest()
        for fold_id in (1, 2, 3):
            train, validation = fold_indices(manifest, fold_id)
            expected = next(fold for fold in manifest["folds"] if fold["fold_id"] == fold_id)
            np.testing.assert_array_equal(train, expected["train_calendar_indices"])
            np.testing.assert_array_equal(validation, expected["validation_calendar_indices"])

    def test_fit_uses_train_and_validation_transform_only(self):
        daily = pd.DataFrame(
            {
                "Date": pd.date_range("2021-01-01", periods=6),
                "calendar_index": range(6),
                "daily_titles": ["alpha beta", "alpha gamma", "beta gamma", "delta", "", "alpha delta"],
            }
        )
        train = np.array([0, 1, 2, 3])
        validation = np.array([4, 5])
        train_matrix, validation_matrix, details = fit_transform_fold(daily, train, validation)
        self.assertEqual(train_matrix.shape, (4, N_COMPONENTS))
        self.assertEqual(validation_matrix.shape, (2, N_COMPONENTS))
        self.assertEqual(details["vectorizer_fit_calendar_indices"], train.tolist())
        self.assertEqual(details["validation_transform_calendar_indices"], validation.tolist())
        self.assertFalse(details["validation_refit"])

    def test_empty_news_transforms_to_zero_vector_and_calendar_is_preserved(self):
        daily = pd.DataFrame(
            {
                "Date": pd.date_range("2021-01-01", periods=5),
                "calendar_index": range(5),
                "daily_titles": ["alpha beta", "", "alpha gamma", "", "beta"],
            }
        )
        availability = pd.DataFrame({"n_articles": [1, 0, 1, 0, 1], "has_news": [1, 0, 1, 0, 1], "log1p_n_articles": np.log1p([1, 0, 1, 0, 1])})
        train = np.array([0, 1, 2])
        validation = np.array([3, 4])
        train_matrix, validation_matrix, _ = fit_transform_fold(daily, train, validation)
        output = build_fold_daily_output(daily, availability, train, validation, train_matrix, validation_matrix)
        self.assertEqual(len(output), 5)
        self.assertTrue(output.loc[output["calendar_index"] == 1, "empty_news_day"].iloc[0])
        self.assertTrue(np.allclose(output.loc[output["calendar_index"] == 1, [f"tfidf_svd_{i}" for i in range(1, 6)]].to_numpy(), 0.0))
        self.assertTrue(np.allclose(output.loc[output["calendar_index"] == 3, [f"tfidf_svd_{i}" for i in range(1, 6)]].to_numpy(), 0.0))

    def test_dimensions_and_reproducibility(self):
        daily = load_daily_inputs("strict").iloc[:80].copy()
        train = np.asarray(list(range(20, 50)))
        validation = np.asarray(list(range(50, 70)))
        first = fit_transform_fold(daily, train, validation)
        second = fit_transform_fold(daily, train, validation)
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        self.assertEqual(first[0].shape[1], first[1].shape[1])
        self.assertEqual(first[0].shape[1], N_COMPONENTS)

    def test_text_input_has_no_target_or_future_columns_and_tracks_are_separate(self):
        strict = load_daily_inputs("strict")
        same_date = load_daily_inputs("same_date")
        self.assertFalse(FORBIDDEN_TEXT_COLUMNS & set(strict.columns))
        self.assertFalse(FORBIDDEN_TEXT_COLUMNS & set(same_date.columns))
        self.assertEqual(set(strict["calendar_index"]), set(same_date["calendar_index"]))
        self.assertFalse(strict["daily_titles"].equals(same_date["daily_titles"]))


if __name__ == "__main__":
    unittest.main()
