import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from module5c_finbert import (
    CALENDAR_ROWS,
    EMBEDDING_DIMENSION,
    MODEL_REVISION,
    aggregate_daily,
    encode_article_embeddings,
    fold_indices,
    fold_scope,
    inspect_headline_lengths,
    load_aligned_articles,
    load_calendar,
    load_fold_manifest,
    load_frozen_model,
)


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [list(range(index + 1)) for index, _ in enumerate(texts)]}


class Module5CFinBERTTests(unittest.TestCase):
    def test_pinned_model_configuration_and_frozen_parameters(self):
        tokenizer, model, _ = load_frozen_model()
        self.assertEqual(MODEL_REVISION, "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031")
        self.assertEqual(model.config.hidden_size, EMBEDDING_DIMENSION)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(tokenizer.cls_token)

    def test_synthetic_cls_shape_finite_nonzero_and_reproducible(self):
        tokenizer, model, device = load_frozen_model()
        frame = pd.DataFrame({
            "article_id": ["synthetic-1", "synthetic-2"],
            "calendar_index": [0, 1],
            "Title_clean": ["Central bank held rates steady.", "Markets rallied on strong data."],
            "headline_empty": [False, False],
            "fold_role": ["train", "validation"],
        })
        frame = inspect_headline_lengths(frame, tokenizer, model.config.max_position_embeddings)
        first, first_vectors = encode_article_embeddings(frame, tokenizer, model, device, batch_size=2)
        second, second_vectors = encode_article_embeddings(frame, tokenizer, model, device, batch_size=2)
        self.assertEqual(first_vectors.shape, (2, EMBEDDING_DIMENSION))
        self.assertTrue(np.isfinite(first_vectors).all())
        self.assertTrue(np.any(first_vectors != 0))
        np.testing.assert_array_equal(first_vectors, second_vectors)
        self.assertTrue(first["successfully_embedded"].all())

    def test_exact_fold_membership_and_locked_test_exclusion(self):
        manifest = load_fold_manifest()
        locked = set(manifest["locked_test"]["calendar_indices"])
        calendar = load_calendar()
        for fold_id in (1, 2, 3):
            train, validation = fold_indices(manifest, fold_id)
            self.assertTrue(set(train).isdisjoint(set(validation)))
            self.assertTrue(set(train).isdisjoint(locked))
            self.assertTrue(set(validation).isdisjoint(locked))
            self.assertEqual(len(train), {1: 232, 2: 465, 3: 698}[fold_id])
            self.assertEqual(len(validation), 233)
            self.assertTrue(np.array_equal(calendar.loc[calendar.calendar_index.isin(train), "calendar_index"].to_numpy(), train))

    def test_article_scope_preserves_provenance_and_excludes_forbidden_fields(self):
        calendar = load_calendar()
        manifest = load_fold_manifest()
        train, validation = fold_indices(manifest, 1)
        for mode in ("strict", "same_date"):
            articles = load_aligned_articles(mode, calendar)
            self.assertNotIn("Text_clean", articles.columns)
            self.assertNotIn("target_direction", articles.columns)
            scoped = fold_scope(articles, train, validation)
            self.assertTrue({"article_id", "source_row_number", "provenance_status", "alignment_mode"} <= set(scoped.columns))
            self.assertTrue(set(scoped.calendar_index).isdisjoint(set(manifest["locked_test"]["calendar_indices"])))

    def test_strict_and_same_date_tracks_are_separate(self):
        calendar = load_calendar()
        strict = load_aligned_articles("strict", calendar)
        same = load_aligned_articles("same_date", calendar)
        self.assertTrue((strict["alignment_mode"] == "strictly_later").all())
        self.assertTrue((same["alignment_mode"] == "same_date").all())
        self.assertNotEqual(strict.loc[0, "Aligned_Date"], same.loc[0, "Aligned_Date"])

    def test_overlength_detection_does_not_silently_truncate(self):
        frame = pd.DataFrame({
            "Title_clean": ["short", "long"],
            "headline_empty": [False, False],
        })
        inspected = inspect_headline_lengths(frame, FakeTokenizer(), maximum_length=1)
        self.assertEqual(inspected.loc[0, "headline_status"], "candidate")
        self.assertEqual(inspected.loc[1, "headline_status"], "overlength")
        self.assertTrue(bool(inspected.loc[1, "overlength_headline"]))

    def test_daily_mean_calendar_coverage_and_empty_unavailable_distinction(self):
        calendar = pd.DataFrame({"Date": pd.date_range("2021-01-01", periods=5), "calendar_index": range(5)})
        availability = calendar.copy()
        availability["n_articles"] = [1, 1, 1, 0, 1]
        availability["has_news"] = availability["n_articles"].gt(0)
        availability["log1p_n_articles"] = np.log1p(availability["n_articles"])
        articles = pd.DataFrame({
            "calendar_index": [0, 1, 2],
            "headline_status": ["candidate", "candidate", "empty"],
            "successfully_embedded": [True, True, False],
        })
        vectors = np.vstack([np.ones(EMBEDDING_DIMENSION), np.full(EMBEDDING_DIMENSION, 3.0), np.zeros(EMBEDDING_DIMENSION)])
        daily = aggregate_daily(calendar, availability, articles, vectors, np.array([0, 1]), np.array([2]))
        self.assertEqual(len(daily), 5)
        self.assertAlmostEqual(float(daily.loc[daily.calendar_index == 0, "finbert_1"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(daily.loc[daily.calendar_index == 1, "finbert_1"].iloc[0]), 3.0)
        self.assertTrue(bool(daily.loc[daily.calendar_index == 3, "empty_news_day"].iloc[0]))
        self.assertTrue(bool(daily.loc[daily.calendar_index == 3, "feature_available"].iloc[0]) is False)
        self.assertTrue(bool(daily.loc[daily.calendar_index == 4, "unavailable_due_to_scope"].iloc[0]))
        self.assertTrue(pd.isna(daily.loc[daily.calendar_index == 4, "finbert_1"]).iloc[0])
        self.assertEqual(float(daily.loc[daily.calendar_index == 3, "finbert_1"].iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
