import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from module5d_roberta import (
    CALENDAR_ROWS, EMBEDDING_DIMENSION, MODEL_REVISION, aggregate_daily,
    encode_article_embeddings, fold_indices, fold_scope, inspect_headline_lengths,
    load_aligned_articles, load_calendar, load_fold_manifest, load_frozen_model,
)


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [list(range(index + 1)) for index, _ in enumerate(texts)]}


class Module5DRobertaTests(unittest.TestCase):
    def test_pinned_configuration_and_frozen_eval_model(self):
        tokenizer, model, _ = load_frozen_model()
        self.assertEqual(MODEL_REVISION, "e2da8e2f811d1448a5b465c236feacd80ffbac7b")
        self.assertEqual((model.config.hidden_size, model.config.num_hidden_layers, model.config.num_attention_heads, model.config.vocab_size, model.config.max_position_embeddings), (768, 12, 12, 50265, 514))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.training)
        self.assertEqual(tokenizer.cls_token_id, 0)

    def test_synthetic_shape_finite_nonzero_and_reproducible(self):
        tokenizer, model, device = load_frozen_model()
        frame = pd.DataFrame({"article_id": ["a", "b"], "calendar_index": [0, 1], "Title_clean": ["Central bank held rates steady.", "Markets rallied strongly."], "headline_empty": [False, False], "fold_role": ["train", "validation"]})
        frame = inspect_headline_lengths(frame, tokenizer, model.config.max_position_embeddings)
        first, vectors1 = encode_article_embeddings(frame, tokenizer, model, device, batch_size=2)
        second, vectors2 = encode_article_embeddings(frame, tokenizer, model, device, batch_size=2)
        self.assertEqual(vectors1.shape, (2, EMBEDDING_DIMENSION))
        self.assertTrue(np.isfinite(vectors1).all() and np.any(vectors1 != 0))
        np.testing.assert_array_equal(vectors1, vectors2)
        self.assertTrue(first["successfully_embedded"].all() and second["successfully_embedded"].all())

    def test_manifest_fold_membership_and_locked_exclusion(self):
        manifest = load_fold_manifest(); calendar = load_calendar(); locked = set(manifest["locked_test"]["calendar_indices"])
        for fold_id in (1, 2, 3):
            train, validation, locked_array = fold_indices(manifest, fold_id)
            self.assertTrue(set(train).isdisjoint(validation) and set(train).isdisjoint(locked) and set(validation).isdisjoint(locked))
            self.assertTrue(np.array_equal(calendar.loc[calendar.calendar_index.isin(train), "calendar_index"].to_numpy(), train))
            self.assertTrue(np.array_equal(locked_array, np.asarray(manifest["locked_test"]["calendar_indices"])))

    def test_article_scope_tracks_and_provenance(self):
        calendar = load_calendar(); manifest = load_fold_manifest(); train, validation, locked = fold_indices(manifest, 1)
        strict = load_aligned_articles("strict", calendar); same = load_aligned_articles("same_date", calendar)
        self.assertTrue((strict.alignment_mode == "strictly_later").all() and (same.alignment_mode == "same_date").all())
        self.assertNotEqual(strict.loc[0, "Aligned_Date"], same.loc[0, "Aligned_Date"])
        for articles in (strict, same):
            scoped = fold_scope(articles, train, validation, locked)
            self.assertTrue(set(scoped.calendar_index).isdisjoint(locked))
            self.assertIn("provenance_status", scoped)
            self.assertNotIn("target_direction", scoped)

    def test_daily_mean_empty_and_out_of_scope(self):
        calendar = pd.DataFrame({"Date": pd.date_range("2021-01-01", periods=5), "calendar_index": range(5)})
        availability = calendar.copy(); availability["n_articles"] = [1, 1, 1, 0, 1]; availability["has_news"] = availability.n_articles.gt(0); availability["log1p_n_articles"] = np.log1p(availability.n_articles)
        articles = pd.DataFrame({"calendar_index": [0, 1], "headline_status": ["candidate", "candidate"], "successfully_embedded": [True, True]})
        vectors = np.vstack([np.ones(EMBEDDING_DIMENSION), np.full(EMBEDDING_DIMENSION, 3.0)])
        daily = aggregate_daily(calendar, availability, articles, vectors, np.array([0, 1]), np.array([2]))
        self.assertEqual(len(daily), 5); self.assertEqual(float(daily.loc[0, "roberta_1"]), 1.0); self.assertEqual(float(daily.loc[1, "roberta_1"]), 3.0)
        self.assertTrue(bool(daily.loc[3, "empty_news_day"])); self.assertFalse(bool(daily.loc[3, "feature_available"])); self.assertEqual(float(daily.loc[3, "roberta_1"]), 0.0)
        self.assertTrue(bool(daily.loc[4, "unavailable_due_to_scope"])); self.assertTrue(pd.isna(daily.loc[4, "roberta_1"]))

    def test_overlength_is_explicit(self):
        frame = pd.DataFrame({"Title_clean": ["short", "long"], "headline_empty": [False, False]})
        inspected = inspect_headline_lengths(frame, FakeTokenizer(), 1)
        self.assertEqual(inspected.loc[0, "headline_status"], "candidate"); self.assertEqual(inspected.loc[1, "headline_status"], "overlength")

    def test_real_calendar_size(self):
        self.assertEqual(len(load_calendar()), CALENDAR_ROWS)


if __name__ == "__main__":
    unittest.main()
