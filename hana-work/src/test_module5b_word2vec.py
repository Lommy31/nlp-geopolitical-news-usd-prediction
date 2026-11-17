import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from module5b_word2vec import (
    CALENDAR_ROWS,
    VECTOR_SIZE,
    aggregate_daily,
    article_vectors,
    fold_indices,
    load_calendar,
    load_aligned_articles,
    load_fold_manifest,
    tokenize_title,
    train_fold_model,
)


class Module5BTests(unittest.TestCase):
    def test_fold_membership_matches_module3_manifest_and_excludes_test(self):
        manifest = load_fold_manifest()
        locked = set(manifest["locked_test"]["calendar_indices"])
        self.assertEqual(len(fold_indices(manifest, 1)[0]), 232)
        self.assertEqual(len(fold_indices(manifest, 2)[0]), 465)
        self.assertEqual(len(fold_indices(manifest, 3)[0]), 698)
        for fold_id in (1, 2, 3):
            train, validation = fold_indices(manifest, fold_id)
            self.assertTrue(set(train).isdisjoint(locked))
            self.assertTrue(set(validation).isdisjoint(locked))

    def _synthetic_articles(self):
        rows = [
            {"article_id": "a1", "Aligned_Date": "2021-01-01", "calendar_index": 0, "Title_clean": "alpha beta beta"},
            {"article_id": "a2", "Aligned_Date": "2021-01-02", "calendar_index": 1, "Title_clean": "alpha gamma"},
            {"article_id": "v1", "Aligned_Date": "2021-01-03", "calendar_index": 2, "Title_clean": "validationonly"},
            {"article_id": "o1", "Aligned_Date": "2021-01-04", "calendar_index": 3, "Title_clean": "unseenword"},
        ]
        frame = pd.DataFrame(rows)
        frame["Aligned_Date"] = pd.to_datetime(frame["Aligned_Date"])
        frame["tokens"] = frame["Title_clean"].map(tokenize_title)
        return frame

    def test_training_only_vocabulary_and_validation_does_not_influence_model(self):
        articles = self._synthetic_articles()
        model = train_fold_model(articles, np.array([0, 1]))
        self.assertIn("alpha", model.wv.key_to_index)
        self.assertNotIn("validationonly", model.wv.key_to_index)
        self.assertNotIn("unseenword", model.wv.key_to_index)
        self.assertEqual(model.sg, 1)

    def test_models_are_independent_and_fixed_seed_reproducible(self):
        articles = self._synthetic_articles()
        first = train_fold_model(articles, np.array([0, 1]))
        second = train_fold_model(articles, np.array([0, 1]))
        different_fold = train_fold_model(articles, np.array([0, 2]))
        self.assertIsNot(first, second)
        np.testing.assert_array_equal(first.wv["alpha"], second.wv["alpha"])
        self.assertNotIn("alpha", different_fold.wv.key_to_index)

    def test_locked_test_indices_are_never_in_article_vector_scope(self):
        articles = self._synthetic_articles()
        model = train_fold_model(articles, np.array([0, 1]))
        vectors = article_vectors(articles, model, np.array([0, 1]), np.array([2]))
        self.assertTrue(set(vectors["calendar_index"]).isdisjoint({3, 4}))

    def test_article_vectors_dimensions_and_oov_only_handling(self):
        articles = self._synthetic_articles()
        model = train_fold_model(articles, np.array([0, 1]))
        vectors = article_vectors(articles, model, np.array([0, 1]), np.array([2, 3]))
        self.assertEqual(vectors.shape[0], 4)
        self.assertTrue(all(vector.shape == (VECTOR_SIZE,) for vector in vectors["vector"]))
        self.assertTrue(bool(vectors.loc[vectors.article_id == "v1", "successfully_embedded"].iloc[0]) is False)
        self.assertTrue(bool(vectors.loc[vectors.article_id == "o1", "successfully_embedded"].iloc[0]) is False)
        np.testing.assert_array_equal(vectors.loc[vectors.article_id == "v1", "vector"].iloc[0], np.zeros(VECTOR_SIZE))

    def test_daily_uniform_mean_empty_news_oov_only_and_calendar_coverage(self):
        calendar = pd.DataFrame({"Date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]), "calendar_index": range(5)})
        availability = pd.DataFrame({"Date": calendar.Date, "calendar_index": range(5), "n_articles": [1, 1, 1, 0, 0], "has_news": [True, True, True, False, False], "log1p_n_articles": np.log1p([1, 1, 1, 0, 0])})
        vectors = pd.DataFrame({"article_id": ["a1", "a2", "v1"], "Aligned_Date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]), "calendar_index": [0, 1, 2], "fold_role": ["train", "train", "validation"], "token_count": [2, 2, 1], "in_vocab_token_count": [2, 2, 0], "oov_token_count": [0, 0, 1], "successfully_embedded": [True, True, False], "vector": [np.ones(VECTOR_SIZE), np.full(VECTOR_SIZE, 3.0), np.zeros(VECTOR_SIZE)]})
        daily = aggregate_daily(calendar, availability, vectors, np.array([0, 1]), np.array([2]))
        self.assertEqual(len(daily), 5)
        self.assertAlmostEqual(float(daily.loc[daily.calendar_index == 0, "word2vec_1"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(daily.loc[daily.calendar_index == 1, "word2vec_1"].iloc[0]), 3.0)
        self.assertTrue(bool(daily.loc[daily.calendar_index == 2, "oov_only_news_day"].iloc[0]))
        self.assertFalse(bool(daily.loc[daily.calendar_index == 2, "empty_news_day"].iloc[0]))
        self.assertTrue(bool(daily.loc[daily.calendar_index == 3, "empty_news_day"].iloc[0]))
        self.assertEqual(float(daily.loc[daily.calendar_index == 2, "word2vec_1"].iloc[0]), 0.0)

    def test_real_calendar_and_timing_tracks_are_separate(self):
        calendar = load_calendar()
        self.assertEqual(len(calendar), CALENDAR_ROWS)
        strict = Path("hana-work/data/intermediate/articles_aligned_strict.csv")
        same = Path("hana-work/data/intermediate/articles_aligned_same_date.csv")
        self.assertTrue(strict.exists())
        self.assertTrue(same.exists())
        strict_sample = pd.read_csv(strict, usecols=["article_id", "Aligned_Date"], nrows=1)
        same_sample = pd.read_csv(same, usecols=["article_id", "Aligned_Date"], nrows=1)
        self.assertEqual(strict_sample.article_id.iloc[0], same_sample.article_id.iloc[0])
        self.assertNotEqual(strict_sample.Aligned_Date.iloc[0], same_sample.Aligned_Date.iloc[0])

    def test_real_embedding_input_is_title_only_and_has_no_target_future_columns(self):
        calendar = load_calendar()
        for mode in ("strict", "same_date"):
            articles = load_aligned_articles(mode, calendar)
            self.assertEqual(set(articles.columns), {"article_id", "Title_clean", "Aligned_Date", "alignment_mode", "alignment_failed", "out_of_calendar_or_invalid_date_flag", "calendar_index", "tokens"})
            self.assertNotIn("Text_clean", articles.columns)
            self.assertTrue(FORBIDDEN_REAL_COLUMNS.isdisjoint(set(articles.columns)))


FORBIDDEN_REAL_COLUMNS = {"target_direction", "target_date", "next_Date", "next_return", "next_USD_IDR", "next_jisdor", "log_return_next", "y", "split", "fold", "fold_id", "module3_split"}


if __name__ == "__main__":
    unittest.main()
