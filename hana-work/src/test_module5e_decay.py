import unittest

import numpy as np
import pandas as pd

from module5e_decay import (
    CALENDAR_ROWS, aggregate_articles, aggregate_daily, decay_lambda,
    exponential_weights, fold_indices, load_calendar, load_fold_manifest, trading_age,
)


class Module5EDecayTests(unittest.TestCase):
    def test_weight_formula_and_normalization(self):
        ages = np.array([0, 1, 2])
        weights = exponential_weights(ages, 1)
        expected = np.exp(-np.log(2) * ages); expected /= expected.sum()
        np.testing.assert_allclose(weights, expected); self.assertAlmostEqual(weights.sum(), 1.0); self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(decay_lambda(3), np.log(2) / 3)

    def test_trading_age_uses_calendar_observations(self):
        dates = pd.DatetimeIndex(pd.to_datetime(["2021-01-01", "2021-01-04", "2021-01-05"]))
        self.assertEqual(trading_age(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-05"), dates), 2)
        self.assertEqual(trading_age(pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-04"), dates), 1)
        with self.assertRaises(AssertionError): trading_age(pd.Timestamp("2021-01-06"), pd.Timestamp("2021-01-05"), dates)

    def test_manual_weighted_mean(self):
        dates = pd.DatetimeIndex(pd.to_datetime(["2021-01-01", "2021-01-04", "2021-01-05"]))
        article_dates = pd.Series(pd.to_datetime(["2021-01-01", "2021-01-04"]))
        vectors = np.array([[1.0, 0.0], [0.0, 3.0]])
        result, weights = aggregate_articles(article_dates, pd.Timestamp("2021-01-05"), vectors, dates, 1)
        expected_weights = np.array([0.25, 0.5]) / 0.75
        np.testing.assert_allclose(weights, expected_weights); np.testing.assert_allclose(result, np.average(vectors, axis=0, weights=expected_weights))

    def test_real_folds_and_locked_isolation(self):
        manifest = load_fold_manifest(); calendar = load_calendar(); locked = set(manifest["locked_test"]["calendar_indices"])
        self.assertEqual(len(calendar), CALENDAR_ROWS)
        for fold_id in (1, 2, 3):
            train, validation, locked_indices = fold_indices(manifest, fold_id)
            self.assertTrue(set(train).isdisjoint(validation)); self.assertTrue(set(np.r_[train, validation]).isdisjoint(locked)); self.assertEqual(set(locked_indices), locked)

    def test_empty_and_unavailable_scope(self):
        calendar = pd.DataFrame({"Date": pd.date_range("2021-01-01", periods=4), "calendar_index": range(4)})
        availability = calendar.copy(); availability["n_articles"]=[1,0,1,1]; availability["has_news"]=availability.n_articles.gt(0); availability["log1p_n_articles"]=np.log1p(availability.n_articles)
        metadata = pd.DataFrame({"calendar_index":[0],"Date":pd.to_datetime(["2021-01-01"]),"Aligned_Date":pd.to_datetime(["2021-01-01"]),"successfully_embedded":[True]})
        vectors=np.ones((1,768),dtype=np.float32)
        daily=aggregate_daily(calendar,availability,metadata,vectors,np.array([0]),np.array([1]),1,"finbert")
        self.assertEqual(len(daily),4); self.assertTrue(bool(daily.loc[1,"empty_news_day"])); self.assertEqual(float(daily.loc[1,"finbert_decay_hl1_1"]),0.0); self.assertTrue(bool(daily.loc[2,"unavailable_due_to_scope"])); self.assertTrue(pd.isna(daily.loc[2,"finbert_decay_hl1_1"]))

    def test_model_output_dimensions_and_shared_settings(self):
        manifest_path = __import__('pathlib').Path('hana-work/data/manifests/module5e_decay_manifest.json')
        if manifest_path.exists():
            data=__import__('json').loads(manifest_path.read_text())
            self.assertEqual(data['models'], ['finbert','roberta']); self.assertEqual(data['half_lives_trading_days'], [1,3,5])
            self.assertEqual(len(data['outputs']), 36)


if __name__ == "__main__": unittest.main()
