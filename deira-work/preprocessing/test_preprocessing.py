"""Contract tests for financial text meaning and temporal information boundaries."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import preprocessing_utils as p


def news_fixture():
    return pd.DataFrame([
        {"Date": date, "Title": title, "Source": "Fixture", "URL": f"https://example.test/{i}",
         "Event": "collection metadata is not a relevance label",
         "Text_clean": (body + " ") * 12}
        for i, (date, title, body) in enumerate([
            ("2024-01-05", "US sanctions hit Russian banks", "Russia faces sanctions and trade restrictions affecting banks."),
            ("2024-01-06", "Bank Indonesia leaves rates unchanged", "Indonesia holds rates as inflation and exchange markets stabilize."),
            ("2024-01-07", "Oil prices rise as supply falls", "Oil supply declines and commodity prices rise on global markets."),
            ("2024-01-08", "China tariffs affect US exports", "China announces tariffs on exports from the United States."),
        ])
    ])


def small_calendar():
    return p.build_calendar(pd.DataFrame({"Date": ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"],
                                         "USD_IDR": [15000, 15000, 15100, 14900]}))


def large_calendar():
    dates = pd.bdate_range("2023-01-02", periods=180)
    prices = 15000 + np.arange(180) + 15 * np.sin(np.arange(180))
    return p.build_calendar(pd.DataFrame({"Date": dates, "USD_IDR": prices}))


class TextMeaningTests(unittest.TestCase):
    def test_static_preserves_negation_entities_signed_numbers(self):
        tokens = p.static_tokens("The U.S. won't ignore Bank Indonesia: -2.5%, +25 bps; can't ease.")
        self.assertIn("US", tokens)
        self.assertIn("Indonesia", tokens)
        self.assertEqual(tokens.count("not"), 2)
        self.assertIn("-2.5%", tokens)
        self.assertIn("+25", tokens)
        self.assertIn("indonesia", p.static_tokens("Indonesia", lowercase=True))

    def test_footer_removal_preserves_ordinary_quotes(self):
        paragraph = "The minister said subscribe to diplomatic talks, not escalation. " * 3
        text = paragraph + "\n\n" + paragraph + "\n\nFollow BBC on social media\n\nRelated future story"
        body, info = p.clean_body(text)
        self.assertEqual(body, paragraph.strip())
        self.assertEqual(info["removed_paragraphs"], 3)
        self.assertEqual(info["footer_cut_reason"], "explicit_footer_boundary")

    def test_geopolitical_triage_does_not_default_to_corpus_filter(self):
        d = p.apply_reviews(p.prepare_news(news_fixture()))
        audit = p.choose_candidates(d)
        self.assertTrue(audit.selected_for_alignment.all())
        self.assertIn("exclude", audit.geo_status.tolist())
        screened = p.choose_candidates(d, geopolitical_policy="rule_screened")
        self.assertLess(screened.selected_for_alignment.sum(), len(d))
        self.assertEqual(p.choose_candidates(d, geopolitical_policy="human_approved").selected_for_alignment.sum(), 0)

    def test_future_date_reference_is_review_not_confirmed_leak(self):
        self.assertEqual(p.later_date_mentions("Tariffs start December 1, 2025", pd.Timestamp("2025-01-01")), ["2025-12-01"])
        prepared = p.prepare_news(news_fixture())
        self.assertFalse(prepared.known_temporal_contamination.any())

    def test_manual_include_cannot_override_known_future_content(self):
        prepared = p.prepare_news(news_fixture(), known_future_ids=[0])
        prepared.loc[0, ["geo_decision", "review_note"]] = ["include", "Synthetic review"]
        chosen = p.choose_candidates(p.apply_reviews(prepared))
        self.assertFalse(chosen.loc[0, "selected_for_alignment"])
        self.assertEqual(chosen.loc[0, "selection_reason"], "known_later_content_quarantine")

    def test_duplicate_keeps_earliest_observed_record(self):
        raw = news_fixture().iloc[[0, 0]].copy().reset_index(drop=True)
        raw.loc[0, "Date"] = "2024-01-09"
        raw.loc[1, "Date"] = "2024-01-05"
        d = p.prepare_news(raw)
        self.assertEqual(d.loc[0, "duplicate_of"], d.loc[1, "article_id"])
        self.assertEqual(d.loc[1, "duplicate_of"], "")


class CalendarTests(unittest.TestCase):
    def test_flat_fixing_remains_before_next_return_calculation(self):
        d = small_calendar()
        self.assertEqual(len(d), 4)
        self.assertEqual(d.loc[0, "target_date"], pd.Timestamp("2024-01-08"))
        self.assertEqual(d.loc[0, "realized_target_gap_days"], 3)
        self.assertEqual(d.loc[0, "target_status"], "FLAT")
        self.assertTrue(pd.isna(d.loc[0, "y"]))
        self.assertEqual(d.loc[1, "y"], 1)
        self.assertEqual(d.loc[2, "y"], 0)
        self.assertAlmostEqual(d.loc[1, "log_return_next"], np.log(15100 / 15000))
        self.assertTrue(pd.isna(d.iloc[-1].y))

    def test_market_features_are_invariant_to_future_prices(self):
        before = large_calendar()
        raw = before[["Date", "USD_IDR"]].copy()
        raw.loc[100:, "USD_IDR"] *= 2
        after = p.build_calendar(raw)
        pd.testing.assert_frame_equal(before.loc[:99, p.MARKET_FEATURES], after.loc[:99, p.MARKET_FEATURES])
        self.assertNotEqual(before.loc[99, "log_return_next"], after.loc[99, "log_return_next"])

    def test_bad_calendar_is_rejected(self):
        for raw in [pd.DataFrame({"Date": ["2024-01-08"] * 2, "USD_IDR": [1, 2]}),
                    pd.DataFrame({"Date": ["2024-01-06"], "USD_IDR": [1]}),
                    pd.DataFrame({"Date": ["2024-01-08"], "USD_IDR": [0]})]:
            with self.subTest(raw=raw.to_dict()):
                with self.assertRaises(ValueError):
                    p.build_calendar(raw)

    def test_fold_labels_end_before_validation_and_test(self):
        daily = large_calendar().assign(n_berita=0, has_news=0)
        partitions, folds, _, final = p.temporal_manifests(daily)
        for fold in folds.fold.unique():
            train = folds.loc[(folds.fold == fold) & folds.role.eq("train")]
            val = folds.loc[(folds.fold == fold) & folds.role.eq("validation")]
            self.assertLess(train.target_date.max(), val.Date.min())
        self.assertLess(final.loc[final.role.eq("train"), "target_date"].max(),
                        final.loc[final.role.eq("test"), "Date"].min())
        self.assertEqual(partitions.partition.eq("boundary_purge").sum(), 1)
        self.assertFalse(set(p.MARKET_FEATURES) & set(p.TARGET_COLUMNS))


class AvailabilityTests(unittest.TestCase):
    def test_date_delay_weekend_and_same_date_behavior(self):
        d = p.choose_candidates(p.apply_reviews(p.prepare_news(news_fixture())))
        aligned = p.align_articles(d, small_calendar())
        self.assertEqual(aligned.anchor_date.tolist(), [pd.Timestamp("2024-01-08")] * 3 + [pd.Timestamp("2024-01-09")])
        daily, windows = p.build_daily_tables(small_calendar(), aligned)
        self.assertEqual(daily.n_berita.tolist(), [0, 3, 1, 0])
        self.assertEqual(windows.iloc[0].article_ids, [])
        self.assertEqual(windows.iloc[0].texts_transformer, [])
        late = d.copy()
        late["source_date"] = pd.Timestamp("2024-01-10")
        self.assertTrue(p.align_articles(late, small_calendar()).anchor_date.isna().all())

    def test_verified_cutoff_inclusive_after_cutoff_and_unverified(self):
        d = p.prepare_news(news_fixture())
        d["available_at"] = ["2024-01-08T08:15:00Z", "2024-01-08T08:15:01Z", "2024-01-09T08:15:00Z", ""]
        d["timestamp_verified"] = [True, True, False, False]
        d["body_asof_verified"] = [True, True, True, False]
        d["evidence"] = ["Synthetic evidence"] * 4
        d = p.choose_candidates(p.apply_reviews(d))
        calendar_frame = small_calendar()
        rows = pd.DataFrame({"Date": calendar_frame.Date.dt.strftime("%Y-%m-%d"),
                             "published_at_wib": calendar_frame.Date.dt.strftime("%Y-%m-%d") + "T15:15:00+07:00",
                             "decision_at_wib": calendar_frame.Date.dt.strftime("%Y-%m-%d") + "T15:15:00+07:00",
                             "verified": True, "source": "Synthetic schedule, not historical BI evidence"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schedule.csv"
            rows.to_csv(path, index=False)
            schedule = p.read_verified_schedule(path, calendar_frame)
            aligned = p.align_articles(d, calendar_frame, mode="strict_forecast", schedule=schedule)
            self.assertEqual(aligned.loc[0, "anchor_date"], pd.Timestamp("2024-01-08"))
            self.assertEqual(aligned.loc[1, "anchor_date"], pd.Timestamp("2024-01-09"))
            self.assertTrue(aligned.loc[[2, 3], "anchor_date"].isna().all())
            rows.loc[0, "decision_at_wib"] = "2024-01-05T15:00:00+07:00"
            rows.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                p.read_verified_schedule(path, calendar_frame)

    def test_naive_times_and_unsupported_verification_rejected(self):
        with self.assertRaises(ValueError):
            p.aware_time("2024-01-08 15:15")
        self.assertEqual(p.aware_time("2024-01-08 08:15", "UTC").hour, 15)
        d = p.prepare_news(news_fixture())
        d.loc[0, "timestamp_verified"] = "yes"
        with self.assertRaises(ValueError):
            p.apply_reviews(d)

    def test_unmatched_review_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            pd.DataFrame({"article_id": ["wrong-snapshot-id"]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                p.apply_reviews(p.prepare_news(news_fixture()), path)

    def test_jsonl_preserves_empty_lists_and_nulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "windows.jsonl"
            p.write_jsonl(pd.DataFrame({"Date": [pd.Timestamp("2024-01-08")],
                                       "article_ids": [[]], "label": [pd.NA]}), path)
            row = json.loads(path.read_text())
            self.assertEqual(row["article_ids"], [])
            self.assertIsNone(row["label"])


if __name__ == "__main__":
    unittest.main()
