"""Focused Module 2 integrity tests using small synthetic inputs."""

import unittest

import pandas as pd

from module2_data_audit import (
    align_news_to_fixings,
    build_master_calendar,
    clean_and_deduplicate_articles,
)


class Module2IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.rates = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]),
                "USD_IDR": [14000, 14010, 14010],
            }
        )
        self.raw = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2021-01-04", "2021-01-04", "2021-01-05", "2021-01-10"]),
                "Title": ["Same title", "Same title", "Different", "Out of range"],
                "Source": ["cnbc.com"] * 4,
                "URL": ["u1", "u1", "u2", "u3"],
                "Text": ["Body one", "Body one", "Body two", "Body three"],
                "source_row_number": [1, 2, 3, 4],
                "publication_timestamp_present": [False] * 4,
                "publication_timestamp_verified": [False] * 4,
                "Date_raw": ["2021-01-04"] * 4,
                "date_parse_failed": [False] * 4,
            }
        )

    def test_exact_duplicate_removed_and_provenance_retained(self):
        articles, summary = clean_and_deduplicate_articles(self.raw)
        self.assertEqual(len(articles), 3)
        self.assertEqual(summary["exact_duplicate_rows_removed"], 1)
        self.assertEqual(articles.iloc[0]["source_row_number"], 1)

    def test_strict_alignment_is_strictly_later(self):
        articles, _ = clean_and_deduplicate_articles(self.raw)
        aligned = align_news_to_fixings(articles, self.rates, "strictly_later")
        valid = aligned[aligned["Aligned_Date"].notna()]
        self.assertTrue((valid["Aligned_Date"] > valid["Date"]).all())
        self.assertEqual(int(aligned["alignment_failed"].sum()), 1)

    def test_same_date_alignment_never_precedes_article_date(self):
        articles, _ = clean_and_deduplicate_articles(self.raw)
        aligned = align_news_to_fixings(articles, self.rates, "same_date")
        valid = aligned[aligned["Aligned_Date"].notna()]
        self.assertTrue((valid["Aligned_Date"] >= valid["Date"]).all())

    def test_calendar_retains_flat_and_empty_news_rows(self):
        articles, _ = clean_and_deduplicate_articles(self.raw)
        aligned = align_news_to_fixings(articles, self.rates, "strictly_later")
        calendar = build_master_calendar(self.rates, aligned)
        self.assertEqual(len(calendar), len(self.rates))
        self.assertEqual(int(calendar["is_flat_movement"].sum()), 1)
        self.assertTrue((~calendar["has_news"]).any())
        self.assertTrue(calendar["Date"].is_unique)


if __name__ == "__main__":
    unittest.main()
