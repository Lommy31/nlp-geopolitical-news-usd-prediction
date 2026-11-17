"""Task 3 Module 2: canonical data integrity and temporal audit.

This module deliberately stops before model-ready feature construction.  It
loads the verified Task 2 inputs, preserves article provenance, creates two
date-only news alignments, and builds an auditable master calendar containing
all observed JISDOR rows, including flat, warm-up, and empty-news rows.

Publication timestamps are not fabricated.  The source has only a calendar
date, so same-date alignment is explicitly marked as intraday-uncertain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FX_PATH = ROOT / "data/processed/usd_idr_jisdor_clean.csv"
NEWS_PATH = ROOT / "data/raw/news_cnbc.csv"
HANA = ROOT / "hana-work"
INTERMEDIATE = HANA / "data/intermediate"
MANIFESTS = HANA / "data/manifests"

REQUIRED_FX_COLUMNS = {"Date", "USD_IDR"}
REQUIRED_NEWS_COLUMNS = {"Date", "Title", "Source", "URL", "Text"}
LOOKBACK_WARMUP_ROWS = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"<[^>]*>", " ", str(value))
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_title(value: Any) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"^[^|]{1,20}\|\s*", "", text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def _normalize_url(value: Any) -> str:
    return _clean_text(value).lower().rstrip("/")


def load_jisdor(path: Path = FX_PATH) -> pd.DataFrame:
    """Load and validate the canonical Task 2 JISDOR calendar."""
    frame = pd.read_csv(path)
    missing = REQUIRED_FX_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"JISDOR input is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["source_row_number"] = range(1, len(frame) + 1)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    if frame["Date"].duplicated().any():
        raise AssertionError("JISDOR calendar contains duplicate dates")
    if not frame["Date"].is_monotonic_increasing:
        frame = frame.sort_values("Date", kind="stable").reset_index(drop=True)
    if frame["USD_IDR"].isna().any():
        raise AssertionError("JISDOR calendar contains missing USD_IDR values")
    return frame


def load_cnbc(path: Path = NEWS_PATH) -> pd.DataFrame:
    """Load the raw CNBC corpus without inventing timestamps or versions."""
    frame = pd.read_csv(path)
    missing = REQUIRED_NEWS_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"CNBC input is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["source_row_number"] = range(1, len(frame) + 1)
    frame["publication_timestamp_present"] = False
    frame["publication_timestamp_verified"] = False
    frame["Date_raw"] = frame["Date"]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame["date_parse_failed"] = frame["Date"].isna()
    return frame


def clean_and_deduplicate_articles(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean text and safely deduplicate only identical article content.

    Without publication timestamps, selecting one record from conflicting
    same-title versions would create unverifiable chronology.  Exact content
    duplicates are safe to collapse; same-title/different-content records are
    retained and explicitly flagged.
    """
    frame = raw.copy()
    frame["Title_clean"] = frame["Title"].map(_clean_text)
    frame["Text_clean"] = frame["Text"].map(_clean_text)
    frame["title_key"] = frame["Title"].map(_normalize_title)
    frame["url_key"] = frame["URL"].map(_normalize_url)
    frame["word_count"] = frame["Text_clean"].str.split().str.len()
    frame["content_key"] = (
        frame["title_key"].fillna("") + "\n" + frame["Text_clean"].str.lower()
    ).map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    frame["exact_content_duplicate_group"] = frame["content_key"].duplicated(keep=False)
    frame["same_title_conflict_group"] = (
        frame.groupby("title_key", dropna=False)["content_key"].transform("nunique") > 1
    ) & frame["title_key"].ne("")
    frame["duplicate_group_size"] = frame.groupby("content_key", dropna=False)["content_key"].transform("size")
    frame["provenance_status"] = "original_record"
    frame.loc[frame["same_title_conflict_group"], "provenance_status"] = "ambiguous_same_title_version_retained"
    frame.loc[frame["exact_content_duplicate_group"], "provenance_status"] = "exact_duplicate_candidate"

    keep = ~frame["content_key"].duplicated(keep="first")
    frame["retained_after_exact_dedup"] = keep
    frame.loc[~keep, "provenance_status"] = "exact_duplicate_removed"
    frame = frame.loc[keep].copy()
    frame["article_id"] = frame.apply(
        lambda row: hashlib.sha256(
            f"{row['source_row_number']}|{row['content_key']}".encode("utf-8")
        ).hexdigest()[:20], axis=1
    )
    frame["timing_uncertainty_flag"] = True
    frame["confirmed_contamination_flag"] = False
    frame["heuristic_contamination_flag"] = frame["same_title_conflict_group"]
    frame = frame.sort_values("source_row_number", kind="stable").reset_index(drop=True)

    summary = {
        "raw_rows": int(len(raw)),
        "valid_date_rows": int((~raw["date_parse_failed"]).sum()),
        "invalid_date_rows": int(raw["date_parse_failed"].sum()),
        "exact_duplicate_rows_removed": int((~keep).sum()),
        "exact_duplicate_representatives_retained": int(frame["exact_content_duplicate_group"].sum()),
        "ambiguous_same_title_rows_retained": int(frame["same_title_conflict_group"].sum()),
        "publication_timestamps_present": False,
        "deduplication_rule": "remove exact cleaned title+body duplicates only; retain conflicting versions",
    }
    return frame, summary


def align_news_to_fixings(articles: pd.DataFrame, rates: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Align article calendar dates to fixing dates using date-only policy."""
    if mode not in {"strictly_later", "same_date"}:
        raise ValueError("mode must be 'strictly_later' or 'same_date'")
    frame = articles.copy()
    valid = frame["Date"].notna()
    fixing_values = rates["Date"].sort_values().drop_duplicates().to_numpy()
    positions = fixing_values.searchsorted(frame.loc[valid, "Date"].to_numpy(), side="right" if mode == "strictly_later" else "left")
    aligned = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    in_range = positions < len(fixing_values)
    aligned.loc[frame.index[valid][in_range]] = fixing_values[positions[in_range]]
    frame["alignment_mode"] = mode
    frame["Aligned_Date"] = aligned
    frame["alignment_failed"] = frame["Aligned_Date"].isna()
    frame["out_of_calendar_or_invalid_date_flag"] = frame["alignment_failed"]
    frame["same_date_intraday_uncertain"] = mode == "same_date"
    # An article outside the JISDOR range is unalignable, not proof of leakage.
    # Missing/invalid dates are data-integrity failures, also not automatically
    # confirmed temporal contamination.
    frame["confirmed_contamination_flag"] = False
    frame["heuristic_contamination_flag"] = frame["same_title_conflict_group"] | frame["same_date_intraday_uncertain"]
    frame["aligned_after_article_date"] = frame["Aligned_Date"].gt(frame["Date"])
    if mode == "strictly_later":
        assert not frame.loc[~frame["alignment_failed"], "Aligned_Date"].le(frame.loc[~frame["alignment_failed"], "Date"]).any()
    else:
        assert not frame.loc[~frame["alignment_failed"], "Aligned_Date"].lt(frame.loc[~frame["alignment_failed"], "Date"]).any()
    return frame


def build_master_calendar(rates: pd.DataFrame, aligned_articles: pd.DataFrame) -> pd.DataFrame:
    """Build the complete calendar and date-level news-window audit fields."""
    calendar = rates.sort_values("Date", kind="stable").reset_index(drop=True).copy()
    calendar["next_Date"] = calendar["Date"].shift(-1)
    calendar["next_USD_IDR"] = calendar["USD_IDR"].shift(-1)
    calendar["next_return"] = (calendar["next_USD_IDR"] - calendar["USD_IDR"]) / calendar["USD_IDR"]
    calendar["target_available"] = calendar["next_USD_IDR"].notna()
    calendar["is_flat_movement"] = calendar["next_return"].eq(0) & calendar["target_available"]
    calendar["target_direction"] = calendar["next_return"].gt(0).where(calendar["target_available"])
    calendar["calendar_index"] = range(len(calendar))
    calendar["is_warmup"] = calendar["calendar_index"] < LOOKBACK_WARMUP_ROWS

    news = aligned_articles[aligned_articles["Aligned_Date"].notna()].copy()
    counts = news.groupby("Aligned_Date", as_index=False).agg(
        news_article_count=("article_id", "size"),
        news_article_ids=("article_id", lambda values: json.dumps(sorted(values.tolist()))),
        news_timing_uncertainty_count=("timing_uncertainty_flag", "sum"),
        news_confirmed_contamination_count=("confirmed_contamination_flag", "sum"),
        news_heuristic_flag_count=("heuristic_contamination_flag", "sum"),
    )
    calendar = calendar.merge(counts, left_on="Date", right_on="Aligned_Date", how="left")
    calendar = calendar.drop(columns=["Aligned_Date"])
    calendar["news_article_count"] = calendar["news_article_count"].fillna(0).astype(int)
    calendar["has_news"] = calendar["news_article_count"].gt(0)
    calendar["news_article_ids"] = calendar["news_article_ids"].fillna("[]")
    for column in ["news_timing_uncertainty_count", "news_confirmed_contamination_count", "news_heuristic_flag_count"]:
        calendar[column] = calendar[column].fillna(0).astype(int)
    calendar["supervised_eligible"] = (
        calendar["target_available"] & ~calendar["is_flat_movement"] & ~calendar["is_warmup"]
    )
    assert len(calendar) == len(rates)
    assert calendar["Date"].is_unique
    assert calendar["news_article_count"].ge(0).all()
    return calendar


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_audit(fx_path: Path = FX_PATH, news_path: Path = NEWS_PATH) -> dict[str, Any]:
    rates = load_jisdor(fx_path)
    raw_news = load_cnbc(news_path)
    articles, dedup_summary = clean_and_deduplicate_articles(raw_news)
    strict = align_news_to_fixings(articles, rates, "strictly_later")
    same_date = align_news_to_fixings(articles, rates, "same_date")
    calendar = build_master_calendar(rates, strict)

    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    articles.to_csv(INTERMEDIATE / "articles_cleaned_deduplicated.csv", index=False)
    strict.to_csv(INTERMEDIATE / "articles_aligned_strict.csv", index=False)
    same_date.to_csv(INTERMEDIATE / "articles_aligned_same_date.csv", index=False)
    calendar.to_csv(INTERMEDIATE / "master_calendar_news_windows.csv", index=False)

    summary: dict[str, Any] = {
        "module": "phase5_module2",
        "inputs": {
            "fx_path": str(fx_path),
            "news_path": str(news_path),
            "fx_sha256": sha256_file(fx_path),
            "news_sha256": sha256_file(news_path),
            "fx_rows": int(len(rates)),
            "news_rows_raw": int(len(raw_news)),
        },
        "deduplication": dedup_summary,
        "alignment": {
            "strictly_later_rows_aligned": int(strict["Aligned_Date"].notna().sum()),
            "strictly_later_rows_failed": int(strict["alignment_failed"].sum()),
            "same_date_rows_aligned": int(same_date["Aligned_Date"].notna().sum()),
            "same_date_rows_failed": int(same_date["alignment_failed"].sum()),
            "same_date_intraday_availability_verified": False,
        },
        "calendar": {
            "rows": int(len(calendar)),
            "date_min": str(calendar["Date"].min().date()),
            "date_max": str(calendar["Date"].max().date()),
            "flat_rows": int(calendar["is_flat_movement"].sum()),
            "warmup_rows": int(calendar["is_warmup"].sum()),
            "empty_news_rows": int((~calendar["has_news"]).sum()),
            "supervised_eligible_rows": int(calendar["supervised_eligible"].sum()),
        },
        "contamination": {
            "confirmed": {
                "invalid_news_dates": int(raw_news["date_parse_failed"].sum()),
                "confirmed_temporal_contamination": int(strict["confirmed_contamination_flag"].sum()),
            },
            "out_of_calendar_or_unalignable": {
                "strict_alignment_failures": int(strict["alignment_failed"].sum()),
            },
            "heuristic_or_uncertain": {
                "publication_timestamp_missing": True,
                "same_title_conflict_rows_retained": int(articles["same_title_conflict_group"].sum()),
                "same_date_intraday_uncertainty": int(same_date["same_date_intraday_uncertain"].sum()),
            },
        },
        "policies": {
            "no_publication_timestamps_fabricated": True,
            "flat_rows_retained_in_master_calendar": True,
            "empty_news_days_retained": True,
            "warmup_rows_retained": True,
            "model_ready_datasets_built": False,
        },
    }
    _write_json(MANIFESTS / "module2_audit_summary.json", summary)
    _write_json(
        MANIFESTS / "module2_input_manifest.json",
        {
            "inputs": summary["inputs"],
            "source_columns": {
                "fx": list(rates.columns),
                "news": list(raw_news.columns),
            },
            "source_timestamp_finding": "CNBC input exposes Date but no verified publication timestamp column",
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fx-path", type=Path, default=FX_PATH)
    parser.add_argument("--news-path", type=Path, default=NEWS_PATH)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.fx_path, args.news_path), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
