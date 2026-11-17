"""Task 3 Module 4: financial and baseline feature datasets.

The module preserves the audited 1,202-row calendar and keeps metadata,
targets, and predictors logically separate.  It contains an exact Task 2
legacy reproduction path and a controlled Task 3 path based on Module 2's
provenance-preserving article outputs.  TF-IDF/SVD is represented only by its
daily text input here; no learned text transformation is fitted globally.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HANA = ROOT / "hana-work"
NLTK_DATA = HANA / "nltk_data"
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA))

import nltk  # noqa: E402

if str(NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA))

import pysentiment2 as ps  # noqa: E402
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: E402


MASTER_PATH = HANA / "data/intermediate/master_calendar_news_windows.csv"
STRICT_ARTICLES_PATH = HANA / "data/intermediate/articles_aligned_strict.csv"
SAME_ARTICLES_PATH = HANA / "data/intermediate/articles_aligned_same_date.csv"
RAW_NEWS_PATH = ROOT / "data/raw/news_cnbc.csv"
FX_PATH = ROOT / "data/processed/usd_idr_jisdor_clean.csv"
FOLD_MANIFEST_PATH = HANA / "data/manifests/module3_fold_manifest.json"
INTERMEDIATE = HANA / "data/intermediate"
PROCESSED = HANA / "data/processed"
MANIFESTS = HANA / "data/manifests"

TECHNICAL_FEATURES = [
    "return_t", "return_lag1", "return_lag2", "return_lag3", "return_lag5",
    "volatility_5d", "volatility_20d", "ma_ratio_5d", "ma_ratio_20d",
    "momentum_5d", "momentum_10d", "rsi_14",
]
TITLE_SENTIMENT_FEATURES = [
    "vader_title_compound", "vader_title_pos", "vader_title_neg",
    "lm_title_polarity", "lm_title_subjectivity", "title_sentiment_decay3",
    "title_sentiment_decay5", "title_shock_extremity", "lm_title_shock",
    "sent_vol_interact",
]
BODY_SENTIMENT_FEATURES = [
    "vader_body_compound", "vader_body_pos", "vader_body_neg",
    "lm_body_polarity", "lm_body_subjectivity",
]
NLP_FEATURES = TITLE_SENTIMENT_FEATURES + BODY_SENTIMENT_FEATURES
AVAILABILITY_FEATURES = ["n_articles", "has_news", "log1p_n_articles"]
MODEL_DATASETS = {"A2", "B0", "B1", "B2"}
TIMING_MODES = {"strict": STRICT_ARTICLES_PATH, "same_date": SAME_ARTICLES_PATH}

METADATA_COLUMNS = [
    "Date", "calendar_index", "module3_split", "next_Date", "target_direction",
    "target_available", "is_flat_movement", "is_warmup", "supervised_eligible",
]
FORBIDDEN_PREDICTOR_COLUMNS = {
    "Date", "calendar_index", "module3_split", "next_Date", "next_USD_IDR",
    "next_return", "target_direction", "target_date", "next_jisdor",
    "log_return_next", "y", "target_available", "is_flat_movement",
    "is_warmup", "supervised_eligible", "split", "fold", "fold_id",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Exact Task 2 RSI formula."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def build_rate_features_exact(df_rates: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact Task 2 formulas while retaining warm-up rows."""
    d = df_rates.sort_values("Date").reset_index(drop=True).copy()
    d["return_t"] = d["USD_IDR"].pct_change()
    d["return_lag1"] = d["return_t"].shift(1)
    d["return_lag2"] = d["return_t"].shift(2)
    d["return_lag3"] = d["return_t"].shift(3)
    d["return_lag5"] = d["return_t"].shift(5)
    d["volatility_5d"] = d["return_t"].rolling(5).std()
    d["volatility_20d"] = d["return_t"].rolling(20).std()
    d["ma_ratio_5d"] = d["USD_IDR"] / d["USD_IDR"].rolling(5).mean()
    d["ma_ratio_20d"] = d["USD_IDR"] / d["USD_IDR"].rolling(20).mean()
    d["momentum_5d"] = (d["USD_IDR"] - d["USD_IDR"].shift(5)) / d["USD_IDR"].shift(5)
    d["momentum_10d"] = (d["USD_IDR"] - d["USD_IDR"].shift(10)) / d["USD_IDR"].shift(10)
    d["rsi_14"] = compute_rsi(d["USD_IDR"])
    return d


def load_master_calendar(path: Path = MASTER_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date", "next_Date"])
    if len(frame) != 1202 or not frame["calendar_index"].equals(pd.Series(range(1202), name="calendar_index")):
        raise AssertionError("Module 2 master calendar must remain the audited 1,202-row calendar")
    return frame


def load_module3_splits(path: Path = FOLD_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def module3_split_labels(calendar: pd.DataFrame, manifest: dict[str, Any]) -> pd.Series:
    labels = pd.Series("development_unused_or_excluded", index=calendar.index, dtype="object")
    labels.loc[calendar["calendar_index"].isin(manifest["configuration"]["final_boundary_purge_indices"])] = "final_boundary_purge"
    labels.loc[calendar["calendar_index"].isin(manifest["locked_test"]["calendar_indices"])] = "locked_test_calendar"
    for fold in manifest["folds"]:
        labels.loc[calendar["calendar_index"].isin(fold["train_calendar_indices"])] = f"fold_{fold['fold_id']}_train"
        labels.loc[calendar["calendar_index"].isin(fold["validation_calendar_indices"])] = f"fold_{fold['fold_id']}_validation"
        labels.loc[calendar["calendar_index"].isin(fold["purged_calendar_indices"])] = f"fold_{fold['fold_id']}_boundary_purge"
    return labels


def build_technical_calendar(calendar: pd.DataFrame, fx_path: Path = FX_PATH) -> pd.DataFrame:
    rates = pd.read_csv(fx_path, parse_dates=["Date"])
    rates = rates[["Date", "USD_IDR"]]
    technical = build_rate_features_exact(rates)
    technical["Date"] = pd.to_datetime(technical["Date"])
    merged = calendar.merge(technical[["Date"] + TECHNICAL_FEATURES], on="Date", how="left", validate="one_to_one")
    if len(merged) != len(calendar):
        raise AssertionError("Technical feature merge changed calendar coverage")
    return merged


def clean_text_field(value: Any) -> str:
    if pd.isna(value):
        return ""
    value = re.sub(r"<.*?>", " ", str(value))
    value = re.sub(r"http\S+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_title(title: Any) -> str:
    title = re.sub(r"^[^|]{1,20}\|\s*", "", str(title)).lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return " ".join(title.split()[:6])


def legacy_task2_clean_news(raw_news: pd.DataFrame, min_words: int = 150) -> pd.DataFrame:
    """Reproduce Task 2 cleaning exactly for the separate legacy track."""
    d = raw_news.copy()
    d["Title"] = d["Title"].apply(clean_text_field)
    d["Text_clean"] = d["Text"].apply(clean_text_field)
    d["word_count"] = d["Text_clean"].str.split().str.len()
    d = d[d["word_count"] >= min_words]
    d["title_key"] = d["Title"].apply(normalize_title)
    d = d.sort_values("word_count", ascending=False).drop_duplicates(subset="title_key")
    return d.drop(columns=["word_count", "title_key", "Text"])


def score_news_sentiment_exact(articles: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Task 2 VADER/LM fields and text truncation lengths."""
    d = articles.copy()
    vader = SentimentIntensityAnalyzer()
    lm = ps.LM()
    title_scores = d["Title"].apply(vader.polarity_scores)
    d["vader_title_compound"] = [score["compound"] for score in title_scores]
    d["vader_title_pos"] = [score["pos"] for score in title_scores]
    d["vader_title_neg"] = [score["neg"] for score in title_scores]
    body_scores = d["Text_clean"].apply(lambda text: vader.polarity_scores(text[:2500]))
    d["vader_body_compound"] = [score["compound"] for score in body_scores]
    d["vader_body_pos"] = [score["pos"] for score in body_scores]
    d["vader_body_neg"] = [score["neg"] for score in body_scores]

    def lm_scores(text: str) -> tuple[float, float]:
        try:
            score = lm.get_score(lm.tokenize(text))
            return float(score["Polarity"]), float(score["Subjectivity"])
        except Exception:
            return 0.0, 0.0

    title_lm = d["Title"].apply(lm_scores)
    d["lm_title_polarity"] = [score[0] for score in title_lm]
    d["lm_title_subjectivity"] = [score[1] for score in title_lm]
    body_lm = d["Text_clean"].apply(lambda text: lm_scores(text[:3000]))
    d["lm_body_polarity"] = [score[0] for score in body_lm]
    d["lm_body_subjectivity"] = [score[1] for score in body_lm]
    return d


def aggregate_basic_nlp(articles: pd.DataFrame, date_column: str = "Aligned_Date") -> pd.DataFrame:
    d = articles.copy()
    d["Aligned_Date_str"] = pd.to_datetime(d[date_column]).dt.strftime("%Y-%m-%d")
    agg_rules = {
        "vader_title_compound": ["mean", "min", "max"],
        "vader_title_pos": "mean", "vader_title_neg": "mean",
        "vader_body_compound": ["mean", "min", "max"],
        "vader_body_pos": "mean", "vader_body_neg": "mean",
        "lm_title_polarity": ["mean", "min", "max"],
        "lm_title_subjectivity": "mean",
        "lm_body_polarity": ["mean", "min", "max"],
        "lm_body_subjectivity": "mean",
    }
    daily = d.groupby("Aligned_Date_str").agg(agg_rules)
    daily.columns = [f"{c}_{s}" if s != "mean" else c for c, s in daily.columns]
    daily["n_articles"] = d.groupby("Aligned_Date_str").size()
    daily["daily_titles"] = d.groupby("Aligned_Date_str")["Title"].apply(lambda values: " ".join(values))
    daily = daily.reset_index().rename(columns={"Aligned_Date_str": "Date"})
    return daily


def add_task2_daily_derived_features(daily: pd.DataFrame, technical_calendar: pd.DataFrame) -> pd.DataFrame:
    """Apply Task 2 daily fill, decay, shock, and interaction definitions."""
    calendar_dates = technical_calendar[["Date", "volatility_5d"]].copy()
    calendar_dates["Date"] = calendar_dates["Date"].dt.strftime("%Y-%m-%d")
    merged = calendar_dates.merge(daily, on="Date", how="left", validate="one_to_one")
    merged["has_news"] = (~merged["n_articles"].isna()).astype(int)
    nlp_cols = [column for column in daily.columns if column not in ("Date", "daily_titles")]
    for column in nlp_cols:
        merged[column] = merged[column].fillna(0.0)
    merged["n_articles"] = merged["n_articles"].fillna(0).astype(int)
    merged["daily_titles"] = merged["daily_titles"].fillna("")
    merged["log1p_n_articles"] = np.log1p(merged["n_articles"])
    merged["title_sentiment_decay3"] = merged["vader_title_compound"].ewm(span=3, adjust=False).mean()
    merged["title_sentiment_decay5"] = merged["vader_title_compound"].ewm(span=5, adjust=False).mean()
    merged["title_shock_extremity"] = merged["vader_title_compound"].abs()
    merged["lm_title_shock"] = merged["lm_title_polarity"].abs()
    merged["sent_vol_interact"] = merged["vader_title_compound"] * merged["volatility_5d"]
    return merged


def build_availability(calendar: pd.DataFrame, aligned_articles: pd.DataFrame) -> pd.DataFrame:
    counts = (
        aligned_articles[aligned_articles["Aligned_Date"].notna()]
        .groupby("Aligned_Date").size().rename("n_articles").reset_index()
    )
    counts["Date"] = pd.to_datetime(counts["Aligned_Date"])
    counts = counts.drop(columns=["Aligned_Date"])
    result = calendar[["Date"]].merge(counts, on="Date", how="left", validate="one_to_one")
    result["n_articles"] = result["n_articles"].fillna(0).astype(int)
    result["has_news"] = result["n_articles"].gt(0).astype(int)
    result["log1p_n_articles"] = np.log1p(result["n_articles"])
    assert len(result) == len(calendar)
    assert (result.loc[result["n_articles"].eq(0), "has_news"] == 0).all()
    return result


def assert_predictor_whitelist(columns: Iterable[str]) -> None:
    forbidden = set(columns) & FORBIDDEN_PREDICTOR_COLUMNS
    if forbidden:
        raise AssertionError(f"Forbidden predictor columns: {sorted(forbidden)}")


def build_logical_datasets(calendar: pd.DataFrame, technical: pd.DataFrame, timing_mode: str, aligned_articles: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, pd.DataFrame]:
    availability = build_availability(calendar, aligned_articles)
    # The Module 2 calendar contains strict-track ``has_news`` for audit
    # context.  Replace all timing-sensitive availability fields explicitly so
    # the same-date track cannot inherit strict-track values through a merge.
    technical_without_availability = technical.drop(
        columns=[column for column in AVAILABILITY_FEATURES if column in technical],
        errors="ignore",
    )
    base = technical_without_availability.merge(availability, on="Date", how="left", validate="one_to_one")
    base["module3_split"] = module3_split_labels(base, manifest).to_numpy()
    metadata = base[METADATA_COLUMNS].copy()
    a2 = base[TECHNICAL_FEATURES].copy()
    b0 = base[TECHNICAL_FEATURES + AVAILABILITY_FEATURES].copy()

    scored = score_news_sentiment_exact(aligned_articles[aligned_articles["Aligned_Date"].notna()])
    daily = aggregate_basic_nlp(scored)
    nlp = add_task2_daily_derived_features(daily, technical)
    nlp["Date"] = pd.to_datetime(nlp["Date"])
    nlp_merge_columns = ["Date"] + NLP_FEATURES + ["daily_titles"]
    b1_full = base.merge(nlp[nlp_merge_columns], on="Date", how="left", validate="one_to_one")
    for column in NLP_FEATURES:
        b1_full[column] = b1_full[column].fillna(0.0)
    b1 = b1_full[TECHNICAL_FEATURES + AVAILABILITY_FEATURES + NLP_FEATURES].copy()

    b2 = b0.copy()
    text_inputs = base[["Date", "calendar_index"]].merge(
        nlp[["Date", "daily_titles"]], on="Date", how="left", validate="one_to_one"
    )
    datasets = {"A2": a2, "B0": b0, "B1": b1, "B2": b2}
    for name, predictors in datasets.items():
        assert_predictor_whitelist(predictors.columns)
        assert len(predictors) == len(metadata) == 1202
        datasets[name] = predictors
    datasets["metadata"] = metadata
    datasets["B2_text_inputs"] = text_inputs
    return datasets


def _save_dataset_bundle(datasets: dict[str, pd.DataFrame], timing_mode: str) -> list[dict[str, Any]]:
    records = []
    for name in MODEL_DATASETS:
        prefix = f"{name.lower()}_{timing_mode}"
        predictor_path = PROCESSED / f"{prefix}_predictors.csv"
        datasets[name].to_csv(predictor_path, index=False)
        records.append({"dataset": name, "timing_mode": timing_mode, "kind": "predictors", "path": str(predictor_path.relative_to(HANA)), "rows": len(datasets[name]), "columns": list(datasets[name].columns)})
    metadata_path = PROCESSED / f"baseline_{timing_mode}_metadata.csv"
    datasets["metadata"].to_csv(metadata_path, index=False)
    text_path = PROCESSED / f"b2_{timing_mode}_daily_titles_input.csv"
    datasets["B2_text_inputs"].to_csv(text_path, index=False)
    records.extend([
        {"dataset": "metadata", "timing_mode": timing_mode, "kind": "metadata", "path": str(metadata_path.relative_to(HANA)), "rows": len(datasets["metadata"]), "columns": list(datasets["metadata"].columns)},
        {"dataset": "B2", "timing_mode": timing_mode, "kind": "unfitted_text_input", "path": str(text_path.relative_to(HANA)), "rows": len(datasets["B2_text_inputs"]), "columns": list(datasets["B2_text_inputs"].columns)},
    ])
    return records


def run_module4() -> dict[str, Any]:
    calendar = load_master_calendar()
    manifest = load_module3_splits()
    technical = build_technical_calendar(calendar)
    outputs: list[dict[str, Any]] = []
    timing_summaries: dict[str, Any] = {}
    for timing_mode, article_path in TIMING_MODES.items():
        articles = pd.read_csv(article_path, parse_dates=["Date", "Aligned_Date"])
        datasets = build_logical_datasets(calendar, technical, timing_mode, articles, manifest)
        outputs.extend(_save_dataset_bundle(datasets, timing_mode))
        timing_summaries[timing_mode] = {
            "article_rows": int(articles["Aligned_Date"].notna().sum()),
            "dataset_rows": {name: len(datasets[name]) for name in MODEL_DATASETS},
            "date_min": str(calendar["Date"].min().date()),
            "date_max": str(calendar["Date"].max().date()),
            "predictor_columns": {name: list(datasets[name].columns) for name in MODEL_DATASETS},
        }

    # Separate exact legacy rate and basic-NLP artifacts, never used as the
    # controlled Task 3 datasets.
    raw_news = pd.read_csv(RAW_NEWS_PATH)
    rates = pd.read_csv(FX_PATH, parse_dates=["Date"])
    legacy_rates = build_rate_features_exact(rates)
    legacy_rates.to_csv(INTERMEDIATE / "legacy_task2_rate_features_untrimmed.csv", index=False)
    legacy_rates_exact = legacy_rates.copy()
    legacy_rates_exact["next_close"] = legacy_rates_exact["USD_IDR"].shift(-1)
    legacy_rates_exact["next_return"] = (legacy_rates_exact["next_close"] - legacy_rates_exact["USD_IDR"]) / legacy_rates_exact["USD_IDR"]
    legacy_rates_exact["target_direction"] = (legacy_rates_exact["next_return"] > 0).astype(int)
    legacy_rates_exact = legacy_rates_exact.dropna().reset_index(drop=True)
    legacy_rate_context = legacy_rates_exact[["Date", "volatility_5d"]].copy()
    legacy_rates_exact["Date"] = legacy_rates_exact["Date"].dt.strftime("%Y-%m-%d")
    legacy_rates_exact.to_csv(INTERMEDIATE / "legacy_task2_rate_features.csv", index=False)
    legacy_news = legacy_task2_clean_news(raw_news)
    legacy_news.to_csv(INTERMEDIATE / "legacy_task2_news_cleaned.csv", index=False)
    legacy_news["Date"] = pd.to_datetime(legacy_news["Date"], errors="coerce").dt.normalize()
    fixing_dates = rates["Date"].sort_values().drop_duplicates().to_numpy()
    for mode, path in TIMING_MODES.items():
        legacy_like = legacy_news.copy()
        positions = fixing_dates.searchsorted(legacy_like["Date"].to_numpy(), side="right" if mode == "strict" else "left")
        aligned_dates = pd.Series(pd.NaT, index=legacy_like.index, dtype="datetime64[ns]")
        valid = positions < len(fixing_dates)
        aligned_dates.loc[legacy_like.index[valid]] = fixing_dates[positions[valid]]
        legacy_like["Aligned_Date"] = aligned_dates
        legacy_like = legacy_like[legacy_like["Aligned_Date"].notna()].copy()
        scored = score_news_sentiment_exact(legacy_like)
        daily = aggregate_basic_nlp(scored)
        legacy_daily = add_task2_daily_derived_features(daily, legacy_rate_context)
        legacy_daily.to_csv(INTERMEDIATE / f"legacy_task2_basic_nlp_daily_{mode}.csv", index=False)

    summary = {
        "module": "phase5_module4",
        "technical_features": TECHNICAL_FEATURES,
        "availability_features": AVAILABILITY_FEATURES,
        "basic_nlp_features": NLP_FEATURES,
        "calendar_rows": len(calendar),
        "timing_summaries": timing_summaries,
        "legacy_task2_clean_news_rows": len(legacy_news),
        "legacy_task2_rate_feature_rows_untrimmed": len(legacy_rates),
        "legacy_task2_rate_feature_rows_exact_task2_dropna": len(legacy_rates_exact),
        "legacy_task2_rate_feature_formula_source": "src/task2_pipeline.py::build_rate_features and ::compute_rsi",
        "legacy_task2_basic_aggregation_source": "src/task2_pipeline.py::score_news_sentiment and ::aggregate_and_merge",
        "tfidf_svd": {"prepared": True, "fitted": False, "vocabulary_fit_scope": "deferred to fold-local Module 5 fitting", "n_components": 5, "max_features": 500, "ngram_range": [1, 2], "stop_words": "english", "sublinear_tf": True, "random_state": 42},
        "nltk_data_path": str(NLTK_DATA),
        "module3_fold_manifest_sha256": _sha256(FOLD_MANIFEST_PATH),
        "outputs": outputs,
        "model_ready_training_or_evaluation": False,
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "module4_feature_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_module4(), indent=2, sort_keys=True, default=str))
