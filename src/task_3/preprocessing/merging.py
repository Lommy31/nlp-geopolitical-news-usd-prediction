import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import cleaning
else:
    import cleaning

MARKET_FEATURES = [
    "log_return_t", "return_lag1", "return_lag2", "return_lag3", "return_lag5",
    "volatility_5", "volatility_20", "ma_ratio_5", "ma_ratio_20",
    "momentum_5", "momentum_10", "rsi_14", "input_gap_days", "decision_weekday",
]

def build_rate_features(raw):
    d = raw.copy()
    if not {"Date", "USD_IDR"}.issubset(d):
        raise ValueError("Expected Date and USD_IDR in the processed JISDOR input.")
    d["Date"] = pd.to_datetime(d.Date, errors="coerce")
    d["USD_IDR"] = pd.to_numeric(d.USD_IDR, errors="coerce")
    if d.Date.isna().any() or d.Date.duplicated().any():
        raise ValueError("Invalid or duplicate JISDOR dates; do not silently discard them.")
    if (d.Date != d.Date.dt.normalize()).any():
        raise ValueError("JISDOR Date must be date-only.")
    if (~np.isfinite(d.USD_IDR) | d.USD_IDR.le(0)).any():
        raise ValueError("JISDOR prices must be finite and positive.")
    if d.Date.dt.dayofweek.ge(5).any():
        raise ValueError("Unexpected weekend observation in the master JISDOR calendar.")
    d = d.sort_values("Date").reset_index(drop=True)
    d.insert(0, "calendar_index", np.arange(len(d)))
    p = d.USD_IDR
    r = np.log(p / p.shift(1))
    d["log_return_t"] = r
    for lag in [1, 2, 3, 5]:
        d[f"return_lag{lag}"] = r.shift(lag)
    for width in [5, 20]:
        d[f"volatility_{width}"] = r.rolling(width, min_periods=width).std()
        d[f"ma_ratio_{width}"] = p / p.rolling(width, min_periods=width).mean() - 1
    for width in [5, 10]:
        d[f"momentum_{width}"] = np.log(p / p.shift(width))
    gain = p.diff().clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-p.diff().clip(upper=0)).rolling(14, min_periods=14).mean()
    d["rsi_14"] = (100 - 100 / (1 + gain / loss)).mask(loss.eq(0) & gain.gt(0), 100).mask(loss.eq(0) & gain.eq(0), 50)
    d["input_gap_days"] = d.Date.diff().dt.days
    d["decision_weekday"] = d.Date.dt.dayofweek
    d["next_jisdor"] = p.shift(-1)
    d["target_date"] = d.Date.shift(-1)
    d["log_return_next"] = np.log(d.next_jisdor / p)
    d["y"] = pd.Series(np.select([d.next_jisdor.gt(p), d.next_jisdor.lt(p)], [1, 0], default=np.nan), dtype="Int64")
    d["target_status"] = np.select([d.next_jisdor.gt(p), d.next_jisdor.lt(p), d.next_jisdor.eq(p)], ["UP", "DOWN", "FLAT"], default="UNAVAILABLE")
    d["target_weekday"] = d.target_date.dt.dayofweek
    d["realized_target_gap_days"] = (d.target_date - d.Date).dt.days
    d["features_ready"] = d[MARKET_FEATURES].notna().all(axis=1)
    d["eligible_binary"] = d.features_ready & d.y.notna()

    return d

def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def align_articles(articles, calendar):
    d = articles.copy()
    indices = np.searchsorted(calendar.Date.to_numpy(), d.source_date.to_numpy(), side="right")
    valid = d.selected_for_alignment & (indices < len(calendar))
    d["anchor_date"] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    d.loc[valid, "anchor_date"] = calendar.Date.to_numpy()[indices[valid]]
    return d

def merge_news(original, cnbc, calendar, progress=None):
    original_views = align_articles(cleaning.prepare_articles(original), calendar)
    counts = {"original_processed": len(original), "original_aligned": int(original_views.anchor_date.notna().sum()),
              "cnbc_raw": len(cnbc)}
    d = cnbc.loc[~cnbc.URL.isin(set(original.URL)) & cnbc.URL.ne("")].drop_duplicates("URL").copy()
    counts["cnbc_new_unique_urls"] = len(d)
    d["Title"] = d.Title.map(cleaning.fix_spaced_punctuation)
    filters = [
        ("domain", lambda x: ~x.Source.isin(cleaning.BLOCKED_DOMAINS)),
        ("single_company", lambda x: ~x.Text.map(cleaning.is_single_company_content)),
        ("wire_mirror", lambda x: ~x.Text.map(cleaning.is_wire_service_mirror)),
        ("placeholder", lambda x: ~x.Text.map(cleaning.is_placeholder_content)),
        ("anchor", lambda x: x.apply(lambda row: cleaning.has_anchor(row.Event, row.Text), axis=1)),
    ]
    for name, mask in filters:
        d = d.loc[mask(d)].copy()
        counts[f"cnbc_after_{name}"] = len(d)
    d["Text_clean"] = d.Text.map(cleaning.clean_scraped_text)
    d["word_count"] = d.Text_clean.str.split().str.len()
    d = d.loc[d.word_count.ge(150)].copy()
    counts["cnbc_after_150_words"] = len(d)
    d["title_key"] = d.Title.map(cleaning.normalize_title)
    original_keys = set(original.Title.map(cleaning.fix_spaced_punctuation).map(cleaning.normalize_title))
    d = d.loc[~d.title_key.isin(original_keys)].sort_values("word_count", ascending=False, kind="stable").drop_duplicates("title_key")
    d = d.sort_values(["Date", "URL"], kind="stable").reset_index(drop=True)
    counts["cnbc_after_title_dedup"] = len(d)
    # Excluded original bodies also block duplicate CNBC copies.
    seen_content = {hashlib.sha256(key.encode()).hexdigest() for key in original_views.content_key if len(key) >= 80}
    chunks = []
    for start in range(0, len(d), 128):
        q = cleaning.prepare_articles(d.iloc[start:start + 128][cleaning.NEWS_COLUMNS], prefix="cnbc_", source_offset=start)
        for index in q.index:
            key = q.at[index, "content_key"]
            if len(key) >= 80:
                digest = hashlib.sha256(key.encode()).hexdigest()
                if digest in seen_content:
                    q.at[index, "duplicate_of"] = "existing_union_content"
                else:
                    seen_content.add(digest)
        chunks.append(align_articles(cleaning.select_articles(q), calendar))
        if progress is not None and start % 2048 == 0:
            progress(f"Prepared {min(start + 128, len(d))}/{len(d)} CNBC candidates")
    added_views = pd.concat(chunks, ignore_index=True) if chunks else original_views.iloc[:0].copy()
    original_views["corpus"] = "original"
    added_views["corpus"] = "cnbc_addition"
    views = pd.concat([original_views, added_views], ignore_index=True)
    articles = views.loc[views.anchor_date.notna()].sort_values("article_id").reset_index(drop=True)
    if not articles.article_id.is_unique or not articles.URL.is_unique:
        raise ValueError("Duplicate article IDs or URLs remain.")
    if articles.loc[articles.content_key.str.len().ge(80), "content_key"].duplicated().any():
        raise ValueError("Duplicate full article bodies remain.")
    if articles.known_temporal_contamination.any() or not articles.anchor_date.gt(articles.source_date).all():
        raise ValueError("Invalid article quarantine or date alignment.")
    counts.update(cnbc_added_aligned=int(articles.corpus.eq("cnbc_addition").sum()), combined_aligned=len(articles))
    excluded = views.loc[views.anchor_date.isna()]
    excluded = excluded.copy()
    excluded.loc[excluded.selected_for_alignment, "selection_reason"] = "after_last_available_fixing"
    exclusions = {corpus: group.selection_reason.value_counts().to_dict() for corpus, group in excluded.groupby("corpus")}
    return articles, counts, exclusions

def build_dataset(root, progress=None):
    root = Path(root)
    paths = {
        "original": root / "data/processed/news_clean.csv",
        "cnbc": root / "data/raw/news_cnbc.csv",
        "rates": root / "data/processed/usd_idr_jisdor_clean.csv",
    }
    hashes = {str(path.relative_to(root)): sha256(path) for path in paths.values()}
    if hashes[str(paths["original"].relative_to(root))] != cleaning.ORIGINAL_SOURCE_SHA256:
        raise ValueError("The original source changed; reconcile the confirmed article quarantine before rebuilding.")
    calendar = build_rate_features(pd.read_csv(paths["rates"]))
    original = pd.read_csv(paths["original"]).fillna("")
    cnbc = pd.read_csv(paths["cnbc"]).fillna("")
    articles, counts, exclusions = merge_news(original, cnbc, calendar, progress)
    counts_per_date = articles.groupby("anchor_date").size()
    calendar["n_berita"] = calendar.Date.map(counts_per_date).fillna(0).astype(int)
    calendar["has_news"] = calendar.n_berita.gt(0).astype(int)
    return calendar, articles, {"input_sha256": hashes, "corpus_counts": counts, "excluded_article_reasons": exclusions}

