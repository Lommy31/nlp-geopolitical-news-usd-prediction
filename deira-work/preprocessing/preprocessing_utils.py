"""Offline, auditable preparation of processed news and JISDOR inputs.

Dependencies: Python 3.10+, numpy, pandas. No network, learned vocabulary,
sentiment model, or embedding model is used here. See the accompanying notebook.
"""
from __future__ import annotations

import calendar
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

WIB = "Asia/Jakarta"
NEWS_COLUMNS = ["Date", "Title", "Source", "URL", "Event", "Text_clean"]
MARKET_FEATURES = [
    "log_return_t", "return_lag1", "return_lag2", "return_lag3", "return_lag5",
    "volatility_5", "volatility_20", "ma_ratio_5", "ma_ratio_20",
    "momentum_5", "momentum_10", "rsi_14", "input_gap_days", "decision_weekday",
]
TARGET_COLUMNS = ["next_jisdor", "target_date", "log_return_next", "y", "target_status",
                  "target_weekday", "realized_target_gap_days"]
REVIEW_COLUMNS = ["article_id", "geo_decision", "review_note", "published_at",
                  "available_at", "source_timezone", "timestamp_verified",
                  "body_asof_verified", "evidence"]
MONTHS = "|".join(calendar.month_name[1:])
FULL_DATE = re.compile(rf"\b({MONTHS})\s+(\d{{1,2}}),?\s+(20\d{{2}})\b", re.I)
TOKEN = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*%?|[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
HTML_TAG = re.compile(r"</?(?:p|div|span|a|br|strong|em|article|section|h[1-6])\b[^>]*>", re.I)
TAIL_BOUNDARIES = [
    re.compile(p, re.I) for p in [
        r"^get your ticket to pro live\b", r"^get a daily dose of\b",
        r"^publish news of your\b", r"^read the latest financial and business news\b",
        r"^please keep your comments respectful\b", r"^you must be logged in to post a comment\b",
        r"^our comment policy\b", r"^follow (?:bbc|us on)\b",
    ]
]
DROP_PARAGRAPH = re.compile(
    r"^(?:reporting by\b|editing by\b|additional reporting\b|our standards:\b|"
    r"thomson reuters trust principles\b|all rights reserved\b|"
    r"stream .{0,70}newscasts for free\b|sign up for .{0,100}newsletter\b)", re.I)
BOILERPLATE = re.compile(r"subscribe|newsletter|logged in|post a comment|affiliate|"
                         r"motley fool|community guidelines|publish news of your", re.I)
GEO_EVENT = re.compile(
    r"\b(?:war|wars|inva(?:sion|de[ds]?)|ceasefire|airstrikes?|missiles?|"
    r"sanctions?|embargo(?:es)?|nuclear|military|troops?|annexation|coup|"
    r"hostilities|blockade|peace talks|peace deal|trade war|tariffs?|"
    r"diplomatic tensions|taiwan strait|south china sea)\b", re.I)
GEO_ACTOR = re.compile(
    r"\b(?:united states|u\.?s\.?|america[n]?|russia[n]?|ukrain\w*|china|chinese|"
    r"taiwan|beijing|israel[i]?|iran(?:ian)?|gaza|hamas|nato|european union|"
    r"europe|eu|indonesia[n]?|japan(?:ese)?|korea[n]?|canada|canadian|"
    r"india[n]?|pakistan[i]?|afghan\w*|taliban|iraq[i]?|syria[n]?|"
    r"middle east|venezuela[n]?|britain|british|trump|biden)\b", re.I)
DIRECT_GEO_PHRASE = re.compile(r"trade war|taiwan strait|south china sea|"
                               r"russia.{0,12}ukraine|israel.{0,12}iran|"
                               r"us.{0,5}china.{0,15}tensions", re.I)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "data/processed/usd_idr_jisdor_clean.csv").is_file():
            return p
    raise FileNotFoundError("Run from the project root or a directory beneath it.")


def clean_plain_text(value):
    value = html.unescape(unicodedata.normalize("NFKC", str(value or "")))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[\u200b-\u200d\ufeff]", "", value)
    value = HTML_TAG.sub(" ", value)
    value = re.sub(r"\bU\s*\.\s*S\s*\.", "US", value)
    value = re.sub(r"[ \t]+", " ", value)
    return "\n".join(line.strip() for line in value.splitlines()).strip()


def normalize_key(value):
    return " ".join(re.findall(r"\w+", str(value).casefold()))


def clean_body(value):
    """Conservative paragraph cleaning; never claim to recover an as-of version."""
    text = clean_plain_text(value)
    paragraphs = [x.strip() for x in re.split(r"\n+", text) if x.strip()]
    kept, seen, dropped = [], set(), []
    cut_reason = ""
    for i, paragraph in enumerate(paragraphs):
        if any(pattern.search(paragraph) for pattern in TAIL_BOUNDARIES):
            cut_reason = "explicit_footer_boundary"
            dropped.extend(paragraphs[i:])
            break
        key = normalize_key(paragraph)
        if len(paragraph) >= 80 and key in seen:
            dropped.append(paragraph)
            continue
        if len(paragraph) <= 350 and DROP_PARAGRAPH.search(paragraph):
            dropped.append(paragraph)
            continue
        kept.append(paragraph)
        seen.add(key)
    return "\n\n".join(kept), {
        "removed_paragraphs": len(dropped), "removed_chars": sum(map(len, dropped)),
        "footer_cut_reason": cut_reason,
    }


def static_tokens(text, lowercase=False):
    """No stemming, lemmatization, or stopword removal; preserve negation/entities."""
    text = clean_plain_text(text).replace("’", "'")
    # Preserve negation rather than splitting it into an unrecognizable fragment.
    text = re.sub(r"\bwon't\b", "will not", text, flags=re.I)
    text = re.sub(r"\bcan't\b", "can not", text, flags=re.I)
    text = re.sub(r"n['’]t\b", " not", text, flags=re.I)
    text = text.lower() if lowercase else text
    return TOKEN.findall(text)


def url_date(url):
    m = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|\b)", str(url))
    if not m:
        m = re.search(r"/(20\d{2})(\d{2})(\d{2})/", str(url))
    try:
        return pd.Timestamp(year=int(m[1]), month=int(m[2]), day=int(m[3])) if m else pd.NaT
    except ValueError:
        return pd.NaT


def later_date_mentions(text, article_date):
    hits = []
    for m in FULL_DATE.finditer(text):
        try:
            mentioned = pd.Timestamp(year=int(m[3]), month=list(calendar.month_name).index(m[1].capitalize()), day=int(m[2]))
        except ValueError:
            continue
        if (mentioned - article_date).days > 90:
            hits.append(str(mentioned.date()))
    return sorted(set(hits))


def screen_geopolitics(title, lead):
    """Triage only. Main-text cues are evidence for review, not gold labels."""
    title_hit = bool(GEO_EVENT.search(title))
    lead_hit = bool(GEO_EVENT.search(lead))
    actor_hit = bool(GEO_ACTOR.search(title + " " + lead))
    direct = bool(DIRECT_GEO_PHRASE.search(title))
    if (title_hit and actor_hit) or direct:
        return "candidate", "headline_event_and_actor_or_explicit_interstate_phrase"
    if title_hit or lead_hit:
        return "review", "geopolitical_cue_needs_main_focus_review"
    return "exclude", "no_main_headline_or_lead_geopolitical_cue"


def prepare_news(raw, max_body_words=180, word2vec_lowercase=False, known_future_ids=()):
    missing = set(NEWS_COLUMNS) - set(raw)
    if missing:
        raise ValueError(f"Missing processed-news columns: {sorted(missing)}")
    d = raw.copy().reset_index(drop=True)
    d.insert(0, "source_row", np.arange(len(d)))
    if "Aligned_Date" in d:
        d = d.rename(columns={"Aligned_Date": "legacy_aligned_date"})
    d["source_date"] = pd.to_datetime(d["Date"], errors="coerce")
    if d.source_date.isna().any():
        raise ValueError("Invalid source dates: inspect the processed input before proceeding.")
    if (d.source_date != d.source_date.dt.normalize()).any():
        raise ValueError("Expected date-only Date; use reviewed timestamp fields for publication times.")
    fingerprints = []
    for row in d.itertuples():
        key = json.dumps([row.Date, row.Title, row.Source, row.URL, row.Text_clean], ensure_ascii=False)
        fingerprints.append(hashlib.sha256(key.encode()).hexdigest()[:20])
    d["row_fingerprint"] = fingerprints
    occurrence = d.groupby("row_fingerprint").cumcount()
    d["article_id"] = d.row_fingerprint + occurrence.map(lambda n: f"_{n}")
    d["title_main"] = d.Title.map(clean_plain_text)
    cleaned = d.Text_clean.map(clean_body)
    d["body_main"] = cleaned.map(lambda x: x[0])
    for col in ["removed_paragraphs", "removed_chars", "footer_cut_reason"]:
        d[col] = cleaned.map(lambda x: x[1][col])
    d["original_word_count"] = d.Text_clean.str.split().str.len()
    d["main_word_count"] = d.body_main.str.split().str.len()
    d["lead_main"] = d.body_main.map(lambda x: " ".join(x.split()[:max_body_words]))
    d["text_shared"] = d.title_main + "\n\n" + d.lead_main
    d["text_transformer"] = d.text_shared
    d["tokens_tfidf"] = d.text_shared.map(lambda x: static_tokens(x, lowercase=True))
    d["text_tfidf"] = d.tokens_tfidf.map(" ".join)
    d["tokens_word2vec"] = d.text_shared.map(lambda x: static_tokens(x, lowercase=word2vec_lowercase))
    d["boilerplate_remaining"] = d.body_main.map(lambda x: bool(BOILERPLATE.search(x)))
    d["encoding_flag"] = d.body_main.map(lambda x: bool(re.search(r"â€|â|Ã.|�", x)))
    d["mentioned_later_dates"] = [later_date_mentions(t, dt) for t, dt in zip(d.Text_clean, d.source_date)]
    d["future_date_heuristic"] = d.mentioned_later_dates.map(bool)
    d["known_temporal_contamination"] = d.source_row.isin(known_future_ids)
    d["url_date"] = d.URL.map(url_date)
    d["url_date_gap_days"] = (d.url_date - d.source_date).dt.days
    d["date_mismatch_review"] = d.url_date_gap_days.abs().gt(1)
    d["too_short"] = d.main_word_count.lt(40) | d.title_main.str.split().str.len().lt(3)
    screened = [screen_geopolitics(t, b) for t, b in zip(d.title_main, d.lead_main)]
    d["geo_status"] = [s[0] for s in screened]
    d["geo_reason"] = [s[1] for s in screened]
    # Keep the earliest source record, not the globally longest article/version.
    d["content_key"] = d.body_main.map(normalize_key)
    d["duplicate_of"] = ""
    seen = {}
    for idx in d.sort_values(["source_date", "source_row"]).index:
        key = d.at[idx, "content_key"]
        if len(key) < 80:
            continue
        if key in seen:
            d.at[idx, "duplicate_of"] = seen[key]
        else:
            seen[key] = d.at[idx, "article_id"]
    d["quality_blocked"] = d.known_temporal_contamination | d.too_short | d.duplicate_of.ne("")
    d["quality_review"] = d.date_mismatch_review | d.encoding_flag | d.future_date_heuristic
    for field in REVIEW_COLUMNS[1:]:
        d[field] = ""
    return d


def audit_future_ids(project_root, news_path):
    metrics_path = Path(project_root) / "deira-work/sanity-check/audit_metrics.json"
    evidence_path = metrics_path.with_name("future_date_review.csv")
    if not metrics_path.is_file() or not evidence_path.is_file():
        return [], "No prior audit evidence loaded; date flags remain heuristic."
    metrics = json.loads(metrics_path.read_text())
    expected = metrics.get("input_sha256", {}).get("data/processed/news_clean.csv")
    if sha256(news_path) != expected:
        return [], "Prior audit input hash differs; row-index annotations were NOT applied."
    evidence = pd.read_csv(evidence_path)
    ids = evidence.loc[evidence.status.eq("confirmed_later_related_content"), "article_id"].astype(int).tolist()
    return ids, f"Applied {len(ids)} confirmed temporal flags after an exact input-hash match."


def truth(value):
    value = str(value).strip().lower()
    if value not in {"", "true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"Invalid boolean review value: {value!r}")
    return value in {"true", "1", "yes"}


def aware_time(value, source_timezone=""):
    if not str(value).strip():
        return pd.NaT
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        if not str(source_timezone).strip():
            raise ValueError("Naive publication timestamp requires an explicit source_timezone.")
        ts = ts.tz_localize(source_timezone, ambiguous="raise", nonexistent="raise")
    return ts.tz_convert(WIB)


def apply_reviews(news, review_path=None):
    d = news.copy()
    if review_path is not None and Path(review_path).is_file():
        reviews = pd.read_csv(review_path, keep_default_na=False)
        if "article_id" not in reviews or reviews.article_id.duplicated().any():
            raise ValueError("Reviews require unique article_id values.")
        unknown = set(reviews.article_id) - set(d.article_id)
        if unknown:
            raise ValueError("Review IDs do not match this input snapshot; regenerate/reconcile the review file.")
        reviews = reviews.set_index("article_id")
        for field in REVIEW_COLUMNS[1:]:
            if field in reviews:
                d[field] = d.article_id.map(reviews[field]).fillna("")
    allowed = {"", "include", "exclude", "review"}
    d["geo_decision"] = d.geo_decision.str.strip().str.lower()
    if not set(d.geo_decision).issubset(allowed):
        raise ValueError("geo_decision must be blank, include, exclude, or review.")
    if (d.geo_decision.ne("") & d.review_note.str.strip().eq("")).any():
        raise ValueError("Every manual geopolitical decision needs a review_note.")
    d["effective_geo_status"] = [
        original if decision == "" else "approved" if decision == "include" else decision
        for decision, original in zip(d.geo_decision, d.geo_status)
    ]
    published, available = [], []
    for row in d.itertuples():
        pub = aware_time(row.published_at, row.source_timezone)
        avail = aware_time(row.available_at, row.source_timezone)
        if pd.isna(avail):
            avail = pub
        if pd.notna(pub) and pd.notna(avail) and avail < pub:
            raise ValueError(f"available_at precedes published_at for {row.article_id}")
        published.append(pub)
        available.append(avail)
    d["published_at_wib"] = pd.Series(published, index=d.index, dtype=f"datetime64[ns, {WIB}]")
    d["available_at_wib"] = pd.Series(available, index=d.index, dtype=f"datetime64[ns, {WIB}]")
    d["timestamp_verified"] = d.timestamp_verified.map(truth)
    d["body_asof_verified"] = d.body_asof_verified.map(truth)
    declared = d.timestamp_verified | d.body_asof_verified
    if (declared & (d.available_at_wib.isna() | d.evidence.str.strip().eq(""))).any():
        raise ValueError("Verified timing/body-version declarations require available_at and evidence.")
    return d


def choose_candidates(news, geopolitical_policy="audit_only", include_rule_review=False):
    d = news.copy()
    if geopolitical_policy == "audit_only":
        # Do not redefine the research corpus with an unvalidated keyword rule.
        geo_ok = ~d.geo_decision.isin(["exclude", "review"])
    elif geopolitical_policy == "rule_screened":
        geo_ok = d.effective_geo_status.isin(["approved", "candidate"] + (["review"] if include_rule_review else []))
    elif geopolitical_policy == "human_approved":
        geo_ok = d.effective_geo_status.eq("approved")
    else:
        raise ValueError("Choose audit_only, rule_screened, or human_approved geopolitical_policy.")
    # Explicit human approval may resolve heuristic quality flags, but not known
    # temporal contamination, unusable content, or exact duplicate records.
    review_ok = ~d.quality_review | d.effective_geo_status.eq("approved")
    d["selected_for_alignment"] = geo_ok & review_ok & ~d.quality_blocked
    d["selection_reason"] = np.select(
        [d.duplicate_of.ne(""), d.known_temporal_contamination, d.too_short,
         d.quality_review & ~d.effective_geo_status.eq("approved"), ~geo_ok],
        ["exact_content_duplicate", "known_later_content_quarantine", "insufficient_main_text",
         "quality_review_required", "geopolitical_policy_not_accepted"], default="quality_pass_or_manually_approved")
    return d


def build_calendar(raw):
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
    assert not set(MARKET_FEATURES) & set(TARGET_COLUMNS)
    return d


def read_verified_schedule(path, calendar_frame):
    schedule = pd.read_csv(path, keep_default_na=False)
    required = {"Date", "published_at_wib", "decision_at_wib", "verified", "source"}
    if not required.issubset(schedule):
        raise ValueError(f"Strict schedule needs {sorted(required)}")
    schedule["Date"] = pd.to_datetime(schedule.Date, errors="raise")
    if schedule.Date.duplicated().any() or set(schedule.Date) != set(calendar_frame.Date):
        raise ValueError("Verified schedule must cover each master-calendar date exactly once.")
    schedule = schedule.set_index("Date").loc[calendar_frame.Date].reset_index()
    if not schedule.verified.map(truth).all() or schedule.source.str.strip().eq("").any():
        raise ValueError("Each schedule row must be explicitly verified and have a source.")
    for col in ["published_at_wib", "decision_at_wib"]:
        schedule[col] = pd.Series([aware_time(t, WIB) for t in schedule[col]], dtype=f"datetime64[ns, {WIB}]")
        if schedule[col].isna().any():
            raise ValueError("Strict schedule times must be populated.")
        if not (schedule[col].dt.tz_localize(None).dt.normalize().to_numpy() == schedule.Date.to_numpy()).all():
            raise ValueError("Schedule timestamps must fall on their stated WIB calendar date.")
    if (schedule.decision_at_wib < schedule.published_at_wib).any() or not schedule.decision_at_wib.is_monotonic_increasing:
        raise ValueError("Decision cutoffs must be ordered and at/after JISDOR availability.")
    return schedule


def align_articles(news, calendar_frame, mode="date_delay_sensitivity", schedule=None):
    d = news.copy()
    d["timing_mode"] = mode
    d["anchor_date"] = pd.NaT
    d["alignment_status"] = "not_selected"
    selected = d.selected_for_alignment
    if mode == "date_delay_sensitivity":
        # Strictly later fixing DATE; no publication hour or zone is fabricated.
        ix = np.searchsorted(calendar_frame.Date.to_numpy(), d.source_date.to_numpy(), side="right")
        valid = selected & (ix < len(calendar_frame))
        d.loc[valid, "anchor_date"] = calendar_frame.Date.to_numpy()[ix[valid]]
        d.loc[valid, "alignment_status"] = "date_only_delayed_assumption"
        d.loc[selected & ~valid, "alignment_status"] = "after_last_available_fixing"
    elif mode == "strict_forecast":
        if schedule is None:
            raise ValueError("Strict forecasting requires a verified per-date availability schedule.")
        if len(schedule) != len(calendar_frame) or not np.array_equal(schedule.Date.to_numpy(), calendar_frame.Date.to_numpy()):
            raise ValueError("Schedule and master calendar must be aligned and equally long.")
        trusted = selected & d.timestamp_verified & d.body_asof_verified & d.available_at_wib.notna()
        d.loc[selected & ~trusted, "alignment_status"] = "unverified_news_availability_or_body_version"
        if trusted.any():
            ix = np.searchsorted(schedule.decision_at_wib.astype("int64").to_numpy(), d.loc[trusted, "available_at_wib"].astype("int64").to_numpy(), side="left")
            indices = d.index[trusted]
            valid_indices = indices[ix < len(calendar_frame)]
            d.loc[valid_indices, "anchor_date"] = calendar_frame.Date.to_numpy()[ix[ix < len(calendar_frame)]]
            d.loc[valid_indices, "alignment_status"] = "verified_availability_window"
            d.loc[indices[ix >= len(calendar_frame)], "alignment_status"] = "after_last_available_fixing"
    else:
        raise ValueError("Choose date_delay_sensitivity or strict_forecast; explanatory labels require a separate protocol.")
    return d


def build_daily_tables(calendar_frame, aligned):
    usable = aligned.loc[aligned.anchor_date.notna()].copy()
    usable = usable.sort_values(["anchor_date", "source_date", "article_id"])
    grouped = {date: g for date, g in usable.groupby("anchor_date", sort=True)}
    windows = []
    for date in calendar_frame.Date:
        g = grouped.get(date)
        windows.append({"Date": date, "article_ids": [] if g is None else g.article_id.tolist(),
                        "texts_transformer": [] if g is None else g.text_transformer.tolist(),
                        "texts_tfidf": [] if g is None else g.text_tfidf.tolist(),
                        "tokens_word2vec": [] if g is None else g.tokens_word2vec.tolist(),
                        "n_berita": 0 if g is None else len(g)})
    windows = pd.DataFrame(windows)
    windows["has_news"] = windows.n_berita.gt(0).astype(int)
    daily = calendar_frame.merge(windows[["Date", "n_berita", "has_news"]], on="Date", validate="one_to_one")
    return daily, windows


def temporal_manifests(daily, test_fraction=.2, n_folds=3):
    n = len(daily)
    test_start = int(n * (1-test_fraction))
    dev_end = test_start-1  # boundary row excluded from fitting/selection
    initial = dev_end // 2
    bounds = np.linspace(initial, dev_end, n_folds+1, dtype=int)
    if initial < 25 or len(set(bounds)) != n_folds+1:
        raise ValueError("Not enough fixing rows for the configured temporal folds.")
    partitions = daily[["calendar_index", "Date", "target_date", "y", "eligible_binary", "n_berita", "has_news"]].copy()
    partitions["partition"] = np.select([partitions.calendar_index < dev_end, partitions.calendar_index >= test_start], ["development", "test"], default="boundary_purge")
    folds, summaries = [], []
    for fold, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        for role, mask in [
            ("train", daily.calendar_index.lt(start-1) & daily.eligible_binary),
            ("validation", daily.calendar_index.ge(start) & daily.calendar_index.lt(end-1) & daily.eligible_binary),
        ]:
            rows = daily.loc[mask]
            for row in rows.itertuples():
                folds.append({"fold": fold, "role": role, "Date": row.Date, "target_date": row.target_date,
                              "y": int(row.y), "n_berita": int(row.n_berita)})
            summaries.append({"fold": fold, "role": role, "rows": len(rows),
                              "first_date": rows.Date.min(), "last_date": rows.Date.max(),
                              "up": int(rows.y.eq(1).sum()), "down": int(rows.y.eq(0).sum()),
                              "days_with_news": int(rows.has_news.sum())})
    folds = pd.DataFrame(folds)
    for fold in range(1, n_folds+1):
        tr = folds[(folds.fold == fold) & folds.role.eq("train")]
        va = folds[(folds.fold == fold) & folds.role.eq("validation")]
        assert tr.target_date.max() < va.Date.min()
    final = partitions.loc[partitions.partition.ne("boundary_purge") & partitions.eligible_binary].copy()
    final["role"] = final.partition.map({"development": "train", "test": "test"})
    assert final.loc[final.role.eq("train"), "target_date"].max() < final.loc[final.role.eq("test"), "Date"].min()
    return partitions, folds, pd.DataFrame(summaries), final


def write_jsonl(frame, path):
    def convert(value):
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return value
    with Path(path).open("w", encoding="utf-8") as f:
        for row in frame.to_dict("records"):
            f.write(json.dumps(convert(row), ensure_ascii=False, allow_nan=False)+"\n")
