"""Reproducible exploratory audit; reads only the two processed CSV inputs.

Run with Python + numpy + pandas. No notebook, network, training labels, or
original CSV is modified. Qualitative annotations are a separate reviewed input.
"""
from pathlib import Path
import calendar
import hashlib
import json
import random
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 20261110
REVIEWED_NEWS_SHA256 = "81ff1e273e49d678fc86f0377c78b08be91004a3a248e660d58d6ec73176c4b1"


def contains(text, pattern):
    return bool(re.search(pattern, text, re.I))


def usd_direction(text):
    # Deliberately transparent, imperfect lexical probe, not sentiment ground truth.
    t = re.sub(r"\s+", " ", text.lower())
    usd = r"(?:u\.?s\.? dollar|dollar|greenback|usd)"
    up = r"(?:strengthen\w*|stronger|surge\w*|gain\w*|rise\w*|rose|rally|rallies|firm\w*|climb\w*|ascendant)"
    down = r"(?:weaken\w*|weaker|fall\w*|fell|drop\w*|slid\w*|slide\w*|retreat\w*|declin\w*|sink\w*|sank)"
    return len(re.findall(usd+r"\W+(?:\w+\W+){0,3}"+up, t)) - len(re.findall(usd+r"\W+(?:\w+\W+){0,3}"+down, t))


def policy_tilt(text):
    if not contains(text, r"federal reserve|\bfed\b|powell|fomc"):
        return 0
    return len(re.findall(r"hawkish|rate hikes?|higher for longer|tighten\w*", text, re.I)) - len(re.findall(r"dovish|rate cuts?|eas(?:e|es|ing)", text, re.I))


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(q, 1)
    return result


def corr_tests(daily):
    """Calendar-aware circular-shift null; news-day selection stays on calendar.

    Target vector is restricted to dates with known outcomes before circular shifts.
    Exclude shifts within 20 trading rows of original alignment. All remaining
    offsets are enumerated (not random Monte Carlo draws). Nonstationarity remains
    a limitation; this is an exploratory diagnostic, not a causal significance test.
    """
    features = [c for c in daily if c.startswith("x_")]
    rows = []
    for h in [0, 1, 3, 5]:
        for kind in ["signed", "absolute"]:
            key = "return_same_pct" if h == 0 else f"return_forward_{h}_pct"
            valid = daily[key].notna()
            frame = daily.loc[valid].reset_index(drop=True)
            y = frame[key].to_numpy(float)
            if kind == "absolute":
                y = abs(y)
            n = len(y)
            offsets = np.arange(21, n-20)
            shifted = y[(np.arange(n)[None, :] - offsets[:, None]) % n]
            for cohort in ["all_trading_days", "observed_news_days"]:
                mask = np.ones(n, bool) if cohort == "all_trading_days" else frame["x_news_count"].to_numpy() > 0
                obs_y = y[mask]
                yc = obs_y-obs_y.mean()
                null_y = shifted[:, mask]
                null_y = null_y-null_y.mean(axis=1, keepdims=True)
                null_norm = np.linalg.norm(null_y, axis=1)
                for feature in features:
                    x = frame.loc[mask, feature].to_numpy(float)
                    xc = x-x.mean()
                    norm = np.linalg.norm(xc)
                    if norm == 0 or np.linalg.norm(yc) == 0:
                        continue
                    r = np.dot(xc, yc)/(norm*np.linalg.norm(yc))
                    null_r = (null_y @ xc) / (null_norm*norm)
                    p = (1+np.sum(abs(null_r) >= abs(r)))/(len(null_r)+1)
                    # Year-demeaned sensitivity checks shared time-regime effects.
                    years = frame.loc[mask, "Date"].dt.year.to_numpy()
                    xd, yd = x.copy(), obs_y.copy()
                    for year in np.unique(years):
                        ii = years == year
                        xd[ii] -= xd[ii].mean()
                        yd[ii] -= yd[ii].mean()
                    rd = np.corrcoef(xd, yd)[0, 1] if xd.std() and yd.std() else np.nan
                    rows.append(dict(horizon_trading_days=h, target_kind=kind, cohort=cohort,
                                     feature=feature, n_days=int(mask.sum()), nonzero_feature_days=int((x != 0).sum()),
                                     pearson_r=r, circular_shift_p=p, year_demeaned_r=rd))
    result = pd.DataFrame(rows)
    result["bh_q"] = np.nan
    # Separate declared families: descriptive contemporaneous vs future horizons.
    for descriptive in [True, False]:
        m = (result.horizon_trading_days == 0) == descriptive
        result.loc[m, "bh_q"] = bh(result.loc[m, "circular_shift_p"])
    result.to_csv(OUT / "correlation_tests.csv", index=False)
    return result


def sensitivities(daily, news):
    """Effect-size diagnostics only: no new significance claims or model selection."""
    confirmed_future = {1, 28, 38, 48, 78, 108, 121, 190, 192, 212}
    exact_duplicates = {105}
    bad_date = set(news.loc[news.url_minus_record_days.abs() > 1, "article_id"])
    remove = confirmed_future | exact_duplicates | bad_date
    filtered = news[~news.article_id.isin(remove)].copy()
    mapping = {"x_news_count": ("article_id", "size"), "x_domestic_count": ("domestic_lead", "sum"),
               "x_macro_count": ("macro_lead", "sum"), "x_usd_title": ("usd_title", "mean"),
               "x_usd_lead": ("usd_lead", "mean"), "x_usd_full": ("usd_full", "mean"), "x_fed_tilt": ("fed_tilt", "mean")}
    clean_agg = filtered.groupby("Aligned_Date").agg(**mapping)
    core = list(mapping)
    clean_daily = daily.drop(columns=core).merge(clean_agg, left_on="Date", right_index=True, how="left")
    clean_daily[core] = clean_daily[core].fillna(0)
    scenarios = [("original", daily, 0), ("news_delayed_one_trading_day", daily, 1),
                 ("screened_421_articles", clean_daily, 0),
                 ("screened_and_delayed", clean_daily, 1),
                 ("original_2021_2023", daily[daily.Date.dt.year <= 2023], 0),
                 ("original_2024_2026", daily[daily.Date.dt.year >= 2024], 0)]
    rows = []
    for name, frame, delay in scenarios:
        for feature in core:
            d = pd.DataFrame({"x": frame[feature].shift(delay), "y": frame.return_forward_1_pct}).dropna()
            rows.append(dict(scenario=name, feature=feature, n_days=len(d),
                             nonzero_feature_days=int((d.x != 0).sum()),
                             pearson_r=d.x.corr(d.y), spearman_r=d.x.rank().corr(d.y.rank())))
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_checks.csv", index=False)
    return sorted(remove)


def bootstrap_selected(daily):
    rng = np.random.default_rng(SEED)
    results = []
    for feature in ["x_news_count", "x_domestic_count", "x_macro_count", "x_usd_title", "x_usd_lead", "x_fed_tilt"]:
        d = daily.dropna(subset=["return_forward_1_pct"])
        x = d[feature].to_numpy(float)
        y = d.return_forward_1_pct.to_numpy(float)
        n = len(x)
        boot = []
        for _ in range(2000):
            starts = rng.integers(0, n, size=(n+9)//10)
            idx = ((starts[:, None]+np.arange(10)) % n).ravel()[:n]
            if x[idx].std() and y[idx].std():
                boot.append(np.corrcoef(x[idx], y[idx])[0, 1])
        results.append(dict(feature=feature, horizon_trading_days=1, n_days=n,
                            pearson_r=np.corrcoef(x, y)[0, 1],
                            block_bootstrap_ci_low=np.quantile(boot, .025),
                            block_bootstrap_ci_high=np.quantile(boot, .975),
                            block_length=10, bootstrap_draws=2000))
    result = pd.DataFrame(results)
    result.to_csv(OUT / "correlation_intervals.csv", index=False)
    return result


def main():
    paths = [ROOT / "data/processed/news_clean.csv", ROOT / "data/processed/usd_idr_jisdor_clean.csv"]
    assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == REVIEWED_NEWS_SHA256, (
        "News input changed: review the sample and qualitative annotations before rerunning."
    )
    news = pd.read_csv(paths[0], keep_default_na=False)
    fx = pd.read_csv(paths[1], parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    assert fx.Date.is_unique and (fx.USD_IDR > 0).all()
    news.insert(0, "article_id", np.arange(len(news)))  # zero-based data-row index, not CSV physical line
    news["Date"] = pd.to_datetime(news.Date)
    news["Aligned_Date"] = pd.to_datetime(news.Aligned_Date)
    assert news.Aligned_Date.isin(fx.Date).all()
    news["lead_200_words"] = news.Text_clean.map(lambda t: " ".join(t.split()[:200]))
    news["word_count"] = news.Text_clean.str.split().str.len()
    news["domestic_lead"] = news.lead_200_words.map(lambda t: contains(t, r"\brupiah\b|bank indonesia|usd\s*[/ -]\s*idr"))
    news["domestic_full"] = news.Text_clean.map(lambda t: contains(t, r"\brupiah\b|bank indonesia|usd\s*[/ -]\s*idr"))
    macro = r"federal reserve|\bfed\b|\bfomc\b|treasury yields?|\bdollar\b|\bgreenback\b|bank of japan|european central bank|bank indonesia|\brupiah\b"
    news["macro_lead"] = (news.Title + " " + news.lead_200_words).map(lambda t: contains(t, macro))
    news["usd_title"] = news.Title.map(usd_direction)
    news["usd_lead"] = news.lead_200_words.map(usd_direction)
    news["usd_full"] = news.Text_clean.map(usd_direction)
    news["fed_tilt"] = news.lead_200_words.map(policy_tilt)
    boiler = r"subscribe|sign up|logged in|sign in|post a comment|comment policy|community guidelines|publish news of your|news release|sponsored|affiliate|motley fool|all rights reserved"
    news["boilerplate_flag"] = news.Text_clean.map(lambda t: contains(t, boiler))
    news["encoding_flag"] = news.Text_clean.map(lambda t: contains(t, r"â€|â|Ã.|�"))

    def repeats(t):
        ps = [re.sub(r"\s+", " ", p).strip().lower() for p in t.split("\n") if len(p.strip()) >= 80]
        return sum(len(p) for p in ps) - sum(len(p) for p in set(ps))
    news["repeated_paragraph_chars"] = news.Text_clean.map(repeats)
    news["repeated_paragraph_fraction"] = news.repeated_paragraph_chars / news.Text_clean.str.len()
    news["alignment_shift_calendar_days"] = (news.Aligned_Date-news.Date).dt.days
    expected = fx.Date.to_numpy()[np.searchsorted(fx.Date.to_numpy(), news.Date.to_numpy())]
    news["alignment_correct_ge_date"] = news.Aligned_Date.to_numpy() == expected

    url_dates = []
    future_dates = []
    date_pattern = re.compile(r"\b("+"|".join(calendar.month_name[1:])+r")\s+(\d{1,2}),?\s+(20\d{2})\b", re.I)
    for r in news.itertuples():
        m = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|\b)", r.URL)
        if not m:
            m = re.search(r"/(20\d{2})(\d{2})(\d{2})/", r.URL)
        try:
            date = pd.Timestamp(year=int(m[1]), month=int(m[2]), day=int(m[3])) if m else pd.NaT
        except ValueError:
            date = pd.NaT
        url_dates.append(date)
        for dm in date_pattern.finditer(r.Text_clean):
            try:
                dt = pd.Timestamp(year=int(dm[3]), month=list(calendar.month_name).index(dm[1].capitalize()), day=int(dm[2]))
            except ValueError:
                continue
            if (dt-r.Date).days > 90:
                future_dates.append(dict(article_id=r.article_id, article_date=str(r.Date.date()), mentioned_date=str(dt.date()),
                                         gap_days=(dt-r.Date).days, character_position=dm.start(),
                                         snippet=r.Text_clean[max(0, dm.start()-100):dm.end()+170].replace("\n", " "),
                                         interpretation="heuristic flag; could be forecast, related content, or update; inspect context"))
    news["url_date"] = url_dates
    news["url_minus_record_days"] = (news.url_date-news.Date).dt.days
    evidence = pd.DataFrame(future_dates)
    evidence.to_csv(OUT / "future_date_flags.csv", index=False)
    future_reviews = []
    for i in sorted(evidence.article_id.unique()):
        status = "legitimate_forward_date" if i in {63, 128} else "confirmed_later_related_content"
        reason = {63: "Tanggal berlakunya penurunan tarif IT Juli 2022 dibicarakan prospektif pada 2021; bukan bukti leakage.",
                  128: "Jadwal penghentian LIBOR Juni 2023 dibicarakan pada 2022; bukan bukti leakage."}.get(i, "Cuplikan berita/kartu rekomendasi lain bertanggal sesudah tanggal artikel; konteks diperiksa pada future_date_flags.csv.")
        future_reviews.append(dict(article_id=int(i), status=status, reason=reason))
    pd.DataFrame(future_reviews).to_csv(OUT / "future_date_review.csv", index=False)
    news["future_date_over90_flag"] = news.article_id.isin(evidence.article_id if len(evidence) else [])

    # Residual duplication screen: title word-set Jaccard >= .70 or exact normalized text.
    title_sets = [set(re.findall(r"[a-z0-9]+", t.lower())) for t in news.Title]
    pairs = []
    normtext = news.Text_clean.map(lambda t: re.sub(r"\W+", "", t.lower())).tolist()
    for i in range(len(news)):
        for j in range(i+1, len(news)):
            sim = len(title_sets[i] & title_sets[j])/max(1, len(title_sets[i] | title_sets[j]))
            exact = normtext[i] == normtext[j]
            if sim >= .70 or exact:
                pairs.append(dict(article_id_a=i, article_id_b=j, title_jaccard=sim,
                                  exact_normalized_text=exact, date_a=str(news.Date.iloc[i].date()),
                                  date_b=str(news.Date.iloc[j].date()), title_a=news.Title.iloc[i], title_b=news.Title.iloc[j]))
    pd.DataFrame(pairs).to_csv(OUT / "duplicate_candidates.csv", index=False)

    rng = random.Random(SEED)
    sample_ids = []
    for event, group in news.groupby("Event", sort=True):
        sample_ids.extend(rng.sample(group.article_id.tolist(), 6))
    sample = news[news.article_id.isin(sample_ids)].copy()
    annotations = json.loads((OUT / "manual_annotations.json").read_text())
    assert set(map(int, annotations)) == set(sample_ids), "Annotations must match deterministic sampling exactly"
    for c in ["relevance", "event_match", "quality_notes", "reason"]:
        sample[c] = sample.article_id.map(lambda i: annotations[str(i)][c])
    sample["review_basis"] = "Title + first 1400-1700 characters + final 250 characters; additional context for flagged cases. Single AI reviewer; not independent human ground truth."
    sample["inclusion_probability"] = sample.Event.map((6/news.Event.value_counts()).to_dict())
    sample["sampling_weight"] = 1/sample.inclusion_probability
    sample["seed"] = SEED
    # Include full stored text so a second reviewer can inspect beyond excerpts.
    sample.to_csv(OUT / "sample_review_66.csv", index=False)

    daily = fx.copy()
    daily["return_same_pct"] = 100*daily.USD_IDR.pct_change()
    for h in [1, 3, 5]:
        daily[f"return_forward_{h}_pct"] = 100*(daily.USD_IDR.shift(-h)/daily.USD_IDR-1)
    agg = news.groupby("Aligned_Date").agg(
        x_news_count=("article_id", "size"), x_domestic_count=("domestic_lead", "sum"),
        x_macro_count=("macro_lead", "sum"), x_usd_title=("usd_title", "mean"),
        x_usd_lead=("usd_lead", "mean"), x_usd_full=("usd_full", "mean"), x_fed_tilt=("fed_tilt", "mean"))
    events = pd.crosstab(news.Aligned_Date, news.Event)
    event_map = {event: f"x_event_{i:02d}" for i, event in enumerate(events.columns)}
    events = events.rename(columns=event_map)
    (OUT / "feature_dictionary.json").write_text(json.dumps({"event_columns": event_map,
        "definitions": {"x_news_count": "observed articles per trading day; zero means no collected article",
                        "x_domestic_count": "lead first 200 words mentions rupiah, Bank Indonesia, or USD/IDR",
                        "x_macro_count": "title + lead mentions predefined monetary/FX anchors (not validated relevance)",
                        "x_usd_title": "mean lexical USD direction in title; + strengthening, - weakening",
                        "x_usd_lead": "mean lexical USD direction in first 200 words",
                        "x_usd_full": "mean lexical USD direction across stored full text (contamination sensitivity)",
                        "x_fed_tilt": "mean hawkish-minus-dovish lexical count, gated on Fed mention, first 200 words"}}, indent=2))
    daily = daily.merge(agg, left_on="Date", right_index=True, how="left").merge(events, left_on="Date", right_index=True, how="left")
    feature_cols = [c for c in daily if c.startswith("x_")]
    daily[feature_cols] = daily[feature_cols].fillna(0)
    daily.to_csv(OUT / "daily_features_and_returns.csv", index=False)
    sample = sample.merge(daily[["Date", "USD_IDR", "return_same_pct", "return_forward_1_pct", "return_forward_3_pct", "return_forward_5_pct"]],
                          left_on="Aligned_Date", right_on="Date", how="left", suffixes=("", "_fx"))
    sample.drop(columns="Date_fx").to_csv(OUT / "sample_review_66.csv", index=False)
    tests = corr_tests(daily)
    inv_events = {v: k for k, v in event_map.items()}
    tests["event_name"] = tests.feature.map(inv_events).fillna("")
    tests.to_csv(OUT / "correlation_tests.csv", index=False)
    intervals = bootstrap_selected(daily)
    screened_ids = sensitivities(daily, news)

    byyear = news.groupby(news.Date.dt.year).agg(articles=("article_id", "size"), news_trading_days=("Aligned_Date", "nunique"), domestic_lead=("domestic_lead", "sum"))
    byyear["fx_days"] = fx.groupby(fx.Date.dt.year).size()
    byyear["coverage_pct"] = 100*byyear.news_trading_days/byyear.fx_days
    byyear.to_csv(OUT / "coverage_by_year.csv")
    pd.crosstab(news.Event, news.Date.dt.year).to_csv(OUT / "event_by_year.csv")
    monthly = news.groupby(news.Date.dt.to_period("M")).size().reindex(pd.period_range(news.Date.min(), news.Date.max(), freq="M"), fill_value=0)
    monthly.rename("articles").to_csv(OUT / "monthly_coverage.csv", index_label="month")
    full_audit_cols = [c for c in news if c not in ["Text_clean", "lead_200_words"]]
    news[full_audit_cols].to_csv(OUT / "article_quality_flags.csv", index=False)
    review_summary = []
    for variable in ["relevance", "event_match"]:
        for label in sorted(sample[variable].unique()):
            s = sample[sample[variable] == label]
            review_summary.append(dict(variable=variable, label=label, sample_count=len(s),
                                       sample_pct=100*len(s)/len(sample),
                                       weighted_population_estimate_pct=100*s.sampling_weight.sum()/len(news)))
    pd.DataFrame(review_summary).to_csv(OUT / "sample_summary.csv", index=False)
    stats = dict(seed=SEED, input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                 articles=len(news), fx_days=len(fx), first_date=str(fx.Date.min().date()), last_date=str(fx.Date.max().date()),
                 covered_trading_days=news.Aligned_Date.nunique(), coverage_pct=100*news.Aligned_Date.nunique()/len(fx),
                 zero_article_months=int((monthly == 0).sum()), total_months=len(monthly),
                 busiest_day_articles=int(news.groupby("Aligned_Date").size().max()),
                 cnbc_articles=int((news.Source == "cnbc.com").sum()), unique_sources=news.Source.nunique(),
                 words_quantiles={str(q): float(news.word_count.quantile(q)) for q in [0, .25, .5, .75, 1]},
                 flags={c: int(news[c].sum()) for c in ["domestic_lead", "domestic_full", "macro_lead", "boilerplate_flag", "encoding_flag", "future_date_over90_flag"]},
                 repeated_paragraph_articles=int((news.repeated_paragraph_chars > 0).sum()),
                 repeated_paragraph_over20pct=int((news.repeated_paragraph_fraction > .2).sum()),
                 confirmed_later_related_content=10, screened_out_article_ids=screened_ids,
                 url_dates_available=int(news.url_date.notna().sum()),
                 url_date_mismatch_any=int((news.url_minus_record_days.fillna(0) != 0).sum()),
                 url_date_mismatch_over1day=int((news.url_minus_record_days.fillna(0).abs() > 1).sum()),
                 alignment_mismatch=int((~news.alignment_correct_ge_date).sum()),
                 shifted_to_later_trading_day=int((news.alignment_shift_calendar_days > 0).sum()),
                 duplicate_candidate_pairs=len(pairs), sample_size=len(sample), sample_ids=sorted(sample_ids),
                 exploratory_future_tests=int((tests.horizon_trading_days > 0).sum()),
                 fx_weekend_rows=int((fx.Date.dt.dayofweek > 4).sum()), fx_duplicate_dates=int(fx.Date.duplicated().sum()),
                 fx_zero_return_days=int((daily.return_same_pct == 0).sum()),
                 future_tests_q_below_005=int(((tests.horizon_trading_days > 0) & (tests.bh_q < .05)).sum()),
                 same_day_tests_q_below_005=int(((tests.horizon_trading_days == 0) & (tests.bh_q < .05)).sum()))
    (OUT / "audit_metrics.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("\nSAMPLE SUMMARY\n", pd.DataFrame(review_summary).to_string(index=False))
    print("\nNEXT-DAY INTERVALS\n", intervals.to_string(index=False))
    print("\nSTRONGEST FUTURE CANDIDATES\n", tests[tests.horizon_trading_days > 0].sort_values("circular_shift_p").head(15).to_string(index=False))


if __name__ == "__main__":
    main()
