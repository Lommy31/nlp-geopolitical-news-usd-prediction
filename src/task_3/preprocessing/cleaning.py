import calendar
import hashlib
import html
import json
import re
import unicodedata

import numpy as np
import pandas as pd

ORIGINAL_SOURCE_SHA256 = "81ff1e273e49d678fc86f0377c78b08be91004a3a248e660d58d6ec73176c4b1"
QUARANTINED_ARTICLE_IDS = {
    "6771dcf4c59508bb70b1_0", "96b49d2c7e13508b3d93_0", "6c85372d11541f236d86_0",
    "12a3f1215a3512cc1501_0", "d807bbfe1b00e188dba0_0", "525c68238dd09c71cc0e_0",
    "0f9f0a7c4020acc88c57_0", "2c753ace2e314dab13af_0", "b5f636597d7a698dacfc_0",
    "30eb28a4572d1328b51b_0",
}
NEWS_COLUMNS = ["Date", "Title", "Source", "URL", "Event", "Text_clean"]

BLOCKED_DOMAINS = {
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "accesswire.com",
    "einpresswire.com",
    "fool.com",
    "fool.co.uk",
    "insidermonkey.com",
    "investorplace.com",
}

SINGLE_COMPANY_MARKERS = re.compile(r"\((?:NYSE|NASDAQ):|earnings call transcript", re.IGNORECASE)

WIRE_SERVICE_MARKERS = ("/prnewswire/", "/business wire/", "globe newswire", "accesswire")

LOREM_IPSUM_MARKERS = ("pellentesque", "ullamcorper", "consectetur", "adipiscing")

ANCHOR_TERMS = {
    "US Fiscal Stimulus + Rising Treasury Yields": ("treasury yield", "fiscal stimulus", "federal reserve", "inflation"),
    "Federal Reserve Monetary Policy": ("federal reserve", "the fed", "jerome powell", "interest rate"),
    "US-China / Taiwan Tensions": ("us-china", "china-us", "taiwan strait", "south china sea", "cross-strait", "beijing"),
    "Banking & Financial Crisis": ("banking crisis", "bank collapse", "bank failure", "silicon valley bank", "credit suisse", "banking sector"),
    "US Presidential Election": ("presidential election", "us election", "white house", "trump", "biden"),
    "Trump Tariffs / Global Trade War": ("trade war", "trump tariff", "china tariff", "import tariff", "trade tariff"),
    "Indonesian Rupiah / Bank Indonesia Policy": ("indonesia", "rupiah", "jakarta"),
    "Russia-Ukraine War & Sanctions": ("russia", "ukraine", "moscow", "kremlin", "sanctions"),
    "Middle East Conflict & Oil Shock": ("israel", "gaza", "iran", "middle east", "oil price"),
    "BOJ / ECB Policy Shifts": ("bank of japan", "european central bank", "ecb"),
    "Emerging Market Currency Contagion": ("emerging market", "currency crisis", "sovereign default"),
}

BOILERPLATE_LINES = (
    "reporting by",
    "editing by",
    "additional reporting",
    "compiled by",
    "our standards:",
    "thomson reuters trust principles",
)

PAYWALL_MARKERS = (
    "you are able to gift",
    "anyone can access the link you share",
    "subscription, you can gift",
    "you do not have any active subscriptions",
    "go to the subscriptions page",
    "to continue reading",
    "create a free account",
    "already a subscriber",
)

def normalize_title(title):
    title = re.sub(r"^[^|]{1,20}\|\s*", "", title)
    title = unicodedata.normalize("NFKC", title).lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return " ".join(title.split()[:6])

def is_placeholder_content(text):
    text_lower = text.lower()
    return any(m in text_lower for m in LOREM_IPSUM_MARKERS)

def fix_spaced_punctuation(text):
    text = re.sub(r"\b([A-Z])\s+\.(?=\s+[A-Z]\s+\.|\s|$)", r"\1.", text)
    text = re.sub(r"(\d)\s+\.\s+(\d)", r"\1.\2", text)
    text = re.sub(r"(?<=[A-Z]\.)\s+(?=[A-Z]\.)", "", text)
    return re.sub(r"\s+([.,;:!?%])", r"\1", text)

def clean_scraped_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    junk = BOILERPLATE_LINES + PAYWALL_MARKERS
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line and not any(m in line.lower() for m in junk)]
    text = "\n\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return fix_spaced_punctuation(text)

def is_single_company_content(text):
    return bool(SINGLE_COMPANY_MARKERS.search(text))

def is_wire_service_mirror(text):
    text_lower = text.lower()
    return any(m in text_lower for m in WIRE_SERVICE_MARKERS)

def has_anchor(event, text):
    terms = ANCHOR_TERMS.get(event)
    if not terms:
        return True
    text_lower = text.lower()
    return any(t in text_lower for t in terms)

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
    text = clean_plain_text(text).replace("’", "'")

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

def prepare_articles(raw, prefix="", source_offset=0, max_body_words=180):
    missing = set(NEWS_COLUMNS) - set(raw)
    if missing:
        raise ValueError(f"Missing article columns: {sorted(missing)}")
    d = raw.copy().reset_index(drop=True)
    d["source_row"] = np.arange(len(d)) + source_offset
    d["source_date"] = pd.to_datetime(d.Date, errors="raise").astype("datetime64[ns]")
    if d.source_date.isna().any() or not d.source_date.eq(d.source_date.dt.normalize()).all():
        raise ValueError("Article dates must be valid date-only values.")
    fingerprints = []
    for row in d.itertuples():
        value = json.dumps([row.Date, row.Title, row.Source, row.URL, row.Text_clean], ensure_ascii=False)
        fingerprints.append(hashlib.sha256(value.encode()).hexdigest()[:20])
    fingerprints = pd.Series(fingerprints, index=d.index)
    occurrence = fingerprints.groupby(fingerprints).cumcount()
    d["article_id"] = prefix + fingerprints + "_" + occurrence.astype(str)
    d["title_main"] = d.Title.map(clean_plain_text)
    d["body_main"] = d.Text_clean.map(lambda text: clean_body(text)[0])
    d["main_word_count"] = d.body_main.str.split().str.len()
    d["lead_main"] = d.body_main.map(lambda text: " ".join(text.split()[:max_body_words]))
    d["text_transformer"] = d.title_main + "\n\n" + d.lead_main
    d["text_tfidf"] = d.text_transformer.map(lambda text: " ".join(static_tokens(text, lowercase=True)))
    d["tokens_word2vec"] = d.text_transformer.map(static_tokens)
    d["content_key"] = d.body_main.map(normalize_key)
    d["known_temporal_contamination"] = d.article_id.isin(QUARANTINED_ARTICLE_IDS)
    d["future_date_heuristic"] = [bool(later_date_mentions(text, date)) for text, date in zip(d.Text_clean, d.source_date)]
    d["date_mismatch_review"] = (d.URL.map(url_date) - d.source_date).dt.days.abs().gt(1)
    d["encoding_flag"] = d.body_main.str.contains(r"â€|â|Ã.|�", regex=True)
    d["too_short"] = d.main_word_count.lt(40) | d.title_main.str.split().str.len().lt(3)
    d["duplicate_of"] = ""
    seen = {}
    for index in d.sort_values(["source_date", "source_row"]).index:
        key = d.at[index, "content_key"]
        if len(key) >= 80:
            if key in seen:
                d.at[index, "duplicate_of"] = seen[key]
            else:
                seen[key] = d.at[index, "article_id"]
    return select_articles(d)

def select_articles(articles):
    d = articles.copy()
    review = d.future_date_heuristic | d.date_mismatch_review | d.encoding_flag
    blocked = d.known_temporal_contamination | d.too_short | d.duplicate_of.ne("")
    d["selected_for_alignment"] = ~review & ~blocked
    d["selection_reason"] = np.select(
        [d.duplicate_of.ne(""), d.known_temporal_contamination, d.too_short, review],
        ["exact_content_duplicate", "known_later_content_quarantine", "insufficient_main_text", "quality_review_required"],
        default="quality_pass",
    )
    return d

