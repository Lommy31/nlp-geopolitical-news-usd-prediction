import os
import re
import sys
import unicodedata
from bisect import bisect_left

import pandas as pd

NEWS_PATH = "data/raw/news_scraped.csv"
FX_PATH = "data/processed/usd_idr_jisdor_clean.csv"
OUTPUT_PATH = "data/processed/news_clean.csv"

MIN_WORDS = 150

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

# single-company stock picks / earnings calls aren't geopolitical news even
# when they happen to mention "dollar" or "Fed"
SINGLE_COMPANY_MARKERS = re.compile(r"\((?:NYSE|NASDAQ):|earnings call transcript", re.IGNORECASE)

# press releases mirrored onto a domain that isn't itself in BLOCKED_DOMAINS
WIRE_SERVICE_MARKERS = ("/prnewswire/", "/business wire/", "globe newswire", "accesswire")

# placeholder-generator vocabulary -- essentially never appears in real English
# news, so a hit means a broken page (empty CMS block) got scraped instead
LOREM_IPSUM_MARKERS = ("pellentesque", "ullamcorper", "consectetur", "adipiscing")

# generic single words like "tariff" or "bank" match too much unrelated content
# (a utility tariff, a pharma trial, a mortgage startup) -- anchor each topic
# to the specific phrases its own query was built around
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
    title = re.sub(r"^[^|]{1,20}\|\s*", "", title)  # strip outlet prefixes like "GLOBALink | "
    title = unicodedata.normalize("NFKC", title).lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return " ".join(title.split()[:6])  # same story, different suffix/attribution -> same key


def is_placeholder_content(text):
    text_lower = text.lower()
    return any(m in text_lower for m in LOREM_IPSUM_MARKERS)


def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    junk = BOILERPLATE_LINES + PAYWALL_MARKERS
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line and not any(m in line.lower() for m in junk)]
    text = "\n\n".join(lines)
    return re.sub(r"[ \t]+", " ", text).strip()


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


def build_trading_days(fx_path):
    fx = pd.read_csv(fx_path, parse_dates=["Date"])
    return sorted(fx["Date"].dt.date.tolist())


def align_date(date, trading_days):
    idx = bisect_left(trading_days, date)
    if idx >= len(trading_days):
        return None
    return trading_days[idx]


def main():
    if not os.path.exists(FX_PATH):
        sys.exit(f"{FX_PATH} not found, run preprocess_fx.py first")

    df = pd.read_csv(NEWS_PATH)
    print(f"loaded {len(df)} raw articles")

    df = df[~df["Source"].isin(BLOCKED_DOMAINS)]
    print(f"{len(df)} left after dropping press-release / stock-picking domains")

    df = df[~df["Text"].apply(is_single_company_content)]
    print(f"{len(df)} left after dropping single-company content")

    df = df[~df["Text"].apply(is_wire_service_mirror)]
    print(f"{len(df)} left after dropping mirrored wire releases")

    df = df[~df["Text"].apply(is_placeholder_content)]
    print(f"{len(df)} left after dropping broken/placeholder pages")

    df = df[df.apply(lambda r: has_anchor(r["Event"], r["Text"]), axis=1)]
    print(f"{len(df)} left after anchor-term relevance check")

    df["Text"] = df["Text"].apply(clean_text)

    df["word_count"] = df["Text"].str.split().str.len()
    df = df[df["word_count"] >= MIN_WORDS]
    print(f"{len(df)} left after text cleaning + length filter")

    df["title_key"] = df["Title"].apply(normalize_title)
    df = df.sort_values("word_count", ascending=False).drop_duplicates(subset="title_key")
    print(f"{len(df)} left after near-duplicate title dedup")

    trading_days = build_trading_days(FX_PATH)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Aligned_Date"] = df["Date"].apply(lambda d: align_date(d, trading_days))

    unaligned = df["Aligned_Date"].isna().sum()
    if unaligned:
        print(f"warning: {unaligned} articles fall after the last trading day, left unaligned")

    df = df.rename(columns={"Text": "Text_clean"})
    df = df[["Date", "Aligned_Date", "Title", "Source", "URL", "Event", "Keyword", "Category", "Text_clean"]]
    df = df.sort_values("Date").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"saved {len(df)} articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
