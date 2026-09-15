import csv
import os
import re
import sys
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

# a few already-saved rows have oversized Text fields from before MAX_TEXT_LENGTH
# existed; raise the limit so resuming can still read past them
csv.field_size_limit(sys.maxsize)

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

START_DATE = datetime(2021, 9, 1)
END_DATE = datetime(2026, 9, 1)

# quick sanity check on a short window before committing to the full 5-year run
if "--probe" in sys.argv:
    START_DATE = END_DATE - timedelta(days=60)

# task-recommended outlets only, so we're not pulling from wire-mirrors, press
# release farms or stock-picking blogs that just happen to mention "dollar"
DOMAIN_ALLOWLIST = ["cnbc.com", "reuters.com", "apnews.com", "tradingeconomics.com"]

# add outlets here if DOMAIN_ALLOWLIST doesn't have enough volume across 5 years
FALLBACK_DOMAINS = ["bbc.com", "aljazeera.com"]

MAX_RECORDS_PER_CALL = 250
MAX_ARTICLES_PER_EVENT = 200
MAX_ARTICLES_PER_WINDOW = 5  # caps how many a single window can contribute, so
                              # one dense period (e.g. 2021-2022) can't eat the
                              # whole per-event budget before later years get a turn
MIN_TEXT_LENGTH = 200
MAX_TEXT_LENGTH = 20_000  # a real news article is a few thousand chars; way past
                           # this means extract_text() swept up a broken page
GDELT_DELAY = 1.2
FETCH_DELAY = 1.0
OUTPUT_PATH = "data/raw/news_scraped_v2.csv"
FIELDNAMES = ["Date", "Title", "Source", "URL", "Event", "Keyword", "Category", "Text"]

QUERIES = [
    {
        "event": "US Fiscal Stimulus + Rising Treasury Yields",
        "category": "Fiscal Policy / Interest Rates",
        "keyword": "US fiscal stimulus; Treasury yields; USD; inflation",
        "query": '(dollar OR USD) ("treasury yields" OR "fiscal stimulus" OR inflation) sourcelang:eng',
    },
    {
        "event": "Federal Reserve Monetary Policy",
        "category": "Monetary Policy",
        "keyword": "Federal Reserve; rate cut; rate hike; Powell; interest rates",
        "query": '(dollar OR USD) ("Federal Reserve" OR "interest rate" OR Powell OR tapering) sourcelang:eng',
    },
    {
        "event": "US-China / Taiwan Tensions",
        "category": "Geopolitical Tension",
        "keyword": "US-China tensions; Taiwan; geopolitics; USD",
        "query": '(dollar OR USD) ("US-China" OR Taiwan OR "South China Sea") sourcelang:eng',
    },
    {
        "event": "Banking & Financial Crisis",
        "category": "Banking / Financial Crisis",
        "keyword": "banking crisis; bank collapse; SVB; Federal Reserve; USD",
        "query": '(dollar OR USD) ("bank collapse" OR "banking crisis" OR SVB) sourcelang:eng',
    },
    {
        "event": "US Presidential Election",
        "category": "US Election / Fiscal & Trade Policy",
        "keyword": "US presidential election; Trump; election; USD",
        "query": '(dollar OR USD) ("presidential election" OR "US election") sourcelang:eng',
    },
    {
        "event": "Trump Tariffs / Global Trade War",
        "category": "Trade Policy / Geopolitical",
        "keyword": "Trump tariffs; trade war; China; USD; markets",
        "query": '(dollar OR USD) (tariff OR "trade war") sourcelang:eng',
    },
    {
        "event": "Indonesian Rupiah / Bank Indonesia Policy",
        "category": "Domestic Monetary Policy",
        "keyword": "Bank Indonesia; rupiah; IDR; BI rate",
        "query": '(rupiah OR IDR OR "Bank Indonesia") (dollar OR USD OR "exchange rate") sourcelang:eng',
    },
    {
        "event": "Russia-Ukraine War & Sanctions",
        "category": "Geopolitical Conflict / Sanctions",
        "keyword": "Russia-Ukraine war; sanctions; SWIFT; USD",
        "query": '(dollar OR USD) ("Ukraine war" OR "Russia sanctions" OR SWIFT) sourcelang:eng',
    },
    {
        "event": "Middle East Conflict & Oil Shock",
        "category": "Geopolitical Conflict / Energy",
        "keyword": "Israel; Gaza; Iran; Middle East; oil prices; USD",
        "query": '(dollar OR USD) (Israel OR Gaza OR Iran OR "Middle East") (oil OR war OR conflict) sourcelang:eng',
    },
    {
        "event": "BOJ / ECB Policy Shifts",
        "category": "Global Monetary Policy",
        "keyword": "Bank of Japan; ECB; yen; euro; interest rate; USD",
        "query": '(dollar OR USD) ("Bank of Japan" OR ECB) ("interest rate" OR yen OR euro) sourcelang:eng',
    },
    {
        "event": "Emerging Market Currency Contagion",
        "category": "EM Currency Contagion",
        "keyword": "emerging market currency; lira; peso; sovereign default; USD",
        "query": '(dollar OR USD) ("emerging market" OR "currency crisis" OR "sovereign default") sourcelang:eng',
    },
]

JUNK_MARKERS = (
    "cookie",
    "subscribe",
    "sign up",
    "sign in",
    "all rights reserved",
    "advertisement",
    "follow us",
    "newsletter",
)


def domain_clause(domains):
    return "(" + " OR ".join(f"domainis:{d}" for d in domains) + ")"


def month_windows(start, end):
    cursor = start
    while cursor < end:
        window_end = min(cursor + timedelta(days=30), end)
        yield cursor, window_end
        cursor = window_end


def query_gdelt(query, start, end):
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": MAX_RECORDS_PER_CALL,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
    }

    try:
        response = requests.get(GDELT_ENDPOINT, params=params, headers=HEADERS, timeout=20)
        response.raise_for_status()
        return response.json().get("articles", [])
    except (requests.RequestException, ValueError) as e:
        # distinguish a real "no articles" from a failed/blocked request --
        # otherwise both look like 0 candidates in the log
        print(f"  request failed for window {start.date()}-{end.date()}: {e}")
        return []


def collect_metadata(event_query, existing_urls):
    seen_urls = set(existing_urls)
    rows = []

    for start, end in month_windows(START_DATE, END_DATE):
        if len(rows) >= MAX_ARTICLES_PER_EVENT:
            break

        full_query = f"{event_query['query']} {domain_clause(DOMAIN_ALLOWLIST)}"
        articles = query_gdelt(full_query, start, end)
        time.sleep(GDELT_DELAY)

        print(f"  window {start.date()} to {end.date()}: {len(articles)} candidates from GDELT")

        taken_this_window = 0

        for article in articles:
            if taken_this_window >= MAX_ARTICLES_PER_WINDOW:
                break

            url = article.get("url")
            seen_date = article.get("seendate")
            if not url or not seen_date or url in seen_urls:
                continue

            seen_urls.add(url)
            rows.append({
                "Date": datetime.strptime(seen_date, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d"),
                "Title": article.get("title", "").strip(),
                "Source": article.get("domain", ""),
                "URL": url,
                "Event": event_query["event"],
                "Keyword": event_query["keyword"],
                "Category": event_query["category"],
            })
            taken_this_window += 1

            if len(rows) >= MAX_ARTICLES_PER_EVENT:
                break

    return rows


def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "figure", "form"]):
        tag.decompose()

    paragraphs = [re.sub(r"\s+", " ", p.get_text(" ", strip=True)) for p in soup.find_all("p")]
    paragraphs = [
        p for p in paragraphs
        if len(p.split()) > 5 and not any(marker in p.lower() for marker in JUNK_MARKERS)
    ]

    return "\n\n".join(paragraphs)


def fetch_article_text(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return extract_text(response.text)
    except requests.RequestException:
        return ""


def load_existing_urls(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["URL"] for row in reader if row.get("URL")}


def open_writer(path):
    is_new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    out_file = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)

    if is_new_file:
        writer.writeheader()
        out_file.flush()

    return out_file, writer


def build_dataset():
    existing_urls = load_existing_urls(OUTPUT_PATH)
    saved_count = len(existing_urls)
    print(f"resuming with {saved_count} articles already saved in {OUTPUT_PATH}")

    out_file, writer = open_writer(OUTPUT_PATH)

    try:
        for event_query in QUERIES:
            print(f"searching GDELT for: {event_query['event']}")
            metadata_rows = collect_metadata(event_query, existing_urls)
            print(f"  found {len(metadata_rows)} new candidate articles")

            for row in metadata_rows:
                text = fetch_article_text(row["URL"])
                time.sleep(FETCH_DELAY)

                if len(text) < MIN_TEXT_LENGTH:
                    print(f"  skip (text too short): {row['Title'][:60]}")
                    continue

                if len(text) > MAX_TEXT_LENGTH:
                    print(f"  skip (extraction too long, {len(text)} chars, likely broken): {row['Title'][:60]}")
                    continue

                row["Text"] = text
                writer.writerow(row)
                out_file.flush()

                existing_urls.add(row["URL"])
                saved_count += 1
                print(f"  [{saved_count}] saved: {row['Title'][:60]}")
    finally:
        out_file.close()

    return saved_count


if __name__ == "__main__":
    total_saved = build_dataset()
    print(f"done, {total_saved} articles total in {OUTPUT_PATH}")
