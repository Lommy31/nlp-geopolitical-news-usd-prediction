"""Scrapes CNBC directly via its public sitemaps instead of going through GDELT.

robots.txt advertises these sitemaps and allows article pages under User-agent: *;
/search/ is disallowed and is never touched here.
"""

import csv
import os
import random
import re
import sys
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

csv.field_size_limit(sys.maxsize)

SITEMAP_INDEX = "https://www.cnbc.com/sitemapAll.xml"
SITEMAP_CACHE = "data/raw/cnbc_sitemaps"
URL_LIST_PATH = "data/raw/cnbc_urls.txt"
OUTPUT_PATH = "data/raw/news_cnbc.csv"

START_DATE = "2021-09-01"
END_DATE = "2026-09-01"

FETCH_DELAY = 0.8
MIN_TEXT_LENGTH = 400
MAX_TEXT_LENGTH = 20_000

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

FIELDNAMES = ["Date", "Title", "Source", "URL", "Event", "Keyword", "Category", "Text"]

# the URL slug carries the headline, so topic can be assigned before fetching
TOPIC_RULES = [
    ("Indonesian Rupiah / Bank Indonesia Policy", "Domestic Monetary Policy",
     "Bank Indonesia; rupiah; IDR",
     ("rupiah", "indonesia", "jakarta")),
    ("Emerging Market Currency Contagion", "EM Currency Contagion",
     "emerging market; currency crisis; peso; lira",
     ("emerging-market", "currency-crisis", "peso", "lira", "capital-outflow")),
    ("BOJ / ECB Policy Shifts", "Global Monetary Policy",
     "Bank of Japan; ECB; yen; euro",
     ("bank-of-japan", "boj", "ecb", "european-central-bank", "yen", "lagarde")),
    ("Russia-Ukraine War & Sanctions", "Geopolitical Conflict / Sanctions",
     "Russia; Ukraine; sanctions; SWIFT",
     ("russia", "ukraine", "sanction", "swift", "putin", "kremlin", "moscow")),
    ("Middle East Conflict & Oil Shock", "Geopolitical Conflict / Energy",
     "Israel; Iran; Gaza; oil; OPEC",
     ("israel", "iran", "gaza", "opec", "oil-price", "crude", "hormuz", "middle-east")),
    ("US-China / Taiwan Tensions", "Geopolitical Tension",
     "US-China; Taiwan; South China Sea",
     ("taiwan", "south-china-sea", "us-china", "china-us", "xi-jinping")),
    ("Trump Tariffs / Global Trade War", "Trade Policy / Geopolitical",
     "tariffs; trade war; trade deal",
     ("tariff", "trade-war", "trade-deal", "trade-talks")),
    ("Banking & Financial Crisis", "Banking / Financial Crisis",
     "banking crisis; bank collapse; SVB",
     ("banking-crisis", "bank-collapse", "svb", "silicon-valley-bank", "credit-suisse",
      "bank-failure", "bank-run")),
    ("Federal Reserve Monetary Policy", "Monetary Policy",
     "Federal Reserve; Powell; rate hike; rate cut",
     ("federal-reserve", "powell", "fomc", "rate-hike", "rate-cut", "interest-rate",
      "-fed-", "/fed-")),
    ("US Presidential Election", "US Election / Fiscal & Trade Policy",
     "US presidential election; Trump; Biden",
     ("election", "presidential", "white-house", "campaign")),
    ("US Fiscal Stimulus + Rising Treasury Yields", "Fiscal Policy / Interest Rates",
     "Treasury yields; fiscal stimulus; inflation",
     ("treasury-yield", "bond-yield", "inflation", "fiscal", "stimulus", "debt-ceiling")),
]

# anything matching none of the above is not collected at all
FX_CORE = ("dollar", "currency", "currencies", "forex", "exchange-rate", "greenback")


def download_sitemaps():
    os.makedirs(SITEMAP_CACHE, exist_ok=True)
    index = requests.get(SITEMAP_INDEX, headers=HEADERS, timeout=30).text
    locs = re.findall(r"<loc>(https://www\.cnbc\.com/CNBCsitemapAll\d+\.xml)</loc>", index)
    print(f"sitemap index lists {len(locs)} sitemaps")

    paths = []
    for loc in locs:
        name = loc.rsplit("/", 1)[-1]
        path = os.path.join(SITEMAP_CACHE, name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"  downloading {name}")
            r = requests.get(loc, headers=HEADERS, timeout=90)
            with open(path, "w", encoding="utf-8") as f:
                f.write(r.text)
            time.sleep(1)
        paths.append(path)
    return paths


def classify(url):
    lowered = url.lower()
    for event, category, keyword, terms in TOPIC_RULES:
        if any(t in lowered for t in terms):
            return event, category, keyword
    if any(t in lowered for t in FX_CORE):
        return ("US Fiscal Stimulus + Rising Treasury Yields",
                "Fiscal Policy / Interest Rates", "dollar; currency")
    return None


def build_url_list(sitemap_paths):
    pattern = re.compile(r"https://www\.cnbc\.com/(\d{4})/(\d{2})/(\d{2})/[^<\"]+?\.html")
    seen = set()
    rows = []

    for path in sitemap_paths:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for m in pattern.finditer(content):
            url = m.group(0)
            if url in seen:
                continue
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            if not (START_DATE <= date <= END_DATE):
                continue
            topic = classify(url)
            if topic is None:
                continue
            seen.add(url)
            rows.append((date, url) + topic)

    rows.sort()
    with open(URL_LIST_PATH, "w", encoding="utf-8") as f:
        for date, url, event, category, keyword in rows:
            f.write(f"{date}\t{url}\t{event}\t{category}\t{keyword}\n")

    print(f"{len(rows)} in-window CNBC articles matched a topic")
    return rows


def load_url_list():
    rows = []
    with open(URL_LIST_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 5:
                rows.append(tuple(parts))
    return rows


def extract_article(html):
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "figure", "form"]):
        tag.decompose()

    paragraphs = [re.sub(r"\s+", " ", p.get_text(" ", strip=True)) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p.split()) > 5]
    return title, "\n\n".join(paragraphs)


def load_done_urls():
    if not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0:
        return set()
    with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
        return {row["URL"] for row in csv.DictReader(f) if row.get("URL")}


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    if not os.path.exists(URL_LIST_PATH):
        build_url_list(download_sitemaps())
    rows = load_url_list()
    print(f"url list: {len(rows)} articles")

    done = load_done_urls()
    todo = [r for r in rows if r[1] not in done]

    # shuffled so an interrupted run still covers the whole 2021-2026 span
    # evenly instead of only the earliest months
    random.Random(42).shuffle(todo)
    if limit:
        todo = todo[:limit]
    print(f"already saved: {len(done)} | fetching now: {len(todo)}\n")

    is_new = not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0
    out = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
    if is_new:
        writer.writeheader()
        out.flush()

    saved = len(done)
    skipped = 0
    try:
        for i, (date, url, event, category, keyword) in enumerate(todo, 1):
            try:
                r = requests.get(url, headers=HEADERS, timeout=20)
                r.raise_for_status()
                title, text = extract_article(r.text)
            except requests.RequestException as e:
                skipped += 1
                if skipped % 25 == 0:
                    print(f"  [{i}] fetch failed ({skipped} so far): {e}")
                time.sleep(FETCH_DELAY)
                continue

            time.sleep(FETCH_DELAY)

            if not (MIN_TEXT_LENGTH <= len(text) <= MAX_TEXT_LENGTH) or not title:
                skipped += 1
                continue

            writer.writerow({
                "Date": date, "Title": title, "Source": "cnbc.com", "URL": url,
                "Event": event, "Keyword": keyword, "Category": category, "Text": text,
            })
            out.flush()
            saved += 1

            if saved % 25 == 0:
                print(f"  [{saved} saved / {i} tried / {skipped} skipped] {date}  {title[:55]}")
    finally:
        out.close()

    print(f"\ndone: {saved} articles in {OUTPUT_PATH} ({skipped} skipped)")


if __name__ == "__main__":
    main()
