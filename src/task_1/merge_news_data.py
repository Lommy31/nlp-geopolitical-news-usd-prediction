import csv
import os
import sys

csv.field_size_limit(sys.maxsize)

INPUT_PATHS = [
    "data/raw/news.csv",
    "data/raw/news_scraped.csv",
    "data/raw/news_scraped_v2.csv",
]
OUTPUT_PATH = "data/raw/news_merged.csv"
FIELDNAMES = ["Date", "Title", "Source", "URL", "Event", "Keyword", "Category", "Text"]


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    seen_urls = set()
    merged = []

    for path in INPUT_PATHS:
        if not os.path.exists(path):
            print(f"skip (not found): {path}")
            continue

        rows = load_rows(path)
        added = 0
        for row in rows:
            url = row.get("URL")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            merged.append(row)
            added += 1

        print(f"{path}: {len(rows)} rows, {added} new after dedup")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(merged)

    print(f"saved {len(merged)} merged articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
