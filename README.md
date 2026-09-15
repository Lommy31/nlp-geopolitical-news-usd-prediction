# Geopolitical Sentiment and USD/IDR Exchange Rate Dynamics: An NLP-Driven Time-Series Analysis

## Overview

This project investigates whether global geopolitical and economic news carries information that helps explain and predict movements in the USD/IDR exchange rate. It combines two data streams: news articles covering major geopolitical and economic events, and daily USD/IDR exchange rate data, then uses NLP techniques to extract structured signal from the news text for analysis against exchange rate movements.

## Objective

- Collect news articles covering major geopolitical and economic events from 2021 to 2026.
- Collect daily USD/IDR exchange rate data from Bank Indonesia for the same period.
- Clean and filter both datasets to remove noise, duplicates, and irrelevant content.
- Align news events with exchange rate observations on a shared trading-day calendar.
- Apply NLP techniques to extract information from news articles for use in later modeling stages.

## Data Sources

**News data** is collected via the GDELT Project's DOC 2.0 API, which indexes hundreds of thousands of global news sources and supports keyword search over a specified date range. Eleven topic queries cover Fed policy, US-China/Taiwan tensions, US fiscal stimulus, banking crises, US elections, trade tariffs, Indonesian Rupiah/Bank Indonesia policy, the Russia-Ukraine war, Middle East conflict and oil shocks, BOJ/ECB policy shifts, and emerging market currency contagion.

**Exchange rate data** is the JISDOR (Jakarta Interbank Spot Dollar Rate), Bank Indonesia's official USD/IDR reference rate, exported from BI's public statistics portal (`https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/Default.aspx`). JISDOR is published only on days the market is open, so it also serves as the authoritative trading-day calendar used later for alignment.

## Repository Structure

```
data/
  raw/         raw scraped news and the JISDOR export
  processed/   cleaned and aligned datasets, ready for analysis
src/
  news_scraper.py      scrapes news candidates from GDELT and fetches article text
  merge_news_data.py   merges and deduplicates raw scrape files into one file
  preprocess_fx.py      cleans the JISDOR exchange rate data
  preprocess_news.py    cleans, filters, and temporally aligns the news data
report/
  pipeline_writeup.md   detailed methodology writeup
  task1_workflow.mmd     workflow diagram (Mermaid)
```

## How to Run

```bash
pip install -r requirements.txt
python3 src/preprocess_fx.py
python3 src/merge_news_data.py
python3 src/preprocess_news.py
```

Running `news_scraper.py` first is only needed if new raw data is being collected; it is resumable and safe to interrupt. `preprocess_fx.py` must run before `preprocess_news.py`, since the news cleaning step depends on the trading-day calendar derived from the exchange rate data.

## Current Dataset

- `data/processed/usd_idr_jisdor_clean.csv`: 1,202 daily USD/IDR observations, 1 September 2021 to 1 September 2026.
- `data/processed/news_clean.csv`: 388 cleaned, deduplicated, and date-aligned news articles across 7 event categories, covering the same period.

See `report/pipeline_writeup.md` for a full explanation of the scraping approach, cleaning pipeline, and temporal alignment strategy, including the reasoning behind each design decision.

## Group Members

- Deira Aisya Refani (24/532821/PA/22539)
- Nareswari Ayu Prabowo (24/532991/PA/22558)
- Yohana Butar Butar (24/546690/PA/23212)
