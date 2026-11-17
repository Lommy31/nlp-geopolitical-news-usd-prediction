# Audit of news relevance and its relationship with USD/IDR

Audit date: 10 November 2026. Inputs: `data/processed/news_clean.csv` and `data/processed/usd_idr_jisdor_clean.csv`. All new outputs are in `deira-work/sanity-check/`. The previous notebook was not read, executed, or used as the basis for the conclusions.

## Main findings

The concern about news quality is supported by evidence: some articles are off-topic, article bodies include other news cards, later-dated information appears in older records, and daily coverage is sparse. The dataset should not be assumed clean for USD/IDR prediction experiments without further auditing.

However, this audit **does not establish that news is unrelated to exchange rates**, nor that data quality is the sole cause of previous model results. The more precise statistical finding is: **the simple news proxies tested have not demonstrated convincing, stable relationships with USD/IDR returns after correction for multiple testing.** Some articles remain economically relevant.

| Check | Result |
|---|---:|
| Articles in the current CSV | 435 |
| Event categories | 11 |
| Exchange-rate observations | 1,202 |
| Date range | 1 September 2021–1 September 2026 |
| Exchange-rate days with at least one article | 212 / 1,202 = 17.64% |
| Months without collected articles | 4 / 61 |
| Random sample for content review | 66, six per category |
| Sampled articles with weak or unclear relevance | 36 / 66 |
| Sampled articles with indirect macroeconomic relevance | 27 / 66 |
| Sampled articles with direct relevance under a strict definition | 3 / 66 |
| Records containing related content dated after the article, confirmed from stored context | at least 10 |
| Future-outcome relationships meeting BH q < 0.05 | 0 / 216 exploratory tests |

Article and category counts were recalculated from the actual file; the older README description was not used as the source of dataset counts.

## 1. What does “correlates” mean in this audit?

There are two separate questions:

1. **Economic relevance:** does the main article provide information that could reasonably explain USD/IDR, such as Bank Indonesia policy, rupiah conditions, changes in Fed expectations, or global risk?
2. **Statistical association:** are news features available on a particular date associated with exchange-rate changes on that date or afterward?

A single article that happens to precede an exchange-rate increase is not evidence of correlation. Because multiple articles share the same exchange-rate date, tests operate at the **trading-day level**, rather than treating every article as an independent exchange-rate observation.

The target is the percentage return, rather than the exchange-rate level:

`R_h(t) = 100 × [USD_IDR(t+h) / USD_IDR(t) − 1]`

`h` denotes the next 1, 3, or 5 exchange-rate observations. A positive return means more rupiah are required to buy one USD. Same-day returns use `100 × [P(t)/P(t−1) − 1]` and serve only as descriptive diagnostics because article publication times are unavailable.

## 2. Sampling method and review limitations

Stratified random sampling was performed across 11 `Event` categories, selecting six articles without replacement per category: 66 articles in total, or 15.17% of the dataset. The seed is `20261110`; event groups are sorted alphabetically and row indices follow the original CSV. `article_id` is the **zero-based data-record index**, not a physical CSV line number, since text fields may contain newlines.

One AI reviewer inspected each title, approximately the first 1,400–1,700 characters, and the final 250 characters. Additional context was read for flagged cases, particularly mixed article content and dates. **This is not an independent human assessment, a complete reading of all 66 article bodies, or final ground truth for training a classifier.** Full stored text is included so another reviewer can inspect it.

The rubric does not use exchange-rate outcomes:

- `direct`: the main article explicitly concerns the rupiah/Indonesia and exchange-rate, reserve, monetary, or rupiah-related energy-cost channels.
- `indirect`: the main article has a plausible macroeconomic/global channel, but its effect on USD/IDR is not explicit.
- `weak`: the connection to USD/IDR is unclear from the reviewed portions. Examples often concern another country's local politics, investment advice for a particular asset, or keywords appearing only as background. This label indicates a need for review or lower weighting, rather than automatic removal of the entire article.

The fit of `Event` is assessed separately as `yes`, `partial`, or `no`. An article can be useful while having an incorrect event label.

| Relevance | Sample count | Sample percentage | Weighted estimate of dataset composition |
|---|---:|---:|---:|
| Direct | 3 | 4.55% | 5.29% |
| Indirect | 27 | 40.91% | 42.76% |
| Weak/unclear | 36 | 54.55% | 51.95% |

Weighted percentages use `N_kategori / 6`. Weighting is necessary because six articles are selected from both small and large categories. These estimates remain approximate: each category's sample is small, and annotations depend on the rubric and reviewer.

For event labels, 19 samples do not match, 31 partially match, and 16 match. A partial match is not treated as completely incorrect. Decisions and reasons are recorded in [sample_review_66.csv](sample_review_66.csv).

## 3. Inspectable evidence of content problems

| article_id | Stored article/date | Finding | Implication for NLP |
|---:|---|---|---|
| 237 | *Philippines bans poultry imports from Japan*, 19 January 2024 | Assigned the rupiah/Bank Indonesia event. After the main article, the text shifts to Indonesia, ASEAN, and other news. | Mentioning Indonesia can pass the filter even though the main article concerns poultry in the Philippines. |
| 53 | *Turkish women gassed by police…*, 30 November 2021 | Assigned the US-China/Taiwan event. The ending includes a newsletter, an invitation to submit releases, and news excerpts mentioning Beijing. | Entity/topic detection over the complete text can capture material outside the main article. |
| 28 | *Samsung and SK…SSD market*, 1 October 2021 | Focuses on SSD products/exports rather than Taiwan tensions; the ending contains a card dated 28 June 2025. | Both topic noise and temporal contamination. |
| 190 | *…Warren Buffett and buying cheap ASX shares*, 25 December 2022 | Motley Fool Australia investment advice; the ending includes recommendations dated 2–14 September 2026. | Features for 2022 can contain information originating in 2026. |
| 64 | *Copper imports in China surged in 2020…*, 23 December 2021 | The URL contains 19 January 2021; the body still forecasts the first half of 2021. The URL and record dates differ by 338 days. | Associating stale text with December returns risks incorrect timing. The URL is a clue rather than a verified timestamp. |
| 40 | *Vietnam President to be on a visit to Russia…*, 29 November 2021 | US-China/Taiwan event; long paragraphs repeat. | Repetition can overweight the same words or entities. |
| 24 | *2:00PM Water Cooler…*, 30 September 2021 | 13,853 words, multiple topics and reader comments; some content does concern US fiscal policy. | A single sentiment score does not represent one article or one shock. Model truncation can capture different portions. |
| 317 | *For bitcoin bulls who self-custody crypto…*, 6 April 2025 | Russia-Ukraine event, but the focus is wallet/seed-phrase storage. | The word “risk” or negative cryptocurrency sentiment is not equivalent to USD/IDR risk. |

Examples of news that remains relevant:

- **145:** rupiah resilience, trade surpluses, and Asian central-bank conditions.
- **150:** the rupiah, Indonesian fuel subsidies, and purchases of Russian oil.
- **157:** Fed tightening, dollar strength, and rupiah/Asian FX weakness; opinion language needs to be separated from macroeconomic facts.
- **378:** the dollar, oil, and safe-haven demand; useful macroeconomic content, although much of it reports price reactions that have already occurred.

Relevance does not require every article to mention Indonesia. Global news can also be relevant. The question is whether the **main content**, economic channel, novelty, and information availability are appropriate for the target.

### Temporal contamination: not every future date indicates leakage

Searching for full English-language dates more than 90 days after `Date` flagged 12 articles. Their contexts were inspected:

- **10 articles** contain excerpts/cards from other news dated later: IDs `1, 28, 38, 48, 78, 108, 121, 190, 192, 212`.
- **2 articles** contain legitimate prospective dates: the effective date of tariffs (63) and the LIBOR transition schedule (128). Neither is counted as evidence of contamination.

The ten cases are a lower bound detected using a narrow date pattern, rather than a complete census of leakage. A reference to 2026 in a forecast is not automatically wrong. The problem is including **other news content published later** within text assigned an older date. Current records also lack archived publication versions that could establish what the text contained as of its original date.

See [future_date_flags.csv](future_date_flags.csv) for character positions and excerpts, and [future_date_review.csv](future_date_review.csv) for the distinction between the two types of cases.

### Automated indicators across the complete dataset

| Indicator | Count | Interpretation limit |
|---|---:|---|
| Contains boilerplate/promotion/login/comment markers | 70 | A heuristic flag, not a measure of all corrupted text. |
| Contains identical long paragraphs repeated within a record | 22 | Only exact paragraphs at least 80 characters long; other duplication formats may be missed. |
| Contains corrupted-encoding markers | 3 | Narrow patterns that do not detect every text error. |
| Mentions rupiah/Bank Indonesia/USD-IDR anywhere in the full text | 42 | May come from additional paragraphs; not a validated count of relevant articles. |
| Mentions those anchors within the first 200 words | 17 | The 200-word cutoff can also miss relevant discussion appearing later. |
| Candidate title/content duplication pairs | 7 | Title word-set Jaccard ≥0.70 or identical normalized text; not every candidate is necessarily a duplicate. |
| Pairs with genuinely identical normalized content | 1 | IDs 104 and 105. |

Within “Indonesian Rupiah / Bank Indonesia Policy,” 16 of 46 articles contain a rupiah/BI anchor in their first 200 words, and 38 of 46 contain one in the full text. This diagnoses a content/position mismatch; it **does not** establish that the remaining articles are irrelevant. Even articles passing the early anchor check can be dominated by another country's currency.

## 4. Coverage and timing constrain the signal

There are 990 exchange-rate days without collected articles. A feature value of zero means **no article was collected**, rather than “no news existed” or an observed neutral sentiment.

| Year | Articles | Days with news | Exchange-rate days | Daily coverage |
|---|---:|---:|---:|---:|
| 2021, starting in September | 75 | 27 | 87 | 31.03% |
| 2022 | 118 | 33 | 246 | 13.41% |
| 2023 | 41 | 24 | 239 | 10.04% |
| 2024 | 67 | 52 | 238 | 21.85% |
| 2025 | 63 | 33 | 237 | 13.92% |
| 2026, through 1 September | 71 | 43 | 155 | 27.74% |

News is also clustered: 27 articles map to one exchange-rate day, 30 March 2022. Of the 46 rupiah/BI-category articles, 38 are from 2022; 15 of the 17 articles with domestic anchors in their first 200 words are also from 2022. Category/domestic features can therefore capture collection periods rather than topic effects that generalize across the five-year sample.

CNBC contributes 122 articles, or 28.05%, among 191 sources. This suggests potential coverage/editorial bias, rather than evidence that the source is poor. Coverage is uneven, and the reasons for missingness cannot be determined from these two CSV files alone.

Date checks:

- Every `Aligned_Date` follows the calendar rule **the first exchange-rate day on or after Date**. No implementation mismatch with that rule was found.
- Calendar alignment shifts 105 articles to a later exchange-rate day.
- Of 204 URLs containing a date in a recognized format, 23 differ from `Date`. Three differ by more than one day: IDs 26, 64, and 97.
- A one-day difference may reflect time zones or indexing time and is not automatically incorrect. Even a two-day difference requires verification against the original publication.
- The dataset stores dates without publication timestamps or time zones. News published after the same day's JISDOR observation must not be treated as available beforehand. Calendar-consistent `Aligned_Date` values do not establish the correct information sequence.

Basic checks of the exchange-rate input found no duplicate dates, nonpositive values, or weekend observations. These are local consistency checks, **not** verification against official Bank Indonesia archives.

## 5. Statistical association results

Daily features tested:

- total article count;
- counts of articles passing domestic and macroeconomic anchor checks;
- mean dollar-direction cues in titles, the first 200 words, and the full text;
- mean hawkish-minus-dovish cues in the opening text that mentions the Fed;
- article counts for each of the 11 event labels.

There are 18 features. Directional features use transparent regex rules rather than a validated financial sentiment model. They can mishandle negation, quotations, expectations, USD movements against other currencies, and retrospective statements. Policy cues can also conflate discussion of the Fed with other central banks. Full definitions are in the script and [feature_dictionary.json](feature_dictionary.json).

For each feature, signed returns and absolute returns are tested at horizons of 1, 3, and 5 days, both for all trading days and for days with collected articles only. There are 216 future-outcome tests. Same-day results form a separate descriptive family of 72 tests.

Inference uses Pearson r, a circular-shift comparison of the target sequence on the trading calendar, and Benjamini–Hochberg correction across the 216 future-outcome hypotheses as one family. All offsets more than 20 observations away from the original alignment are enumerated. For the news-day subset, the selected calendar days remain fixed while the target shifts. This helps preserve the target's serial structure, although stationarity and comparability across periods remain imperfect assumptions. The p-values are exploratory diagnostics.

### Association with next-day returns, all exchange-rate days

| Proxy | Days with nonzero feature | Pearson r | Circular-shift p | BH q |
|---|---:|---:|---:|---:|
| News count | 211 | +0.0327 | 0.2394 | 0.9309 |
| Count of articles with domestic anchors | 14 | +0.0375 | 0.1766 | 0.9309 |
| Count of articles with macroeconomic anchors | 130 | +0.0449 | 0.1146 | 0.9165 |
| USD direction cues from titles | 10 | −0.0096 | 0.7244 | 0.9672 |
| USD direction cues from opening text | 22 | −0.0304 | 0.2748 | 0.9309 |
| USD direction cues from full text | 42 | −0.0301 | 0.3066 | 0.9309 |
| Fed hawkish-minus-dovish cues | 40 | +0.0020 | 0.9363 | 0.9824 |

Each row uses 1,201 day pairs with available outcomes. The number of nonzero feature days is small, particularly for directional features. Small effects can reflect weak signals, proxies that fail to capture those signals, limited coverage, or a combination of these factors.

One notable nominal finding: the Middle East category's news count correlates with the **magnitude** of next-day returns on the news-day subset, with `r = 0.2554` and `p = 0.0017`. After accounting for all tests, `q = 0.1860`, which is insufficient for a convincing finding. This concerns movement size rather than direction. Russia-Ukraine candidates also involve only 11 nonzero-feature days, and their topic labels have relevance problems.

**No future-outcome test has q < 0.05.** No same-day test meets that threshold either. A lack of significant findings is not proof of an exactly zero relationship, and this audit does not test every possible nonlinear relationship or NLP representation.

Additional uncertainty intervals are in [correlation_intervals.csv](correlation_intervals.csv): a circular block bootstrap with 2,000 repetitions and a block length of 10 days. These are marginal intervals without multiple-testing correction and use resampling assumptions different from the circular-shift null. One interval for the opening-text USD proxy does not cross zero, but its circular-shift p-value and q-value do not support a strong conclusion. Do not select only the most favorable inference result.

### Sensitivity to dates and screening of identified problems

Limited screening removes ten records with later content, three records whose URL dates differ by more than one day, and one exact content duplicate. This leaves 421 articles. Removing date mismatches is a conservative sensitivity check rather than a claim that every URL contains the correct publication date. This screening **does not** remove every weakly relevant article or clean all boilerplate.

| Proxy | Original | News delayed by one trading day | Screened: 421 articles | Screening + delay |
|---|---:|---:|---:|---:|
| News count | +0.0327 | −0.0168 | +0.0359 | −0.0202 |
| Macroeconomic anchors | +0.0449 | −0.0055 | +0.0475 | −0.0101 |
| USD direction from opening text | −0.0304 | +0.0257 | −0.0304 | +0.0258 |
| Fed hawkish-minus-dovish cues | +0.0020 | −0.0036 | +0.0020 | −0.0037 |

All values are correlations with next-day returns. Under the delay scenario, the previous date's features are used for the outcome from today to the following day. The delay is a conservative check rather than a verified timestamp correction.

Effects remain small, and some signs reverse when timing shifts. Splitting the sample also shows instability: the correlation for news count is −0.0065 in 2021–2023 and +0.1160 in 2024–2026. These are effect-size diagnostics without new significance claims, model training, or hyperparameter selection. Details are in [sensitivity_checks.csv](sensitivity_checks.csv).

## 6. Why might data like this produce weak associations?

1. **News selection does not reliably preserve the main topic.** Keywords such as politicians' names, Beijing, inflation, or dollar may appear as background, monetary amounts, or related content. Search labels are not automatically labels of the main article.
2. **An article body is not necessarily one clean document.** Promotions, comments, repetitions, and other news alter word frequencies, topics, and sentiment scores. Later-dated content also violates the historical information sequence.
3. **The sentiment target is unclear.** News that is negative for US consumers, bondholders, gold prices, or humanitarian conditions is not automatically negative for the dollar. “The dollar strengthens against the yen” does not guarantee the same direction against the rupiah.
4. **New shocks are mixed with reactions and commentary.** Many articles explain movements that have already happened, repeat events from earlier days, or offer opinions. A decision matching consensus can be topically relevant while adding little new information for prediction.
5. **Coverage is sparse and clustered.** The dataset is not a consistent daily news stream. Missing observations, duplicate stories, and dominance of a particular period reduce the independent variation available for learning.
6. **Dates do not establish information availability.** Differences between indexing, publication, update dates, and time zones can shift news–return pairings.
7. **USD/IDR is affected by factors outside these two CSV files.** This audit cannot separate news effects from every market/domestic factor or identify a single cause. Nonlinear news relationships and relationships that vary over time have not been tested either.

Points 1–6 have direct evidence in the data or diagnostics above. Their individual contributions to previous model scores have not been measured; that would require controlled comparison experiments.

## 7. Recommended next steps

Prioritize dataset improvements before increasing model size:

1. **Audit the main article text first.** Separate main content from related stories, comments, navigation, and promotions. For updated pages, cleaning alone may not recover the historical version. Obtain an as-of archive or mark the text as unreliable for backtesting.
2. **Record temporal provenance.** Distinguish `published_at`, `updated_at`, time zones, retrieval time, and `available_at`. Where timestamps are unknown, state the uncertainty and use several lag scenarios.
3. **Label relevance and mechanisms in the main text.** Distinguish rupiah/BI/Indonesian economic news, global USD macroeconomic news, and geopolitics with clear transmission channels. Correct event labels rather than merely expanding keyword lists.
4. **Have a human reviewer reassess the sample.** Use these 66 articles as a starting point, read the full text, resolve ambiguous cases, and expand sampling by year/source. Do not treat the current AI labels as final ground truth.
5. **Deduplicate stories and improve collection coverage.** Multiple articles about one event are not independent shocks. Establish consistent collection windows, document days when article retrieval fails, and add domestic news outside the September 2022 concentration.
6. **Extract features with explicit targets.** Examples include changes in Fed expectations, surprises relative to consensus, USD direction, rupiah direction, energy pressure, and capital flows. Distinguish announcements, expectations, and retrospective reactions. Compare titles, leads, and cleaned main bodies.
7. **Evaluate incremental value chronologically.** Compare a baseline without news against the same baseline plus news features over the same evaluation period. Use a horizon-appropriate embargo and avoid using test results to select filters or models. This correlation audit is not an out-of-sample prediction validation.

Recommended working approach: retain the two original CSV files as audit inputs and save newly cleaned data as a separate, comparable version. Increasing model complexity alone does not address the identified topic, timing, and text mismatches.

## 8. Reproduction and supporting files

Environment: Python 3.12, pandas 2.2.3, and numpy 2.3.5 from the bundled runtime. No new packages were installed. The script depends on the two processed CSV files and `manual_annotations.json`; the notebook is not a dependency.

In an environment with numpy/pandas:

```bash
python3 deira-work/sanity-check/audit_news.py
```

In this workspace:

```bash
PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 /home/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python deira-work/sanity-check/audit_news.py
```

Input SHA-256 hashes are recorded in [audit_metrics.json](audit_metrics.json). Annotations are tied to this input version and its indices. If the input changes, review the sample and annotations again. The final labels and figures in this report were not produced by reading the previous notebook's results.

| File | Purpose |
|---|---|
| `sample_review_66.csv` | Complete sample, annotations, reasons, full text, sampling weights, and paired returns |
| `manual_annotations.json` | Reviewer annotations as an editable audit input |
| `article_quality_flags.csv` | Quality indicators for all 435 articles |
| `future_date_flags.csv`, `future_date_review.csv` | Evidence and interpretation of later dates |
| `duplicate_candidates.csv` | Candidate duplicate pairs, including one pair with identical content |
| `coverage_by_year.csv`, `monthly_coverage.csv`, `event_by_year.csv` | Sparse coverage and its concentration |
| `daily_features_and_returns.csv` | One record per exchange-rate day, with features and outcomes |
| `feature_dictionary.json` | Feature definitions and numbered event mappings |
| `correlation_tests.csv` | All 288 results, including the descriptive same-day family |
| `correlation_intervals.csv` | Marginal block-bootstrap intervals for six main proxies |
| `sensitivity_checks.csv` | Effects of delay, screening, and period splits |
| `audit_metrics.json` | Summary figures and input hashes |
| `audit_news.py` | Reproducible audit script |
