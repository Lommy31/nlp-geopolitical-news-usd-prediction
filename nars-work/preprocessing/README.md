# Preprocessing for the three-level NLP ablation

Start with [01_preprocessing.ipynb](01_preprocessing.ipynb). All three text representations use the same cleaned headline and first 180 body words. The notebook prepares inputs and temporal manifests; it does not fit a vocabulary, SVD, scaler, word vectors, or transformer.

## Inputs and execution

The requested text path is `data/preprocessed/news_clean.csv`. That path does not exist in the current snapshot. The notebook explicitly reports and uses `data/processed/news_clean.csv` when the requested path is absent. If both exist, it uses the requested path. The resolved path and SHA-256 are saved in the manifest. No raw-news input is read. JISDOR comes from `data/processed/usd_idr_jisdor_clean.csv`.

Requirements: Python 3.10+, NumPy, and pandas. Run all cells from the project root or the notebook directory in Jupyter. Alternatively, from the project root:

```bash
python deira-work/preprocessing/run_notebook.py
python -m unittest discover -s deira-work/preprocessing -p 'test_*.py'
```

The runner executes ordinary Python cells sequentially and saves stream outputs. It does not validate the Jupyter kernel protocol. Rerunning replaces generated artifacts for the same configuration. Human review files in `reviews/` are never overwritten.

## Text preparation

- Shared: Unicode/HTML normalization, conservative paragraph/footer removal, exact content deduplication, and a fixed headline/lead span. The original CSV is unchanged. Cleaning cannot restore the historical version of a scraped article.
- TF-IDF: lowercase tokens, retaining negation, entities, and numbers; no stopword removal or stemming. Later use `TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)` on `text_tfidf`, fit on training articles only; fit SVD on training vectors, then transform validation/test. Aggregate article vectors with a mean per window.
- Word2Vec: case-preserving tokens by default, configurable to match the chosen pretrained vocabulary. Use pretrained vectors, report vocabulary coverage/OOV rates, average in-vocabulary word vectors per article, then average articles per window. An all-OOV article needs a zero-vector fallback plus an explicit coverage diagnostic; it is not an empty collected-news window.
- Transformer: shared text with natural casing and punctuation; use the checkpoint's own tokenizer and record token limits/truncation. The 180-word span is shared, but wordpiece lengths can differ. Do not feed lowercased static tokens to the transformer.

## Screening and timing limits

`GEOPOLITICAL_POLICY = "audit_only"` is the default: keyword triage is reported for review and does not automatically discard macro/policy stories. This is a quality-screened collected-news corpus, **not a human-validated geopolitical corpus**. `rule_screened` is an exploratory rule-based subset; `human_approved` requires reviewed inclusions. Fix the corpus policy across representations before training, and do not select it using test outcomes.

Confirmed later-content records from the existing sanity audit are quarantined only if its input SHA-256 matches exactly. Short/duplicate records are excluded; uncertain date/encoding flags require review. A future date mention alone is not proof of leakage. Use the review queue to resolve legitimate forward-looking references. Any later audit annotation is retrospective data curation, not evidence of historical availability.

The available news input has dates only. The runnable default, `date_delay_sensitivity`, assigns each article to the first fixing with a strictly later date. Saturday/Sunday enter Monday when available; Monday-dated news enters Tuesday. This deliberately discards same-date availability and assumes the source date's calendar interpretation. It cannot verify the source timezone, body version, or news availability, so results must not be labeled verified strict forecasting.

For `strict_forecast`, populate `reviews/article_review.csv` with verified availability times, body-as-of evidence, and timezone information; also supply a complete verified historical schedule in `reviews/jisdor_availability.csv`. Templates are exported under the output directory. Decision times must be at/after JISDOR publication. Unverified articles are never aligned in strict mode. If no trusted articles remain, the notebook stops rather than exporting an apparently usable strict text dataset. No historical fixing hour is invented.

## Outputs and downstream use

Artifacts are written to `outputs/<timing_mode>__<geopolitical_policy>/`:

| Artifact | Purpose |
|---|---|
| `article_views.jsonl` | All records, prepared text views, cleaning/review/selection provenance |
| `aligned_articles.jsonl` | Retained articles with a decision anchor |
| `review_queue.csv`, `excluded_articles.csv` | Flags and explicit selection/alignment reasons |
| `article_review_template.csv` | IDs and blank human review fields; copy to `reviews/article_review.csv` |
| `jisdor_availability_template.csv` | Blank historical publication/decision schedule; copy to `reviews/jisdor_availability.csv` |
| `daily_market.csv` | Full fixing calendar, causal market features, targets, and availability counts |
| `news_windows.jsonl` | Article lists and representation inputs per fixing, including empty lists |
| `calendar_partitions.csv`, `fold_manifest.csv`, `fold_summary.csv`, `final_fit_manifest.csv` | Fixed chronological development/test split and purged expanding folds |
| `feature_contract.json` | Explicit predictor whitelist and target/diagnostic exclusions |
| `pipeline_manifest.json` | Configuration, input/code hashes, versions, and coverage |

`y = 1` means USD/IDR UP (rupiah depreciation); `y = 0` means DOWN. Exact FLAT pairs and the final unavailable label are masked **after** all adjacent-fixing labels and market features are calculated. Warmup rows remain in the master calendar. The final 20% is reserved by calendar index; boundary rows are conservatively purged. Missing-news days remain in every experiment.

Use `fold_manifest.csv` to select training decision dates, then select articles by `anchor_date`. Never fit text transformations on every row of `article_views.jsonl`. Fit the market scaler only on that fold's training rows and the explicit feature whitelist. Target endpoint, next price, future return, and realized future gap are never predictors. `input_gap_days` is known history; in strict mode actual decision-interval duration is also exported for diagnostics, but is not a default predictor.

For empty news, create zero text vectors plus counts/masks downstream. Bypass attention for all-empty samples before softmax; pad and mask variable-length article sequences only during batching. Keep article embeddings separate for Level 3 attention. No sentiment, embedding, learned relevance score, or fabricated timestamp is generated here.
