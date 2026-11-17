# Proposed binary USD/IDR classification protocol

Status: the user has fixed the output to **UP/DOWN classification of USD/IDR using geopolitical news**. The protocol below is a recommended implementation specification; model training has not started. Strict forecasting is recommended as the primary experiment, with explanatory analysis reported separately if used.

This supplements [preprocessing_rules.md](preprocessing_rules.md). Detailed model recommendations are in [the paper adaptation review](../literature-review/dakalbab_2025_usdidr_binary_review.md).

## 1. Target and sign convention

Let `J_t` be the JISDOR value at the current available observation and `J_next` the next observation on the original JISDOR calendar:

```text
r_next = log(J_next / J_t)
y = 1 (UP)   if J_next > J_t
y = 0 (DOWN) if J_next < J_t
y = missing if J_next == J_t or J_next is unavailable
```

UP means a higher USD/IDR quote: USD strengthens against IDR, or IDR weakens against USD. DOWN means the opposite. A positive sentiment label is not an UP market label.

Keep exact equality as a separate diagnostic status rather than silently assigning it to DOWN. Exclude equal-price pairs from supervised loss and binary evaluation, not from the master price calendar. Calculate features and adjacent-fixing returns **before** masking these labels. Otherwise, deleting a fixing changes the meaning of the next-fixing target.

Local diagnostics before feature warmup or other screening: 1,202 fixing rows; 1,201 adjacent pairs; 665 UP, 523 DOWN, and 13 FLAT. This leaves 1,188 nonflat pairs at most before further exclusions. Class counts must be recalculated within every split. The global UP share of approximately 55.98% is descriptive, not a valid fitted baseline.

Do not search for a profitable return threshold on the full dataset. The primary task uses the sign of nonzero returns. A prespecified small-movement exclusion can be a secondary sensitivity study with its retained-sample coverage reported; it changes the evaluation population.

## 2. Prediction time and news windows

Maintain separate times for the JISDOR observation cutoff, its publication/availability, and the model's decision. If `J_t` is an input, decision time `c_t` must be at or after its actual availability.

```text
Primary forecast news window: c_previous < available_at <= c_t
Primary outcome:             log(J_next / J_t)
Required condition:          every input is available by c_t
```

Convert verified publication timestamps to `Asia/Jakarta`. Record the source time zone and whether the article's stored text version existed at that time. An old publication date attached to a later-updated body does not establish historical availability.

The processed news CSV currently has dates, not verified publication times. Do not invent midnight timestamps and describe them as observed. Until times are recovered, label a delayed date-only experiment as a timing sensitivity analysis. A verified forecast claim requires a verified information cutoff or an explicitly justified conservative availability bound.

Use a versioned JISDOR schedule instead of assuming one hour for 2021–2026. Bank Indonesia's [current specification](https://www.bi.go.id/id/fungsi-utama/moneter/informasi-kurs/default.aspx) distinguishes transaction cutoff from publication and allows schedule adjustments. Its [2021 announcement](https://www.bi.go.id/id/publikasi/ruang-media/news-release/Pages/sp_233021.aspx) also describes a pandemic adjustment. The complete historical schedule is not established by this review.

An explanatory experiment may use news arriving after `c_t` and up to the next reference observation to analyze the same return. Specify its interval boundaries precisely, retain a separate dataset and result table, and do not call it a forecast. Explanatory performance is not guaranteed to exceed forecasting performance.

## 3. Calendar and gap features

- Keep one row per actual JISDOR observation, including rows whose classification label is masked.
- Calculate historical price features on that index, without calendar-day interpolation.
- `input_gap_days` measures the elapsed interval since the previous observed fixing and is known at decision time.
- `scheduled_target_gap_days` may be a predictor only when the next publication schedule was known in advance.
- `realized_target_gap_days` is always available for retrospective evaluation groups. Do not automatically use it as an ex-ante feature when a gap could reflect unexpected nonpublication or missing source data.
- Report performance by the **target endpoint's weekday** and gap group, so a Friday-to-Monday label is categorized correctly even though the decision row is Friday.

Longer intervals may have different return dispersion; larger variance is an empirical possibility, not a mathematical guarantee. Compare gap groups rather than asserting the difference without checking.

## 4. Empty and sparse news inputs

Preserve `n_berita`, `has_news`, and input-window duration. Zero text features plus `has_news = 0` represent missing collected news, not an observed neutral sentiment.

Do not carry forward an old sentiment score into a new empty window. An explicitly modeled historical news sequence is a separate design choice and must retain original ages and masks.

For attention, mask padded elements. If a sample has no valid keys, bypass news attention and use the price path, with a zero news contribution. Passing an entirely masked row through softmax can produce invalid values. Handle the empty case before attention rather than inventing a sentiment observation.

Use normalized pooling or a mean as the baseline. If using decay weights, compute article age relative to the decision cutoff and tune decay only on development data. Keep article count separate from text intensity.

## 5. Temporal evaluation

Recommended design: reserve the final 20% of the calendar as an untouched test period and use expanding chronological folds within the earlier 80% for model selection. Freeze boundaries before fitting preprocessing or choosing models. Report the realized date ranges and usable class/news counts; percentages alone are insufficient.

Purge training labels whose outcomes are not yet available at the next fold's training cutoff. Use a conservative one-observation boundary gap for the one-fixing horizon, or enforce the rule through explicit label-end timestamps. Sharing already-known historical context across a boundary is not itself leakage; future outcomes are the issue.

Fit scalers, TF-IDF, dimensionality reduction, class weights, supervised relevance models, and any encoder fine-tuning on the appropriate training fold only. Tune thresholds, early stopping, and hyperparameters on development data. Never select an architecture or filter using the final test score.

Keep near-duplicate versions of the same story from inflating cross-split performance. Distinct updates about a prolonged geopolitical conflict are not automatically duplicate stories.

## 6. Three-level comparisons and reporting

The authoritative experiment design is [ablation_protocol.md](ablation_protocol.md). It replaces the earlier mandatory eight-cell grid for each representation.

1. **Level 1: text value.** Compare training-majority, last-direction, market-only, market-plus-text concatenation, and market-plus-news-availability without content. The availability control is required. Compare content against that control as well as against market-only inputs.
2. **Level 2: representation.** Keep simple concatenation fixed and compare TF-IDF + SVD, pretrained Word2Vec, FinBERT, and RoBERTa-base. A predeclared DeBERTa variant is optional. Select using chronological validation MCC, with macro-F1 as a tie-breaker. Report the best overall representation and the best Level 3-eligible representation separately; R1 remains a benchmark but is outside the Level 3 scope. Add a text-only model for the best overall representation.
3. **Level 3: components.** Fix the eligible winner and run A000, A100, A010, A001, and A111. A110, A101, and A011 are extensions and are required for claims about individual components inside the complete model. They are not mandatory for the minimum standalone-component study.

Level 3 uses learned task-query pooling for T, one-way price-query attention over intact article embeddings for C, and a learned scalar gate from price and news count for G. T and C operate in parallel. T=0 uses uniform mean pooling, not a generic semantic query. Keep concatenation and fixed-size text slots throughout; inactive optional slots are zero. A generic-versus-USD/IDR semantic-query test is a separate matched experiment if stronger target-awareness claims are made.

The deterministic availability mask is retained in every variant and is not the learned-gate factor. G=0 sets the learned multiplier to 1; empty news still has zero text contribution. Hold inputs, temporal splits, shared head structure, and training budget fixed within component comparisons.

Primary metric: MCC. Also report macro-F1, balanced accuracy, accuracy, per-class precision/recall/F1, confusion matrices, ROC-AUC where both classes exist, and probability loss/calibration diagnostics. Use paired temporal block resampling and several training seeds for prespecified score differences.

Report all-day and news-available-day results separately, with empty/singleton/multi-article counts. Do not drop empty-news dates from only some models. Neither attention weights nor a higher test score alone establishes causality. No model, including A111, is guaranteed to win.

Document exclusion counts, tokenizer and checkpoint revisions, model availability/pretraining dates where known, random seeds, feature definitions, source hashes, fold boundaries, selected hyperparameters, and both representation winners. A checkpoint released after a historical backtest period requires an explicit retrospective-design caveat.
