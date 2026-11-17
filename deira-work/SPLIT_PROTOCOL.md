# Shared temporal split: handoff for all ablation experiments

Recorded on 2026-11-10. Read this before implementing or resuming an experiment after a context reset.

**All Level 1, Level 2, and Level 3 experiments must reuse the same saved decision dates, labels, and fold assignments.** This includes market-only, text-only, availability controls, TF-IDF, Word2Vec, transformers, and all A-series variants. Load the manifests below instead of independently generating a split inside each model notebook.

The current design is a chronological final holdout with three expanding-window validation folds inside development. No model training or representation selection has been completed by the preprocessing notebook.

## 1. Authoritative files

Current artifact directory, relative to the repository root:

```text
deira-work/preprocessing/outputs/date_delay_sensitivity__audit_only/
```

| File | How to use it |
|---|---|
| [calendar_partitions.csv](preprocessing/outputs/date_delay_sensitivity__audit_only/calendar_partitions.csv) | Full fixing calendar, development/test assignment, boundary purge, and eligibility flag |
| [fold_manifest.csv](preprocessing/outputs/date_delay_sensitivity__audit_only/fold_manifest.csv) | Exact eligible train/validation decision dates and labels for each development fold |
| [fold_summary.csv](preprocessing/outputs/date_delay_sensitivity__audit_only/fold_summary.csv) | Human-readable development-fold dates and counts |
| [final_fit_manifest.csv](preprocessing/outputs/date_delay_sensitivity__audit_only/final_fit_manifest.csv) | Exact eligible dates for final development training and final test evaluation |
| [feature_contract.json](preprocessing/outputs/date_delay_sensitivity__audit_only/feature_contract.json) | Predictor whitelist, target exclusions, and representation input fields |
| [pipeline_manifest.json](preprocessing/outputs/date_delay_sensitivity__audit_only/pipeline_manifest.json) | Input/code hashes, preprocessing configuration, versions, and limitations |

The generator is `temporal_manifests()` in [preprocessing_utils.py](preprocessing/preprocessing_utils.py), called by [01_preprocessing.ipynb](preprocessing/01_preprocessing.ipynb). Existing manifests are the experiment inputs. Date ranges in this document are summaries; reconstructing membership from date ranges alone would miss eligibility masks and boundary purges.

## 2. Calendar, decision date, and label

The master calendar contains the 1,202 actual JISDOR observations from 2021-09-01 through 2026-09-01. Do not interpolate weekends or split by article count.

- `Date` is the model's decision-anchor date.
- `target_date` is the next observation on the original JISDOR calendar.
- `y = 1` when the next fixing is higher: USD/IDR UP, meaning IDR depreciation.
- `y = 0` when the next fixing is lower: USD/IDR DOWN.
- Exact FLAT pairs and unavailable next fixings have missing labels.

Compute next-fixing labels and historical market features before masking FLAT labels or removing feature-warmup rows from supervised samples. Keep those rows in the master calendar. The current common supervised eligibility produces 1,168 rows before the final development/test boundary purge.

## 3. Final holdout

The final 20% is reserved by full-calendar row index, before supervised eligibility masking. With 1,202 observations, `test_start = int(1202 * 0.8) = 961` using zero-based indices. One adjacent boundary row is excluded from fitting and selection.

| Partition | Calendar indices | Calendar dates | Calendar rows |
|---|---|---|---:|
| Development | 0–959 | 2021-09-01–2025-08-27 | 960 |
| Boundary purge | 960 | 2025-08-28 | 1 |
| Final test | 961–1201 | 2025-08-29–2026-09-01 | 241 |

After common eligibility masking, `final_fit_manifest.csv` contains:

| Role | First eligible decision | Last eligible decision | Eligible rows |
|---|---|---|---:|
| train | 2021-09-29 | 2025-08-27 | 932 |
| test | 2025-08-29 | 2026-08-31 | 235 |

The final calendar date, 2026-09-01, has no observed next-fixing label. It remains in the calendar but is not an evaluable test sample.

## 4. Expanding validation inside development

Training grows forward in time. Validation periods are chronological, and each fold refits its learned preprocessing using only that fold's training rows.

| Fold | Eligible train decisions | Train rows | Eligible validation decisions | Validation rows |
|---|---|---:|---|---:|
| 1 | 2021-09-29–2023-08-15 | 453 | 2023-08-21–2024-04-19 | 158 |
| 2 | 2021-09-29–2024-04-19 | 612 | 2024-04-23–2024-12-12 | 159 |
| 3 | 2021-09-29–2024-12-12 | 772 | 2024-12-16–2025-08-26 | 158 |

The current generator uses development validation boundaries `[480, 640, 800, 960]` on the full calendar. For a validation interval `[start, end)`, training candidates have `calendar_index < start - 1`; validation candidates have `start <= calendar_index < end - 1`. Both then apply the same `eligible_binary` mask. This conservative boundary treatment ensures that no retained label reaches the following period's cutoff.

Required label-boundary checks:

```text
max(train.target_date) < min(validation.Date)   # every development fold
max(final_train.target_date) < min(test.Date)   # final fit
```

Historical feature context may use already-observed earlier prices across a boundary. Future labels may not be used as training information. Any downstream market sequence must remain causal and preserve the shared evaluation dates.

## 5. Shared use across ablation levels

1. Load `fold_manifest.csv` for development comparisons and join features by decision `Date`. Join articles by `anchor_date`, matching that same decision date. Preserve manifest order or sort dates consistently before comparing predictions.
2. Reuse the same article selection, timing mode, text span, market inputs, and eligibility across representation and component comparisons. Retain days without news in every variant. An all-OOV article or empty window needs a documented fallback/mask; it must not independently remove dates from one model's evaluation.
3. Fit TF-IDF vocabulary/IDF, SVD, scalers, class weights, learned adapters, and any fine-tuning only on each fold's training subset. Validation and test only receive transformed inputs. The preprocessing notebook has not fitted these transforms.
4. Select settings using development folds. The ablation protocol specifies mean validation MCC, with macro-F1 as the tie-breaker for representation selection. Record the best overall representation and the best Level 3-eligible representation separately.
5. After settings are frozen, use the `train` role in `final_fit_manifest.csv` to refit preprocessing and the final model on all eligible development rows. Determine training duration and thresholds using development evidence. Use the `test` role for final reporting across the prespecified variants.
6. Keep final-test outcomes out of choices about filtering, representation, architecture, thresholds, early stopping, and training budget. Save predictions with `Date`, `target_date`, `y`, probability, model ID, and seed so comparisons remain paired on identical outcomes.

Training-majority and momentum baselines use these same validation/test rows. Estimate the majority class from the relevant training subset; the momentum baseline uses the most recently observed direction at the decision time, with a prespecified fallback for a zero previous return.

## 6. Loading example

Run from the repository root. This selects membership only; model fitting belongs in the experiment notebook.

```python
from pathlib import Path
import pandas as pd

base = Path("deira-work/preprocessing/outputs/date_delay_sensitivity__audit_only")
folds = pd.read_csv(base / "fold_manifest.csv", parse_dates=["Date", "target_date"])
final = pd.read_csv(base / "final_fit_manifest.csv", parse_dates=["Date", "target_date"])

fold_id = 1
train_rows = folds.loc[(folds["fold"] == fold_id) & folds["role"].eq("train")].copy()
val_rows = folds.loc[(folds["fold"] == fold_id) & folds["role"].eq("validation")].copy()
assert train_rows["target_date"].max() < val_rows["Date"].min()

final_train_rows = final.loc[final["role"].eq("train")].copy()
test_rows = final.loc[final["role"].eq("test")].copy()
assert final_train_rows["target_date"].max() < test_rows["Date"].min()

# After joining model inputs, verify that Date, target_date, y, and row counts
# exactly match these selected manifest rows. Never silently drop missing inputs.
```

## 7. Snapshot and timing limitations

The requested text path was `data/preprocessed/news_clean.csv`; that file was absent when these artifacts were generated. The notebook explicitly resolved to `data/processed/news_clean.csv`. JISDOR comes from `data/processed/usd_idr_jisdor_clean.csv`. Input hashes are recorded in `pipeline_manifest.json`.

The saved preprocessing configuration is `date_delay_sensitivity` with geopolitical policy `audit_only`. News dates map to the first fixing with a strictly later date. Publication timestamps and historical body availability are not verified, and keyword triage does not establish a human-validated geopolitical corpus. These are limitations of the current text inputs; the temporal split does not resolve them.

Treat this snapshot's manifests as shared, fixed experiment inputs. If source data, common feature eligibility, or the label policy changes, version the affected artifacts and update this document explicitly. Rebuild shared manifests consistently before comparing any affected models. A corpus/timing change that leaves the price calendar and eligibility intact should preserve the existing `(fold, role, Date, target_date, y)` assignments; news counts may change and must be recorded consistently across variants.

Related references: [ablation protocol](rules/ablation_protocol.md), [binary classification protocol](rules/binary_classification_protocol.md), and [preprocessing README](preprocessing/README.md). Earlier notebooks are not the reference implementation for this split.
