# Improved training protocol

This version implements `../improvement_plan.md` without VADER. The original daily UP/DOWN labels, combined original + CNBC corpus, fixing windows and three expanding folds remain fixed. `data/final/metadata.json` defines the data contract; target and endpoint columns are excluded from predictors.

All active notebooks use `models.py`, `attention.py`, `data.py`, `evaluation.py` and `experiment.py`. Earlier modules remain available for loading historical saved models. Previous notebook sources and tables are archived under `output/model/improved_v1/legacy_training/`; original model runs remain in their existing directories.

## Training and selection

- MLP: one 16-unit ReLU layer, float64 Adam, batch32, three seeds42/123/2026, fixed threshold0.5. Shared market/output weights use the same initial states across representations and fusion conditions.
- Checkpoints: minimum trailing-five-epoch validation log loss, min_delta0.001, maximum60 epochs and patience10. The saved checkpoint is the current epoch when the smoothed criterion improves; warmup is five epochs.
- The previously verified validation search over alpha0.01/0.1/1, dropout0.2/0.5 and supported centering options supplies each representation's settings. No test-driven search expansion is allowed. All active notebook MLPs are trained again with the promoted implementation.
- Level 1 uses the FinBERT-selected settings identically for market-only, market+availability and market+text+availability, isolating the input ablation. An independently validation-tuned price model is an extra reference.
- Level 2 keeps simple concatenation and the same 16-unit classifier family. Necessary native input widths and validation-selected regularization may differ, so this compares representation pipelines rather than an isolated causal encoder effect. R3 aliases the identical L1-d run. The text-only condition uses the updated overall representation winner and retains news availability.
- L2 logistic regression with C0.01/0.1/1 is a required benchmark on the same fixed inputs in Level 1–2. Only configurations converged on all three folds enter its comparison table. Failures are recorded rather than silently accepted.
- Select the overall and Level 3-eligible representations by minimum mean validation log loss, averaging seeds within folds and then folds equally. MCC remains a reported classification outcome, not the checkpoint criterion.
- Level 3 runs A000, A100, A010, A001 and A111 with one updated eligible representation and inherited regularization. Query and cross-attention read article embeddings in parallel. Fixed zero slots preserve the classifier interface. Gating scales the text slots; the availability mask is always present. The three optional interaction variants are outside this version, and no inside-the-complete-model interaction attribution is claimed.

## Preprocessing and diagnostics

TF-IDF/IDF/SVD fit only training articles. Pretrained Word2Vec, FinBERT and RoBERTa article vectors are frozen and reused from checked caches. Scalers and optional PCA fit only the current training partition. News-present training means define text scales; empty windows stay exactly zero, including when centering is enabled.

Training-only cosine similarity, first-component variance and embedding offset diagnostics accompany text conditions. Previously verified R2/R3 block-permutation diagnostics are linked with hashes; they are descriptive conditional references, not formal p-values. Earlier regularized models did not pass the prespecified prior gate, so PCA, scalar sentiment and rolling-delta branches are not expanded merely to search for a higher score. High embedding similarity does not establish absence of label signal.

## Evaluation and final test

The success gate requires log loss below the training-prior reference in at least two validation folds and a paired temporal-bootstrap upper95% bound below zero. Text attribution also requires matched market-only and availability controls. All planned levels continue even when this gate fails; failure limits the claims.

Complete all validation conditions first. Freeze every condition's configuration, representation, source hashes and per-seed median epochs. Refit all fifteen primary notebook conditions on all development decisions, without scaling the epoch duration. Also retain the existing matched market-only, independently tuned market-only and matched availability controls. Each notebook reports its own test results. No threshold selection, class-weight search or representation choice uses final-test labels.

The user explicitly expanded the earlier selected-only test scope to every ablation. `all_conditions_test_v1/frozen_config.json` records this amendment and preserves the earlier sources and notebooks. The original validation choices and models remain unchanged. Identical already verified refits are reused; R3 aliases the identical L1-d pipeline. These aliases are not independent evidence. Test results provide a descriptive comparison and do not replace the validation-selected representation or model.

Final test has already been accessed repeatedly in earlier work. This entire update is exploratory; do not describe the period as a new untouched holdout or make uncorrected confirmatory claims. The final comparison is run to complete the user's requested pipeline even if the validation gate fails, and that failure remains visible.

Report MCC, macro-F1, balanced accuracy, accuracy, ROC-AUC, Brier, log loss, ECE-10, per-class metrics, confusion matrices and temporal intervals. Average individual seed metrics without ensembling. Validation uncertainty uses paired blocks of ten decisions within each fold; test uses the same paired blocks on final dates. Intervals are conditional on fitted/selected models and omit selection uncertainty.

One writer records all levels and phases. Sample counts come from actual frames and fit records; test use for fitting/stopping is derived from evaluation scope and fixed-duration records. Data, configuration, encoder and model hashes accompany the outputs. Model reloads and independent probability reconstruction verify saved predictions.

Date-only news timestamps, historical article-body versions and historical JISDOR availability retain their documented limitations. The earlier audit did not establish checkpoint optimism as the sole cause of weak generalization.
