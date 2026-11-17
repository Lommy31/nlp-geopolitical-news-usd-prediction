# Geopolitical News and USD/IDR Direction: Text Representations and Fusion

Course project, UGM NLP, semester 5. Authors: Deira Aisya Refani (24/532821/PA/22539), Nareswari Ayu Prabowo (24/532991/PA/22558), Yohana Butar Butar (24/546690/PA/23212).

The paper is `report/main.pdf` (IEEE format, five pages, source in `report/`). This README explains what the code does, where every number and figure in the paper comes from, and how to read them.

## What the project asks

Given the market history and the news published up to a decision day, will the next JISDOR fixing of USD/IDR be higher (UP) or lower (DOWN) than today's? Three questions:

1. Does news content add information beyond market history and the number of articles?
2. Which text representation works best under simple concatenation?
3. With one representation fixed, do learned-query pooling, cross-attention, or a gate help?

Short answer, as reported in the paper: no. Every model scores near chance on the test period, and the paper spends its error analysis on why.

## Data

| File | Content |
| --- | --- |
| `data/processed/usd_idr_jisdor_clean.csv` | 1,202 JISDOR fixings, 2021-09-01 to 2026-09-01 |
| `data/final/train.csv`, `val.csv`, `test.csv` | One row per decision day: label `y` (1 = next fixing is higher), 14 market features, `n_berita` (article count), `has_news`, and the day's articles |
| `data/final/metadata.json` | Data contract: which columns are predictors and which are targets |

The corpus has 16,723 articles (412 from the original collection, 16,311 from CNBC). Articles are dated without a time of day, so an article only enters the first fixing strictly after its date. The test period is the final 20% of the calendar: 235 decisions from 2025-08-29 to 2026-08-31. The other 930 decisions form three expanding validation folds. The earlier-task scripts in `src/task_1` and `src/task_2` and the code in `src/task_3/preprocessing/` build these files. `src/task_3/preprocessing/` builds the final splits.

## Models (Levels 1 to 3)

All models share a small head: one 16-unit ReLU layer, Adam, batch size 32, seeds 42, 123 and 2026, threshold 0.5. The text encoders are frozen. No language model is trained. Code is in `src/task_3/training/`, and each condition has a notebook.

| Level | Question | Notebooks (`src/task_3/training/level_*/`) |
| --- | --- | --- |
| 1 | Does news add anything beyond market history? | `level_1/01` to `05`: training majority, last direction, market only, market + FinBERT + availability, market + availability |
| 2 | Which representation? | `level_2/01` to `05`: TF-IDF + SVD, Word2Vec, FinBERT, RoBERTa-base, text only |
| 3 | Do pooling, cross-attention or a gate help? | `level_3/01` to `05`: A000 mean pooling, A100 task query, A010 cross-attention, A001 gate, A111 all three |

Level 3 keeps the validation-selected representation (FinBERT). `controlled_fusion.py` implements the slots: mean, task query, and cross-attention vectors are concatenated with the market and availability inputs, and the gate scales the text slots. `PROTOCOL.md` states the selection rules: checkpoints and representations are chosen by validation log loss, not MCC.

The reported run is `improved_v1`. Its results are under `src/task_3/output/model/improved_v1/` (only the small CSV, JSON and Markdown files are tracked; saved models and embedding caches are not). The files the paper reads:

- `all_conditions_test_v1/validation_test_comparison.csv` and `result.md`: the scores in the three paper tables (Tables II to IV).
- `all_conditions_test_v1/paired_contrasts.csv`: the paired block-bootstrap intervals quoted in Results.
- `all_conditions_test_v1/<condition>/predictions.csv`: per-day predictions used by the error figures.
- `representation_selection.json`, `level_*/validation_summary.csv`: validation scores and the FinBERT choice.

## Where each figure and table comes from

Scripts are in `report/eda-visualization/`. Run them from the repository root, for example `python report/eda-visualization/make_eda_figures.py`. Each writes a PDF to `report/figures/assets/` and a PNG next to the script.

| Paper item | Script | Inputs | What it shows and how to read it |
| --- | --- | --- | --- |
| Fig. 1 | `make_eda_figures.py` | `data/final/*.csv`, JISDOR csv | The rate series with the three periods marked, and articles per decision day. Check that the split is chronological and that news coverage is uneven (median 12 per day, maximum 139). |
| Fig. 2 | `figures/drawio/architecture.drawio` (exported to `figures/assets/model_architecture.png`) | none | Level 3 layout: frozen encoder, mean/task/cross slots, gate, market and availability inputs, 16-unit head. Hand-drawn, not generated from data. |
| Fig. 3 | `make_error_figures.py` | `all_conditions_test_v1/A000` and `A111` `predictions.csv` | Share of test days each model predicts UP, per seed. A bar at 1.0 means UP on every day (the majority rule). The dashed line is the true UP share (0.583). Bars far from the line mean the model is not tracking the label. |
| Table V | `make_probe_table.py` (output `probe_auc.csv`) | frozen embeddings, development days | Linear probe AUC from the day-mean embedding. The keyword rows are an easy control (does the embedding carry topic?). The direction row is the real target. Compare it with the shuffled-label 95% range: inside it means no detectable signal. |
| Fig. 4 | `make_error_figures.py` | FinBERT day-mean vectors, development days | PCA of the same vectors, coloured by next-fixing direction. Overlapping colours mean direction is not visible along the main variation of the text. It is a visual check only; the probe is the test. |
| Fig. 5 | `make_error_figures.py` | saved A111 models, test day 2025-09-01 | Attention weights on the largest false-UP day, divided by the uniform weight, so 1 means the same weight as plain averaging. Near-uniform rows mean attention did not pick out particular articles. Descriptive only. |
| Fig. 6 | `make_leadlag.py` | development days only | Correlation of each daily news feature with the return k days later. Positive k means news leads. The grey band is what circularly shifted copies of the news series produce by chance. Bars outside it are the ones worth attention; only 3 of 66 are. |
| Fig. 7 | `make_learning_curve.py` | development days, 158 validation days for scoring | Probe AUC as the training size grows. The keyword control rises, direction stays near 0.5 inside the grey band, so more data does not reveal a direction signal in these features. |

Other PNG and PDF files in those folders (`err_probability`, `err_attention_concentration`, `pipeline*`) are earlier drafts that the paper does not use.

The scripts for Fig. 4, Fig. 5, Table V and Fig. 7 need the embedding caches in `src/task_3/output/model/encoders/`. These are not tracked because of their size. Running the Level 2 notebooks recreates them. The committed PDFs in `report/figures/assets/` are the figures as printed in the paper.

## Rebuilding the paper

```bash
cd report
bash build.sh        # needs latexmk; fails if the PDF is longer than five pages
```

`report/AGENTS.md` lists the writing and layout rules used for the paper.

## Repository layout

```
data/                    raw, processed and final datasets
src/task_1, task_2/      scraping, cleaning, alignment
src/task_3/
  preprocessing/         final train/val/test construction
  training/              models, protocol, Level 1-3 notebooks
  sanity_check/          validation and test checks, including the improved_v1 final test
  output/                results (mostly untracked)
report/                  LaTeX source, figures, tables, figure scripts, main.pdf
deira-work/, nars-work/, hana-work/   individual members' working folders
```
