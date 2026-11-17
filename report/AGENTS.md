# Report writing instructions

Read this file before writing or revising anything in `report/`, including after a context reset. Use the current manuscript and experiment artifacts, not remembered chat details. Write the report in English. Keep it compact, specific, and readable.

## Fixed requirements

- Maximum **five pages total**, including title, authors, figures, tables, and references.
- IEEEtran conference layout, two columns, 10 pt body text, A4 paper, and the user's one-inch margins. These margins override IEEE defaults.
- Preserve the approved title: **Evaluating Text Representations and Multimodal Fusion for USD/IDR Exchange Rate Direction Prediction**.
- Preserve the authors and order:
  - Deira Aisya Refani — 24/532821/PA/22539
  - Nareswari Ayu Prabowo — 24/532991/PA/22558
  - Yohana Butar Butar — 24/546690/PA/23212
- `main.tex` holds the title, authors, and section inputs. Write section content in `chapters/`. Keep figure environments in `figures/`, image files in `figures/assets/`, and table environments in `tables/`.
- Do not add a separate Related Work section. Integrate essential prior studies briefly into the Introduction.
- Do not add an Abstract or Index Terms. Start directly with the Introduction after the authors.
- Preserve the compact spacing below the authors configured in `preamble.tex`.
- Use `\input`, without forced chapter page breaks. Preserve the page-limit check in `preamble.tex` and `build.sh`.

## Natural academic English

- Give each paragraph a clear point, supporting evidence, and a reason the evidence matters. Check its connection to neighboring paragraphs.
- Use concrete subjects and direct verbs. Prefer “use,” “show,” and “compare” over inflated alternatives. Explain necessary technical terms at first use and keep their names consistent.
- Prefer periods to semicolons. Use a semicolon only when it makes a sentence clearer. Avoid repeated em dashes, parenthetical detours, and long noun chains.
- Avoid stock wording such as “delve,” “leverage,” “utilize,” “foster,” “tapestry,” “multifaceted,” “groundbreaking,” “pivotal,” “seamless,” and “robust” without a specific, supported meaning.
- Cut empty frames such as “it is worth noting,” “in today's rapidly evolving landscape,” “plays a crucial role,” “a testament to,” and “this underscores the importance.”
- Do not start every sentence with “Furthermore,” “Moreover,” “Additionally,” or “These findings.” State the actual connection. Avoid repetitive sentence templates, promotional claims, and artificial synonym rotation.
- Keep necessary technical words even if they are common in AI-generated text. Judge clarity in context. Do not add errors or casual filler to make prose sound human.

## Content and space budget

Use these allocations as planning estimates, without inserting page breaks:

| Material | Approximate space | Include |
| --- | --- | --- |
| Title, authors, introduction | 1 page | Problem, objective, scope, and concise contribution |
| Methodology | 1.5 pages | Data, labels, timing, representations, fusion, evaluation |
| Results and discussion | 1.5 pages | Main comparisons, useful ablations, interpretation, limitations |
| Conclusion and references | 1 page | Supported answer and essential cited sources |

Aim for an 80–120-word conclusion if the evidence fits. This is a project target, not an IEEE rule. Answer the objective without introducing new results.

Report a comparison's model, baseline, metric, split, and relevant uncertainty. Put detailed scores in a compact table and discuss only the findings needed to answer the research question. Include a figure only when it explains something more efficiently than prose. Keep captions informative and readable at the final column width.

When space is tight, remove repeated background, textbook definitions, duplicated scores, and unnecessary experiment detail first. Preserve essential methods, comparators, citations, and limitations. Do not shrink fonts, margins, or figures to unreadable sizes.

## Evidence and project context

- Read `../deira-work/SPLIT_PROTOCOL.md` and `../deira-work/FINAL_TEST_STATUS.md` before describing evaluation or test access. Follow their links to current manifests, frozen configurations, and result artifacts.
- Treat development comparisons, final-test results, and exploratory sanity checks as distinct evidence. Do not call the holdout untouched without checking its current status.
- Copy numbers only from verified artifacts. Check labels, dates, sample counts, metrics, model names, and settings. Never invent results, references, significance, causal explanations, or novelty claims.
- Keep direction classification distinct from predicting exchange-rate values. Keep an observed score improvement distinct from statistical significance or trading usefulness.
- Verify each citation against the source's actual claim. Use IEEE numbered citations. Missing evidence stays a marked draft TODO, never a plausible-sounding replacement.

## Resume and finish

1. Read this guide, `main.tex`, the relevant chapter, and its evidence sources. Inspect current files before editing. Preserve the author's existing work.
2. Make the requested change. Recheck the affected introduction, conclusion, captions, and references when findings change.
3. Run `bash build.sh` from `report/` after manuscript edits. Inspect the compiled PDF for clipping, float placement, legibility, and page count. A failed build can leave an older `main.pdf` in place.
4. If the report exceeds five pages, shorten it and rebuild. Do not truncate content or disable the limit.
5. Before pausing substantial report work, update the resume note below with completed files, unresolved evidence, and the next concrete task.

## Resume note

Updated on 2026-11-17: the full-paper audit fixes are applied and the architecture figure is in place (Fig. 2, `figures/model_architecture.tex`, source `figures/drawio/architecture.drawio`, export `figures/assets/model_architecture.pdf`). The build is 5 pages, so page 5 has no free space left. The figure was checked against `src/task_3/training/attention.py`: slots are concatenated, inactive slots are zero vectors, and the gate reads the 14 market inputs and the log article count. Results rely on `improved_v1` artifacts (Level 3 uses FinBERT); `CORPUS_PROTOCOL.md` still names Word2Vec and is older. Remaining optional audit items: define MCC, JISDOR, RSI, PCA at first use; shorten repeated test-access remarks; add a contribution sentence to the Introduction. Next task: after any text change, re-run `bash build.sh` and check the page count.

## Writing reference

Based on [WenyuChiou/academic-writing-skills](https://github.com/WenyuChiou/academic-writing-skills), especially its [prose and citation guide](https://github.com/WenyuChiou/academic-writing-skills/blob/main/skills/academic-writing-skills/references/prose-and-citation-editing.md) and [abstract and conclusion guide](https://github.com/WenyuChiou/academic-writing-skills/blob/main/skills/academic-writing-skills/references/conclusion-and-abstract.md). The five-page budget, formatting decisions, punctuation preferences, and wording examples above are project-specific instructions.
