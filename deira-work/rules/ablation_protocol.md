# Three-level ablation protocol for geopolitical news and USD/IDR

Status: proposed study design; no models have been trained under this protocol. The fixed task is binary next-fixing USD/IDR classification. This design replaces the earlier requirement to run all eight component variants for every representation.

Use [binary_classification_protocol.md](binary_classification_protocol.md) for labels, fixing calendars, information availability, and temporal splits. The [paper adaptation review](../literature-review/dakalbab_2025_usdidr_binary_review.md) provides research context; this document is authoritative for the experiment matrix.

## 1. Research questions and scope

The study separates three questions:

1. **Text value:** does news content add information beyond market history and news availability?
2. **Representation:** which text representation works best under a fixed, simple fusion setup?
3. **Components:** with one eligible representation fixed, do learned-query pooling, price-conditioned cross-attention, and learned gating improve the model?

Level 2 reports all required representation candidates. Level 3 uses one validation-selected eligible winner, not a three-representation-by-eight-variant grid. Failure to establish a benefit at an earlier level does not automatically cancel later levels, but limits the resulting claims.

## 2. Shared experimental contract

- Predict the sign of `log(J_next / J_t)`. UP means a higher USD/IDR quote and rupiah depreciation. Mask exact FLAT labels without removing their fixing observations from feature calculations.
- Use identical eligible outcome dates, verified information cutoffs, article selection, geopolitical relevance rules, and duplicate/version treatment across comparisons.
- Use the same ex-ante market inputs. Macroeconomic features, if included, must be available by the decision cutoff and retained consistently across variants.
- Reserve the final 20% of the JISDOR calendar as a locked test period. Use expanding chronological folds within the earlier period for selection and boundary purging based on label availability.
- Fit vocabulary, IDF, SVD, scalers, class weights, and other learned preprocessing within the appropriate training fold. Use pretrained encoders/word vectors frozen initially; record checkpoint revisions and pretraining/availability limits.
- Primary metric: **MCC**. Report macro-F1, balanced accuracy, accuracy, per-class metrics, confusion matrices, and probability metrics where defined.
- Use a fixed selection budget, paired seeds, common initial states for shared layers, and identical outcome rows. Report parameter counts, runtime, and coverage differences.
- Preserve empty-news days. `has_news = 0` means no retained available article, not observed neutral sentiment. Never carry forward sentiment into an empty current window.

The final-test labels must not select the representation, architecture, query wording, article filter, decision threshold, or training settings.

## 3. Level 1: Does text help?

Use simple concatenation and a fixed baseline representation: frozen FinBERT with uniform mean pooling. Keep the market branch and classifier family consistent across trainable models.

| ID | Model | Purpose |
|---|---|---|
| L1-a | Training-period majority class | Naive reference; not a mathematical lower bound |
| L1-b | Repeat the most recent observed direction | Momentum/persistence reference |
| L1-c | Market features only: price and any prespecified macro inputs | Performance without news |
| L1-d | Market + news content + news-availability features, concatenated | Added news information |
| L1-e | Market + `n_berita`/`has_news`, without content | Required control for collection activity |

L1-e is **required**, rather than optional. L1-d and L1-e receive the same news-availability features and base calendar metadata; their difference is access to text content. L1-c excludes news-availability features but retains the same market and base calendar inputs.

Prespecified comparisons:

- `L1-d - L1-c`: total benefit from adding news-related inputs.
- `L1-d - L1-e`: incremental benefit of text content beyond availability/counts.
- `L1-e - L1-c`: benefit associated with news activity alone.

A positive first contrast alone does not establish that linguistic content is useful. If the text baseline does not improve, continue the planned later levels and report the limitation; it does not establish that every possible text representation is useless.

## 4. Level 2: Compare text representations

Hold fusion fixed to concatenation: no learned-query pooling, no cross-attention, and no learned gate. Change only the representation recipe and its necessary input adapter. Keep market inputs, metadata, classifier structure, article text spans, normalized window aggregation, and selection budget consistent.

| ID | Representation | Type | Input to the prediction model | Level 3 eligibility |
|---|---|---|---|---|
| R1 | TF-IDF + SVD | Lexical sparse features reduced to dense vectors | One dense vector per news window | No, by the study's declared scope |
| R2 | Pretrained Word2Vec | Static word embeddings | Mean word vector per article, then window mean | Yes |
| R3 | FinBERT | Contextual, financial domain | Mean-pooled contextual vector per article, then window mean | Yes |
| R4 | RoBERTa-base | Contextual, general domain | Mean-pooled contextual vector per article, then window mean | Yes |

RoBERTa-base is the primary R4 candidate. DeBERTa-v3-base may be an additional candidate only if declared within the development budget before final-test evaluation; do not silently replace R4 using test results.

For R1, vectorize articles, apply fold-local SVD, and use the same normalized article-to-window aggregation. Its final output is a dense window vector. This avoids changing aggregation as well as representation. TF-IDF could technically be retained per article for attention; excluding it from Level 3 is a scope decision, not a claim of mathematical incompatibility.

Use the same cleaned headline/lead spans, with representation-appropriate tokenization. Keep negation, entities, and event meaning; document casing, vocabulary coverage, out-of-vocabulary handling, and any model-specific truncation. Do not train Word2Vec from scratch on the small project corpus as the sole static baseline.

Keep classifier input width consistent using documented adapters, and report their parameter counts. Equal width does not make pretraining corpora or information budgets identical. This comparison evaluates representation pipelines, not an isolated causal effect of architecture alone.

### 4.1 Representation selection

Select on mean validation MCC across the prespecified chronological folds, with macro-F1 as the declared tie-breaker. Report fold variation and all candidate results.

Record two winners separately:

- **Best overall representation:** highest validation score among R1–R4, plus any predeclared additional candidate.
- **Best Level 3-eligible representation:** highest validation score among R2–R4, plus any predeclared eligible addition.

Use the second winner in Level 3. If R1 is the overall winner, retain it as an important benchmark and state why Level 3 uses another representation. Do not hide a stronger TF-IDF result.

Add one **text-only** model using the best overall representation, selected on development data, with market features removed and empty-window handling retained. Compare it on the same outcome dates. If the two winners differ, an eligible-winner text-only model is optional and must be labeled separately.

Hypotheses to test, not assumptions: contextual representations outperform static representations; domain-specific FinBERT outperforms general-domain RoBERTa. Either can fail, particularly for geopolitical text.

If L1-d and R3 are exactly the same configured pipeline, reuse the same fitted predictions and record the alias rather than treating a repeated fit as independent evidence.

## 5. Level 3: Proposed components

Fix the validation-selected eligible representation, price encoder, input windows, and training protocol. IDs encode `(T, C, G)`:

- **T:** a learned query, assigned to the USD/IDR prediction task, pools the available article embeddings.
- **C:** the price representation provides a query; keys and values are the available article embeddings. The primary block is **one-way price-to-news attention**.
- **G:** a learned scalar in `[0, 1]`, computed from price and news count, scales the text contribution.

The mandatory availability mask is present in every variant and is not factor G. T=0 uses uniform article mean pooling as the baseline, not the fixed generic semantic-query control from the superseded design.

### 5.1 Main variants

| ID | T | C | G | Model |
|---|---:|---:|---:|---|
| A000 | 0 | 0 | 0 | Mean-pooled news + market, concatenated |
| A100 | 1 | 0 | 0 | Add learned task-query pooling |
| A010 | 0 | 1 | 0 | Add price-query cross-attention |
| A001 | 0 | 0 | 1 | Add learned gating of the mean-pooled news contribution |
| A111 | 1 | 1 | 1 | All three components |

### 5.2 Extended variants

| ID | T | C | G | Model |
|---|---:|---:|---:|---|
| A110 | 1 | 1 | 0 | Learned-query pooling + cross-attention, without learned gate |
| A101 | 1 | 0 | 1 | Learned-query pooling + learned gate, without cross-attention |
| A011 | 0 | 1 | 1 | Cross-attention + learned gate, without learned-query pooling |

The five main variants are the minimum study. Declare whether the three extensions will be run before inspecting final-test results. The extensions become necessary if the report claims to identify each component's contribution **inside the complete model**. They are not necessary merely to test standalone additions to A000.

Expect A111 to outperform A000 and possibly the single-component variants, but do not impose `A111 >= every alternative` as a guaranteed outcome. Negative or mixed results must be reported.

## 6. Keep the fusion interface controlled

Let `H_N` be the intact sequence of article embeddings and `z_P` the market summary. T and C read `H_N` **in parallel**. Do not first collapse the news with T and then use that one vector as C's entire key/value set.

```text
z_mean = masked_article_mean(H_N)
z_T = Attention(Q=q_task, K=H_N, V=H_N)       if T == 1 else zeros(d)
z_C = Attention(Q=project(z_P), K=H_N, V=H_N) if C == 1 else zeros(d)
z_text = concat(z_mean, z_T, z_C)

g = 1                                           if G == 0
g = sigmoid(gate([z_P, log1p(n_berita)]))        if G == 1

input_head = concat(z_P, has_news * g * z_text, metadata)
p_up = sigmoid(classifier(input_head))
```

Use fixed-size zero slots for inactive T/C summaries, so classifier dimensions and architecture stay the same across all Level 3 variants. A000 remains simple concatenation with only its mean-text slot active. Optional components add parameters; report them and use capacity-matched controls when making attention-specific claims.

Do not switch to `price + g * text` only in gated variants. Changing concatenation to addition would mix a fusion change with the gate intervention. Keep the gate's inputs and structure identical in A001, A101, A011, and A111.

Before attention, bypass empty-news samples: all text slots are zero and the availability mask disables their contribution. Do not softmax an all-masked sequence. Preserve `n_berita` and `has_news` for activity controls.

With one valid article, the attention weight over articles is 1. Thus neither T nor C can demonstrate selection among articles on that sample, although their learned projections may still change the representation. Report singleton and multi-article coverage. Sentence-level attention is an optional separately declared extension, not an unreported change to the primary key axis.

## 7. What does the query ablation establish?

`A100 - A000` tests adding **learned task-query pooling**. With only USD/IDR in the dataset, a freely learned constant query does not by itself establish semantic understanding of USD versus IDR.

A generic freely learned query with the same architecture may be functionally the same model. Simply renaming its parameter is not an informative semantic control. If initialization differs, that comparison tests initialization, not necessarily persistent target conditioning.

If the report makes a stronger USD/IDR semantic-conditioning claim, add a separate matched test: fixed text-derived USD/IDR and generic queries, encoded with the same eligible encoder and scored through the same frozen similarity/pooling operation. Hold everything else fixed and report it outside the main learned-query contrast. State encoder coverage limits and do not infer cross-currency generalization from one pair.

Until that additional evidence exists, use **learned task-query pooling** as the operational description and treat stronger target-awareness claims as unestablished.

## 8. Prespecified comparisons and claim limits

Let `S(model)` denote an out-of-sample score, with MCC primary.

| Claim | Comparison | Interpretation |
|---|---|---|
| Added query pooling alone | A100 - A000 | Standalone learned-pooling contribution |
| Added price-query attention alone | A010 - A000 | Standalone cross-attention contribution |
| Added learned gate alone | A001 - A000 | Standalone gating contribution |
| Complete model versus baseline | A111 - A000 | Joint change; does not isolate components |
| Query pooling inside the complete model | A111 - A011 | Requires extension A011 |
| Cross-attention inside the complete model | A111 - A101 | Requires extension A101 |
| Gating inside the complete model | A111 - A110 | Requires extension A110 |

If all eight cells are run, additional matched contrasts and component interactions may be reported with the full design. Without the corresponding cells, do not estimate or claim those interactions.

Choose the main confirmatory contrast before the locked test according to the main claim. Use a declared multiplicity policy for additional confirmatory tests, and label other contrasts exploratory. Resample the same temporal blocks and use paired seeds for model differences. Seed variability and sampling uncertainty are different quantities.

## 9. Reporting and execution checklist

- Produce separate tables for Level 1, Level 2, and Level 3, with validation and final-test columns clearly distinguished.
- Report the validation selection rule, both representation winners, all candidate results, and why the Level 3 representation was chosen.
- Report all eligible days and news-present/empty/singleton/multi-article subsets with counts. Use the target endpoint's weekday for gap-related groups.
- Keep source hashes, encoder/vectorizer revisions, preprocessing, fold dates, warmup/exclusions, class counts, hyperparameters, and training budgets reproducible.
- Keep article counts and sentiment/embedding intensity separate. Do not filter articles using future returns.
- Do not describe attention weights, gate values, or better classification scores as proof of economic causality or tradable profit.
- Stop short of a strong forecasting claim when news timestamps/body versions or historical JISDOR availability remain unverified.

The goal is to establish which parts help under a declared protocol, including the possibility that TF-IDF or simple concatenation remains the strongest benchmark.
