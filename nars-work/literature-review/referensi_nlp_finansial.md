# NLP references for geopolitical news and USD/IDR

Reviewed on 10 November 2026. Purpose: select task formulations and architectures in which text understanding is a central component. These literature notes support the main project; dataset audits remain separate in `../sanity-check/`.

**Current user decision: binary UP/DOWN classification of USD/IDR using geopolitical news.** The alternatives below are background options from the initial literature exploration, not the current task selection. See the [focused paper review](dakalbab_2025_usdidr_binary_review.md) and [proposed binary protocol](../rules/binary_classification_protocol.md).

Classifying a future direction is still forecasting; classifying sentiment or impact expressed within an article is a different task. Success on one task does not establish success on the other.

## Project focus following the user's clarification

**The research concerns the relationship between geopolitical news and USD/IDR exchange-rate levels or movements.** Flexibility in task selection does not change that subject. The initial recommendation of standalone financial sentiment classification was too broad for this purpose.

Proposed research question: *Do NLP representations of geopolitical news provide additional information about USD/IDR movements compared with exchange-rate history alone?* This is an empirical question, rather than an assumption that geopolitics always has an effect or a particular direction of influence.

Relevance classification, risk intensity, target identification, and event extraction serve as text-processing modules. The research output may be USD/IDR movement classification, return regression, or an association analysis that distinguishes news and time periods. Forecasting requires all inputs to be available before the outcome period.

Proposed scope: conflict or war, international sanctions, escalation or de-escalation, interstate tensions, and trade policies associated with interstate conflicts. Fed news, inflation, company earnings, and domestic politics are not automatically geopolitical news. The inclusion of tariffs or trade wars must be explained in the project's definition; do not assume it exactly matches every GPR definition in the literature.

## Additional foundations for the geopolitical domain

### Measuring Geopolitical Risk

Caldara & Iacoviello, **American Economic Review 112(4), 2022**, 1194–1225. [Official article](https://www.aeaweb.org/articles?id=10.1257/aer.20191823).

A news-based index measuring geopolitical risk, distinguishing threats from realized events. Useful for defining the construct and a text-search benchmark. This is a domain foundation, rather than a neural architecture reference or specific evidence for USD/IDR. The publisher's abstract and indexed methodology sections were inspected.

### The AI-GPR Index: Measuring Geopolitical Risk using Artificial Intelligence

Iacoviello & Tong, **working paper, 2026**. [Authors' page and paper](https://www.matteoiacoviello.com/ai_gpr).

The methodology page describes filtering articles, scoring geopolitical risk intensity with an LLM, and aggregating the scores into an index. Breakdowns by event type, country, and role are available. This is the most direct addition for the geopolitical NLP module; it does not establish the ability to predict USD/IDR. Its status is a working paper, rather than a final journal article. The authors' page was inspected; the complete PDF has not been reviewed.

### Exchange rate return predictability in times of geopolitical risk

Iyke, Phan & Narayan, **International Review of Financial Analysis 81, 2022**, 102099. [Publisher's page](https://www.sciencedirect.com/science/article/pii/S1057521922000692).

Tests the usefulness of geopolitical risk information for forecasting exchange-rate returns in 17 countries. It connects the geopolitical construct to forex outcomes, rather than providing the main reference for modern NLP architectures. Cross-country results do not automatically apply to USD/IDR. The abstract and publisher's summary were inspected; specific coverage of Indonesia has not been verified.

## Main readings

### 1. Group, Extract and Aggregate: Summarizing a Large Amount of Finance News for Forex Movement Prediction

Chen et al., **ECONLP 2019**, a workshop rather than the main EMNLP conference. [Official paper and PDF](https://aclanthology.org/D19-5106/).

- **Task and method:** intraday forex direction classification. BHAM groups news, selects important items using BERT representations, then aggregates them and models their interaction with trading data through attention.
- **Key idea:** news selection is part of the architecture. The experiments use **headlines**, rather than complete article bodies. The authors compare grouping by time, topic, and category.
- **Applicability limits:** Reuters, minute-level data, and tens of thousands of samples, rather than our scale of 435 articles. Adapting the idea is more reasonable than reproducing the model size or minute-level horizon.
- **Sections to read:** §3; §4.1–4.2; §5.4 on the number of selected news items. The PDF is available and its methods/data sections were inspected.

### 2. Enhancing exchange rate forecasting with contextual sentiment indices: A fine-tuned FinBERT approach

**Applied Soft Computing 189, 2026**, article 114556. [Publisher's page](https://www.sciencedirect.com/science/article/pii/S1568494626000049). [DOI](https://doi.org/10.1016/j.asoc.2026.114556).

- **Task and method:** annotate EUR/USD news, fine-tune FinBERT on an annual rolling basis, construct contextual sentiment indices, and then forecast exchange rates.
- **Key idea:** sentiment is adapted to the currency pair; generic positive/negative labels are insufficient. This provides a foundation for classifying the impact expressed in text first.
- **Applicability limits:** the reported **84.33%** is forex sentiment classification accuracy, not exchange-rate direction prediction accuracy. The paper also mentions interpolation of missing data; we should not automatically adopt its calendar treatment.
- **Review depth:** abstract and indexed sections on the publisher's page. The full PDF, appendices, and code/dataset have not been verified.

### 3. FinEntity: Entity-level Sentiment Classification for Financial Texts

Tang et al., **EMNLP 2023**. [Official paper/PDF](https://aclanthology.org/2023.emnlp-main.956/). [Authors' repository](https://github.com/yixuantt/FinEntity).

- **Task and method:** entity-level sentiment classification using a BERT/FinBERT encoder with either a linear token head or a CRF. The final dataset contains 979 paragraphs and 2,131 entities.
- **Key idea:** one paragraph can be positive for one entity and negative for another. This provides a strong foundation for distinguishing USD and IDR as targets.
- **Applicability limits:** transfer to currencies is our proposed adaptation, not a result established by the paper for USD/IDR. A CRF is useful when performing span/token tagging; it is not required for classifying one target per article.
- **Sections to read:** §3–4 for annotation/architecture; §5 for the cryptocurrency study. The PDF and methodology sections were inspected.

### 4. Event-Driven Learning of Systematic Behaviours in Stock Markets

Wu, **Findings of EMNLP 2020**. [Official paper/PDF](https://aclanthology.org/2020.findings-emnlp.220/).

- **Task and method:** stock movement classification using coreference resolution + OpenIE, BERT/ALBERT event representations, and attention at the event, news, and temporal levels.
- **Key idea:** representing “who does what to whom” can preserve policy/event meaning that is lost in a single document sentiment score.
- **Applicability limits:** results concern US stocks and indices. The complete system is large for our dataset; event extraction can fail, and attention does not establish causality.
- **Sections to read:** §2–3 and Figure 2. The PDF and methods were inspected.

### 5. Advancing Forex prediction through multimodal text-driven model and attention mechanisms

Dakalbab et al., **Intelligent Systems with Applications 26, 2025**, article 200518. [Publisher's page/DOI](https://doi.org/10.1016/j.iswa.2025.200518).

- **Task and method:** up/stable/down forex classification. Sentiment and price features pass through separate neural branches, with comparisons of several attention mechanisms including cross-modal attention.
- **Key idea:** fusion models interactions between the two modalities beyond simple concatenation.
- **Applicability limits:** the text branch uses sentiment features, so it is not equivalent to end-to-end interaction between news tokens and prices. The evaluation describes a 70/30 split and five-fold CV; our forecasting protocol must ensure chronological separation and an embargo.
- **Sections to read:** §4.4–4.6. The abstract and indexed publisher sections were inspected; the implementation has not been independently reproduced.

### 6. A hierarchical attention network for stock prediction based on attentive multi-view news learning

Chen et al., **Neurocomputing 504, 2022**, pages 1–15. [Publisher's page](https://www.sciencedirect.com/science/article/pii/S0925231222008372). [DOI](https://doi.org/10.1016/j.neucom.2022.06.106).

- **Task and method:** stock prediction. NMNL encodes headlines, bodies, and sentiment as separate views, aggregates important news, combines market indicators, and uses a BiLSTM for temporal context.
- **Key idea:** supports experiments on whether headlines, cleaned article bodies, and sentiment provide additional information.
- **Applicability limits:** stock markets. Bodies contaminated with sidebars still require cleaning. Attention does not automatically correct temporal provenance.
- **Review depth:** abstract and architecture summary on the publisher's page; the complete PDF has not been verified.

## Additional readings on architecture choices

### 7. Stock Movement Prediction from Tweets and Historical Prices

Xu & Cohen, **ACL 2018**. [Official paper/PDF](https://aclanthology.org/P18-1183/).

StockNet uses a generative model with recurrent latent variables and variational inference to combine text and prices, along with a temporal auxiliary objective. This is a reading for probabilistic approaches; its text input consists of **tweets**, rather than news articles. It is not the first choice for our small dataset. The official metadata and abstract were inspected.

### 8. FinBERT: A Pretrained Language Model for Financial Communications

Yang, Uy & Huang, **preprint, 2020**. [Paper](https://arxiv.org/abs/2006.08097). [Authors' repository](https://github.com/yya518/FinBERT).

A foundation for financial language encoders through pretraining on financial communication corpora. Useful as a backbone/baseline, but this paper does not validate USD/IDR forecasting. Distinguish it from Araci's FinBERT (2019): a shared name does not guarantee the same model, corpus, or checkpoint. The metadata and abstract were inspected.

## Suggested tasks given the current dataset

This section presents our proposals based on the audit and literature, rather than architectures claimed exactly as written in a particular paper.

| Option | Learning target | Main inspiration | Requirements and evaluation |
|---|---|---|---|
| A. Text classification with a currency target | Relevance to USD/IDR, followed by the pressure expressed in the text toward a particular currency | FXFinBERT + FinEntity | Human annotation with a clear target; macro-F1, per-class F1, and error analysis. Labels must not simply copy the return after publication. |
| B. Future movement classification | Direction of USD/IDR returns at a specified horizon | BHAM; cross-modal attention | Clean main text, clear news availability and timestamps, and broader coverage; chronological splits, macro-F1/MCC, and a baseline without text. |
| C. Event extraction/classification | Actors, policy actions, and event types; may feed into prediction | Event-HAN | Event annotation and extraction evaluation; do not assess only the final market score. |
| D. Return regression | Magnitude of future returns | Adapted pooling/fusion from the architecture papers | Numeric outcomes and chronological evaluation; MAE/RMSE, a zero-return baseline, and tests of the added value of text. This is an adaptation, not the original target of every paper. |

**Current priority: B, binary future-movement classification.** A and C become intermediate modules evaluated separately; D remains a background alternative. Start by checking geopolitical relevance and improving timing/quality, then test whether text representations add information beyond the exchange-rate baseline. The current dataset size limits model complexity and the strength of conclusions; it is not a reason to change the research subject to general financial sentiment.

435 articles do not constitute 435 independent exchange-rate-day observations: the dataset covered only 212 exchange-rate days at the time of the audit. Splitting articles into sentences does not automatically increase the number of independent observations either. For text classification, keep duplicates and versions of the same story from crossing train–test boundaries. For forecasting, split by time, fit preprocessing only on training data, and ensure an encoder trained with return labels never sees the test period.

When developing a model, start with TF-IDF + a simple classifier and a pretrained encoder + a small head. Then compare mean pooling with attention or target-aware pooling. This separates architectural gains from the benefits of pretraining and data improvements. These are proposed experiments, not tested results.

Reading order after clarification: **Measuring Geopolitical Risk → AI-GPR → BHAM → Event-HAN/FXFinBERT**. FinEntity, StockNet, and the other stock papers are methodological references to adapt, rather than evidence about the geopolitics–USD/IDR domain.

The selected references do not establish that any particular architecture will work on USD/IDR and our dataset. Full-text access and code availability are distinguished explicitly. Repository links originate from the papers, but their contents and executability have not been audited.
