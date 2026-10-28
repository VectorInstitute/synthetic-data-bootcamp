# Per-section scorecard

One table per Item. `real` is the same measure on real 10-Ks (for self-retrieval and cross-document overlap, a real sample of the same size as the sets). See `evals/section_scorecard.py` for each metric.

## Suggested method per section

| Item | Suggested set | Best figure accuracy | Best retrieval utility |
|---|---|---|---|
| Item 1 Business | m1_gpt4o | m1_opus | m1_gpt4o |
| Item 1A Risk Factors | m2_opus | m1_gemini | m1_gemini |
| Item 1C Cybersecurity | m1_gemini |  |  |
| Item 7 MD&A | m1_gemini | m1_gemini | m2_gpt4o |
| Item 8 Financial Statements | m1_gemini | m1_gemini | m1_gpt4o |

Suggested set = highest weighted rank across the metrics (figure accuracy and fact retrieval count twice). Bold = best set on that metric.

## Item 1 Business

| Metric | m1_gemini | m2_gemini | m1_gpt4o | m2_gpt4o | m1_opus | m2_opus | real |
|---|---|---|---|---|---|---|---|
| n_docs | 20 | 20 | 20 | 20 | 20 | 10 | 20 |
| median_words | 2126 | 2062 | 926 | 646 | 3849 | 10780 | 6933 |
| n_figures | 31 | 41 | 19 | 4 | 44 | 16 | 313 |
| n_questions | 21 | 6 | 18 | 4 | 20 | 10 | 9 |
| traced_pct | 93.5 | 31.7 |  |  | **100** | 100 | 13.7 |
| fact_recall_at_10 | 0.333 | 0 | **1** |  | 0.25 | 0.5 | 1 |
| self_retrieval_mrr | 0.983 | 0.966 | **1** | 1 | 0.971 | 0.959 | 1 |
| length_ratio | 0.31 | 0.3 | 0.13 | 0.09 | **0.56** | 1.55 | 1 |
| vocab_jaccard | 0.472 | 0.476 | 0.356 | 0.329 | **0.521** | 0.516 | 1 |
| figures_per_1k | 0.53 | 0.46 | **1.07** | 0 | 0.54 | 0.15 | 1.88 |
| cross_doc_overlap | 0.093 | 0.045 | 0.025 | **0.003** | 0.091 | 0.045 | 0.01 |
| copied_from_real | 0.061 | 0.041 | 0.006 | 0 | 0.055 | **0.028** | 0.031 |

## Item 1A Risk Factors

| Metric | m1_gemini | m2_gemini | m1_gpt4o | m2_gpt4o | m1_opus | m2_opus | real |
|---|---|---|---|---|---|---|---|
| n_docs | 20 | 20 | 20 | 20 | 20 | 10 | 20 |
| median_words | 3542 | 2160 | 1155 | 766 | 4392 | 13458 | 11254 |
| n_figures | 151 | 43 | 0 | 3 | 0 | 70 | 138 |
| n_questions | 11 | 4 | 0 | 2 | 0 | 6 | 0 |
| traced_pct | **99.3** | 62.8 |  |  |  | 74.3 | 11.6 |
| fact_recall_at_10 | **0.545** |  |  |  |  | 0.167 |  |
| self_retrieval_mrr | 0.97 | **0.944** | 1 | 1 | 0.947 | 0.947 | 0.864 |
| length_ratio | 0.31 | 0.19 | 0.1 | 0.07 | 0.39 | **1.2** | 1 |
| vocab_jaccard | **0.592** | 0.472 | 0.453 | 0.421 | 0.553 | 0.579 | 1 |
| figures_per_1k | 2.46 | **0.49** | 0 | 0 | 0 | 0.38 | 0.48 |
| cross_doc_overlap | 0.103 | 0.086 | 0.03 | **0.015** | 0.078 | 0.048 | 0.013 |
| copied_from_real | 0.065 | **0.089** | 0.033 | 0.026 | 0.038 | 0.031 | 0.11 |

## Item 1C Cybersecurity

| Metric | m1_gemini | m2_gemini | m1_gpt4o | m2_gpt4o | m1_opus | m2_opus | real |
|---|---|---|---|---|---|---|---|
| n_docs | 20 | 20 | 20 | 20 | 20 | 10 | 20 |
| median_words | 1036 | 986 | 788 | 735 | 1080 | 1068 | 883 |
| n_figures | 0 | 15 | 0 | 0 | 0 | 0 | 0 |
| n_questions | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| traced_pct |  |  |  |  |  |  |  |
| fact_recall_at_10 |  |  |  |  |  |  |  |
| self_retrieval_mrr | **0.961** | 0.912 | 1 | 1 | 0.936 | 0.928 | 0.956 |
| length_ratio | 1.17 | 1.12 | **0.89** | 0.83 | 1.22 | 1.21 | 1 |
| vocab_jaccard | **0.519** | 0.438 | 0.48 | 0.409 | 0.459 | 0.451 | 1 |
| figures_per_1k | **0** | 0 | 0 | 0 | 0 | 0 | 0 |
| cross_doc_overlap | 0.096 | 0.101 | 0.021 | **0.011** | 0.135 | 0.101 | 0.01 |
| copied_from_real | 0.047 | 0.051 | 0.023 | 0.005 | 0.049 | **0.058** | 0.072 |

## Item 7 MD&A

| Metric | m1_gemini | m2_gemini | m1_gpt4o | m2_gpt4o | m1_opus | m2_opus | real |
|---|---|---|---|---|---|---|---|
| n_docs | 20 | 20 | 20 | 20 | 20 | 10 | 20 |
| median_words | 2562 | 2305 | 1478 | 738 | 4240 | 13094 | 9908 |
| n_figures | 1520 | 179 | 992 | 140 | 2063 | 1534 | 3182 |
| n_questions | 57 | 21 | 49 | 19 | 73 | 21 | 21 |
| traced_pct | **99.5** | 70.9 | 99 | 97.1 | 99.5 | 67.1 | 11.7 |
| fact_recall_at_10 | 0.509 | 0.238 | 0.51 | **0.579** | 0.438 | 0.381 | 1 |
| self_retrieval_mrr | **0.953** | 0.965 | 1 | 1 | 0.901 | 0.953 | 0.939 |
| length_ratio | 0.26 | 0.23 | 0.15 | 0.07 | 0.43 | **1.32** | 1 |
| vocab_jaccard | 0.489 | 0.427 | 0.382 | 0.278 | 0.517 | **0.548** | 1 |
| figures_per_1k | 28.98 | 2.82 | 33.31 | 0.73 | 25.15 | **12.62** | 12.23 |
| cross_doc_overlap | 0.194 | 0.112 | 0.118 | **0.018** | 0.132 | 0.05 | 0.019 |
| copied_from_real | 0.123 | **0.087** | 0.085 | 0.02 | 0.055 | 0.03 | 0.088 |

## Item 8 Financial Statements

| Metric | m1_gemini | m2_gemini | m1_gpt4o | m2_gpt4o | m1_opus | m2_opus |
|---|---|---|---|---|---|---|
| n_docs | 20 | 20 | 20 | 20 | 20 | 3 |
| median_words | 435 | 2668 | 435 | 750 | 435 | 17445 |
| n_figures | 1320 | 205 | 1320 | 82 | 1320 | 198 |
| n_questions | 11 | 2 | 20 | 0 | 5 | 0 |
| traced_pct | **100** | 23.4 | 100 | 75.6 | 100 | 54.5 |
| fact_recall_at_10 | 0 |  | **1** |  | 0 |  |
| self_retrieval_mrr | 1 | 0.758 | 1 | 0.952 | 1 | 0.95 |
| length_ratio |  |  |  |  |  |  |
| vocab_jaccard |  |  |  |  |  |  |
| figures_per_1k | 151.72 | 3.6 | 151.72 | 0 | 151.72 | 3.04 |
| cross_doc_overlap | 0.042 | 0.295 | 0.042 | 0.026 | 0.042 | 0.102 |
| copied_from_real |  |  |  |  |  |  |
