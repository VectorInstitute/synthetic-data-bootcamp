# Fidelity, utility and privacy: synthetic sets vs real 10-Ks
Real corpus: 305 filings. Real sample for retrieval: 15 filings. Embedding models: text-embedding-3-small, minishlab/potion-retrieval-32M.

## Fidelity

| | Real 10-Ks | m1_gemini | m2_gemini | m1_gpt4o | m2_gpt4o | m1_opus | m2_opus |
|---|---|---|---|---|---|---|---|
| Companies | 305 | 20 | 20 | 20 | 20 | 20 | 10 |
| Median words, Item 1 | 6933 | 2126 | 2062 | 926 | 646 | 3849 | 10780 |
| Median words, Item 1A | 11254 | 3542 | 2160 | 1155 | 766 | 4392 | 13458 |
| Median words, Item 1C | 883 | 1036 | 986 | 788 | 735 | 1080 | 1068 |
| Median words, Item 7 | 9908 | 2562 | 2305 | 1478 | 738 | 4240 | 13094 |
| Median words, Item 8 | n/a | 435 | 2668 | 435 | 750 | 435 | 17445 |
| Top-3,000 term overlap with real (Jaccard) | 1.0 | 0.54 | 0.48 | 0.43 | 0.40 | 0.56 | 0.59 |
| Dollar figures per 1,000 words (median) | 0.14 | 1.73 | 0.00 | 1.07 | 0.00 | 0.26 | 0.18 |
| Percent figures per 1,000 words (median) | 0.40 | 0.00 | 0.53 | 0.00 | 0.00 | 0.00 | 0.14 |
| Sections with 150+ words |  | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | 43/43 |
| Item 8 has all three statement headings (singular or plural) |  | 20/20 | 20/20 | 20/20 | 5/20 | 20/20 | 3/3 |
| Items 7 and 8 with every figure traced |  | 20/40 | 14/40 | 20/40 | 35/40 | 20/40 | 0/13 |
| Figures traced to the statements |  | 91.8% | 45.6% | 93.5% | 89.2% | 94.7% | 65.6% |
| … or explained as year-over-year |  | 99.7% | 45.6% | 99.6% | 89.2% | 99.7% | 65.6% |

## Utility (retrieval benchmark)

| | Real 10-Ks | m1_gemini | m2_gemini | m1_gpt4o | m2_gpt4o | m1_opus | m2_opus |
|---|---|---|---|---|---|---|---|
| Core QA pairs (5 facts per company) | 31 | 100 (5.0/company) | 34 (1.7/company) | 87 (4.3/company) | 25 (1.2/company) | 98 (4.9/company) | 37 (3.7/company) |
| Scenario QA pairs |  | 0 | 0 | 0 | 0 | 0 | 0 |
| Recall@10, bm25, on its own corpus | 1.00 | 0.42 | 0.15 | 0.72 | 0.68 | 0.38 | 0.38 |
| Recall@10, semantic:text-embedding-3-small, on its own corpus | 1.00 | 0.65 | 0.21 | 0.72 | 0.84 | 0.70 | 0.16 |
| Recall@10, hybrid:bm25+text-embedding-3-small, on its own corpus | 1.00 | 0.65 | 0.18 | 0.77 | 0.84 | 0.74 | 0.30 |
| Recall@10, semantic:potion-retrieval-32M, on its own corpus | 0.90 | 0.28 | 0.21 | 0.40 | 0.68 | 0.23 | 0.19 |
| Recall@10, hybrid:bm25+potion-retrieval-32M, on its own corpus | 1.00 | 0.37 | 0.21 | 0.49 | 0.76 | 0.29 | 0.35 |
| Recall@10, bm25, synthetic + real mixed |  | 0.54 | 0.46 | 0.76 | 0.80 | 0.50 | 0.35 |
| Recall@10, semantic:text-embedding-3-small, synthetic + real mixed |  | 0.62 | 0.35 | 0.68 | 0.71 | 0.65 | 0.34 |
| Recall@10, hybrid:bm25+text-embedding-3-small, synthetic + real mixed |  | 0.65 | 0.38 | 0.75 | 0.89 | 0.68 | 0.40 |
| Recall@10, semantic:potion-retrieval-32M, synthetic + real mixed |  | 0.26 | 0.29 | 0.36 | 0.66 | 0.21 | 0.25 |
| Recall@10, hybrid:bm25+potion-retrieval-32M, synthetic + real mixed |  | 0.37 | 0.43 | 0.46 | 0.77 | 0.28 | 0.34 |
| MRR, bm25, on its own corpus | 0.83 | 0.19 | 0.04 | 0.38 | 0.36 | 0.13 | 0.11 |
| MRR, semantic:text-embedding-3-small, on its own corpus | 0.73 | 0.59 | 0.12 | 0.49 | 0.45 | 0.60 | 0.07 |
| MRR, hybrid:bm25+text-embedding-3-small, on its own corpus | 0.83 | 0.28 | 0.07 | 0.46 | 0.42 | 0.26 | 0.08 |
| MRR, semantic:potion-retrieval-32M, on its own corpus | 0.54 | 0.14 | 0.05 | 0.27 | 0.30 | 0.09 | 0.06 |
| MRR, hybrid:bm25+potion-retrieval-32M, on its own corpus | 0.74 | 0.17 | 0.06 | 0.32 | 0.36 | 0.11 | 0.10 |
| MRR, bm25, synthetic + real mixed |  | 0.35 | 0.31 | 0.50 | 0.52 | 0.29 | 0.12 |
| MRR, semantic:text-embedding-3-small, synthetic + real mixed |  | 0.55 | 0.28 | 0.49 | 0.48 | 0.56 | 0.24 |
| MRR, hybrid:bm25+text-embedding-3-small, synthetic + real mixed |  | 0.37 | 0.28 | 0.49 | 0.49 | 0.35 | 0.16 |
| MRR, semantic:potion-retrieval-32M, synthetic + real mixed |  | 0.13 | 0.14 | 0.25 | 0.36 | 0.09 | 0.11 |
| MRR, hybrid:bm25+potion-retrieval-32M, synthetic + real mixed |  | 0.20 | 0.22 | 0.30 | 0.48 | 0.16 | 0.14 |
| Retriever order matches real (Spearman, MRR) | 1.0 | 0.30 | -0.30 | 0.30 | 0.10 | 0.30 | 0.90 |
| Best retriever (by MRR) | bm25 | semantic:text-embedding-3-small | semantic:text-embedding-3-small | semantic:text-embedding-3-small | semantic:text-embedding-3-small | semantic:text-embedding-3-small | bm25 |

## Privacy

| | Real 10-Ks | m1_gemini | m2_gemini | m1_gpt4o | m2_gpt4o | m1_opus | m2_opus |
|---|---|---|---|---|---|---|---|
| Filings naming a real S&P 500 company |  | 4/20 | 6/20 | 0/20 | 1/20 | 2/20 | 1/10 |
| Names found |  | Northern (4) | Northern (2), Hartford (1), American Electric (1), Public Service (1), Waste Management (1), Paper (1) | none | Paper (1) | Northern (2) | Paper (1) |
| Template company's name or ticker in text |  | n/a (no template) | 0/20 | n/a (no template) | 0/20 | n/a (no template) | 0/10 |
| Median distance to nearest real company-year |  | 7.8% | 6.7% | 7.8% | 6.7% | 7.8% | 7.5% |
| Companies within 5% of a real company on every field |  | 0 | 0 | 0 | 0 | 0 | 0 |
| Median distance to own template |  | n/a | 17.9% | n/a | 17.9% | n/a | 20.9% |
| 8-word sequences also in real filings (median) | 9.0% | 8.0% | 10.0% | 4.4% | 1.7% | 5.3% | 3.7% |
| … highest single filing | 21.8% | 9.9% | 13.0% | 7.8% | 7.0% | 6.0% | 6.3% |
| Filings over 5% copied |  | 20 | 20 | 8 | 1 | 12 | 3 |

Notes: the real column's lengths and densities cover every real filing; its retrieval figures are the real_only experiment and its copied share compares the real sample with the other real filings. Distances are the median relative difference over total assets, liabilities, equity, revenue, net income and operating income. Name hits are case-sensitive S&P 500 names, so some may be ordinary words; see privacy_text.csv.
