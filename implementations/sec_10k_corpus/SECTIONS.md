# 10-K Section Scope

Which 10-K items the parser (Phase 1 / step 3) extracts, and why. This applies to both the real
filings in `data/parsed/` and the synthetic filings generated later. See `PLAN.md` for how this
fits into the overall plan.

## Included

| Item | Why |
|---|---|
| **1 — Business** | Segments, products, and competitive position. Needed for descriptive questions ("what lines of business does X operate") and as background for other answers. |
| **1A — Risk Factors** | The most analyst-relevant qualitative section. Risk factors are specific enough per company to make good retrieval questions, but similarly phrased across banks, which makes them a genuinely hard retrieval case — useful for comparing BM25, dense, and hybrid retrieval. |
| **1C — Cybersecurity** | Short and recent (required from FY2023 onward only), so it tests whether a retriever can find a small, narrow section instead of defaulting to the largest ones. A "needle in haystack" case. |
| **7 — MD&A** | The core analysis section: what drove revenue, margin, and credit-quality changes. Most "why did X happen" equity-analysis questions map here, and it's the section real analysts read first. |
| **8 — Financial Statements** | The only section with hard numbers and footnotes. Needed for any question with a verifiable numeric answer (for example, "what was net interest income in FY2023"). For synthetic filings, this section's numbers come from the computed spec rather than the model, so it is the ground-truth anchor for correctness checks. |

Together these cover description, risk, a narrow/recent topic, analysis, and numbers — a spread of
question types without pulling in sections that add parsing risk for little benefit.

## Excluded

| Item | Why excluded |
|---|---|
| **1B — Unresolved Staff Comments** | Almost always "None." No QA value. |
| **2 — Properties** | For banks, usually one paragraph on branch count. Low value for the parsing effort. |
| **4 — Mine Safety Disclosures** | Not applicable to banks or most S&P 500 companies in this list. |
| **5 — Market for Registrant's Equity** | Stock performance and dividend data. Not excluded for lack of value, just lower priority than the five included items; revisit if dividend/buyback questions are wanted later. |
| **6 — Reserved** | Literally reserved; no content since the SEC removed the item. |
| **9, 9A, 9B, 9C — Accountant changes / controls / other / foreign jurisdictions** | Almost never has company-specific content; standard boilerplate. |
| **10–14 — Governance, compensation, ownership, related-party, auditor fees** | Frequently **incorporated by reference to the proxy statement (DEF 14A)**, meaning the content is often not in the 10-K body at all. Including these would produce a lot of "missing section" noise for sections that were never going to be there. |
| **15, 16 — Exhibits / Form 10-K Summary** | Exhibit list and an optional summary almost nobody files. Not narrative content. |

## Borderline: flagged but not included yet

- **7A — Quantitative and Qualitative Disclosures about Market Risk.** For banks specifically, this
  is where interest-rate sensitivity tables live — a real equity-analysis topic (NIM sensitivity to
  rate moves). Quantitative and bank-specific, so it fits the project's theme. Not added yet because
  the 35 non-bank S&P 500 companies in the corpus either lack this section or treat it as boilerplate,
  which would make it inconsistent across the corpus. Revisit for the 15 banks only if the retrieval
  evaluation turns out too financial-statement-heavy.
- **3 — Legal Proceedings.** Analysts do ask about material litigation, but this section is
  frequently just "see Note X," pointing back into the Item 8 notes. Including it risks a lot of
  near-empty parses for low signal. Not recommended unless a specific litigation-QA use case comes up.

## Decision

Ship Phase 1 with the five included items above. Revisit Item 7A (banks only) after the retrieval
evaluation if the question set feels too narrow.

## Implementation (Phase 1, `src/parse_sections.py`)

Run with `python -m src.parse_sections`; see the README for usage. Result over the full corpus
(50 companies x 3 fiscal years x 5 items = 750 sections): 687 ok, 36 short, 27 missing. Every
row's outcome is in `data/parse_status.csv`.

- Several companies (BAC, GS, JPM, TFC, and WFC among them) "incorporate by reference" Item 1C
  or 1A to their annual report instead of writing the content directly. Only Item 7 and 8 have
  an EX-13 fallback (see PLAN.md Phase 1), so these are left as `short` rows for manual review.
- Citigroup's 10-K body doesn't use plain-text "Item N" headings at all (it's a page-number
  cross-reference index into a combined annual report); all 5 items are `missing`. The Hartford's
  Item 8 heading is similarly absent from the plain text, likely lost in HTML-to-text conversion
  of a table-based heading. Neither has an EX-13 exhibit to fall back to.
- 9 non-December fiscal-year-end companies have Item 1C `missing` for their earliest fiscal year,
  because that year's period of report predates the item's effective date.
