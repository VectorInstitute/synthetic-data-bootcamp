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
  most non-bank companies in the corpus either lack this section or treat it as boilerplate, which
  would make it inconsistent across the corpus. Revisit for banks only if the retrieval evaluation
  turns out too financial-statement-heavy (the corpus is now a random S&P 500 sample rather than a
  bank-heavy list, so there may be few banks to revisit this for; see PLAN.md open question 4).
- **3 — Legal Proceedings.** Analysts do ask about material litigation, but this section is
  frequently just "see Note X," pointing back into the Item 8 notes. Including it risks a lot of
  near-empty parses for low signal. Not recommended unless a specific litigation-QA use case comes up.

## Decision

Ship Phase 1 with the five included items above. Revisit Item 7A (banks only) after the retrieval
evaluation if the question set feels too narrow.

## Implementation (Phase 1, `src/parse_sections.py`)

Run with `python -m src.parse_sections`; see the README for usage. Result over the current corpus
(147 downloaded filings x 5 items = 735 sections): 724 ok, 5 short, 6 missing. Every row's outcome
is in `data/parse_status.csv`.

Any item that's only "incorporated by reference" instead of written out — whichever of the five,
not just Item 7 and 8 — is resolved by finding the heading named in its cross-reference sentence
inside the filing's EX-13 exhibit (PGR, USB, WFC) if it has one, or the 10-K body itself otherwise.
A heading reprinted as a running header on every page (seen in a few large banks and insurers,
e.g. AIG and AFL's Item 8) is detected separately, by an unusually high occurrence count, and every
page it owns is combined rather than returning just one page of it. What's left after both of
these:

- **AEP's Item 7 (3 years, `short`):** its cross-reference points at *a different filing* — its
  own prior-year Form 10-K — for the year-over-year commentary. That filing isn't part of this
  corpus, so there's nothing to resolve it against.
- **AVGO and QCOM's Item 1C (`short`, one year each) and AZO, BEN, CRM, CRWD, DRI, and STX's
  Item 1C (`missing`, one year each):** all 7 have a non-December fiscal year-end, and in each
  case the affected fiscal year's period of report predates the Item 1C effective date (fiscal
  years ending on or after December 15, 2023). Expected, not a parsing gap.
- **PulteGroup's (PHM) Item 1 — fixed.** Its 10-K body heads the section "ITEM I. BUSINESS" — a
  capital letter I instead of the digit 1 — in every year downloaded. `parse_sections.py` special-
  cases this one pattern (`ITEM_ONE_TYPO_RE`) rather than generalizing Roman-numeral headings, since
  it's the only filer observed doing this.
