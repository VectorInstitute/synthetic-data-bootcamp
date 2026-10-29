# Shared fact questions

`qa/fact_questions.py` is the one place the financial-fact questions are made, for
every synthetic set (Method 1, Method 2) and for real filings. `evals/section_scorecard.py`
and `evals/compare_sets.py` both call it.

- **Fixed bank:** ten questions per company: revenue, net income, operating income,
  total assets, total liabilities, stockholders' equity, cash, operating cash flow,
  net margin, operating margin. Answers come from the statements (financials.json or
  XBRL), never from the text. Real filings get nine (operating cash flow isn't in the
  XBRL extract).
- **fact_coverage:** share of the questions whose answer is stated in the text at all.
- **fact_recall_at_10:** share of the questions with a stating chunk in BM25's top 10.
  Unanswerable questions count as misses, so the denominator is always the full bank.
- **fact_correct_pct:** of the figures the text attaches to one of the ten labels
  ("revenues of $X", "Total assets | Y"), the share equal to the true current or
  prior-year value or the change. Replaces traced_pct in the scorecard ranking.

`compare_sets` keeps only answerable questions, since retrieval evaluation needs a gold chunk.
