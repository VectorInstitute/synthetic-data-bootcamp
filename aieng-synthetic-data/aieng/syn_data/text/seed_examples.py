"""Finance-domain seed and few-shot Q&A exemplars.

Edit this file when adapting the tutorial to a new domain.
"""

from __future__ import annotations

from aieng.syn_data.text.schemas import Paragraph, QASample


INSTRUCTION = "Answer using the source passage. Respond in one sentence."

# (question, gold_answer) — keep these short; they only teach format and style.
_EXAMPLES = [
    (
        "What is the grace period for new purchases?",
        "The grace period ends 21 days after the close of the billing cycle.",
    ),
    (
        "What is the annual percentage rate (APR) for purchases?",
        "The purchase APR is a variable rate equal to the Prime Rate plus a margin.",
    ),
    (
        "Does this document recommend a specific investment to buy?",
        "No. The bulletin explains investor-protection concepts and does not recommend a specific investment.",
    ),
]


def default_seed_example(paragraph: Paragraph) -> QASample:
    """One-shot exemplar (first item in the domain list)."""
    return _to_sample(paragraph, 0)


def default_few_shot_examples(paragraph: Paragraph) -> list[QASample]:
    """Few-shot exemplars for the starter finance domain."""
    return [_to_sample(paragraph, i) for i in range(len(_EXAMPLES))]


def _to_sample(paragraph: Paragraph, index: int) -> QASample:
    question, gold_answer = _EXAMPLES[index]
    return QASample(
        id=f"seed-{index}",
        question=question,
        gold_answer=gold_answer,
        doc_id=paragraph.doc_id,
        para_id=paragraph.para_id,
        context=paragraph.text,
        instruction=INSTRUCTION,
    )
