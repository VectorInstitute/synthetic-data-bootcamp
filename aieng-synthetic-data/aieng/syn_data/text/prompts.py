"""Central prompt templates for synthetic Q&A workflows.

Edit these strings when adapting the reference implementation to a new domain.
"""

from __future__ import annotations

from aieng.syn_data.text.schemas import QASample


# The teacher may read the passage; the student at train/eval will not.
STANDALONE_QUESTION_RULES = (
    "The question must be usable without showing the passage to the answering "
    "model. Write it as a natural user question to an assistant that already "
    "knows this document. Do not mention or allude to a hidden source: no "
    '"passage", "context", "excerpt", "provided text", "the text above", '
    '"according to the document above", or "based on the information given". '
    "If you need to name a source, use the document's real title or issuer, "
    "(for example, SEC investor bulletin). The gold answer must stay faithful "
    "to the passage. You may cite rules or names provided in the passage, but must not say "
    '"the passage" or "the context".'
)


# ---------------------------------------------------------------------------
# Instruction back-translation (notebook 04)
# Technique: given text y, generate instruction x for which y is a good answer.
# See: https://openreview.net/forum?id=1oijHJBRsT
# ---------------------------------------------------------------------------

INSTRUCTION_BACKTRANSLATION_SYSTEM = (
    "You generate natural standalone questions for which the provided passage "
    "is a good answer. Do not invent facts beyond the passage. Do not refer to "
    "the passage in the question wording."
)


def instruction_backtranslation_prompt(passage: str) -> str:
    """Ask the teacher to invent a question answered by the passage."""
    return (
        "Instruction back-translation task.\n"
        "The passage below is a good answer/response.\n"
        "Write one natural question for which this passage would be a good answer.\n"
        "The question must be fully answerable from the passage, but must not "
        "assume the reader can see the passage.\n"
        f"{STANDALONE_QUESTION_RULES}\n"
        "Return JSON with a single key: question.\n"
        "Do not include the answer in the question.\n\n"
        f"Passage:\n{passage}"
    )


# ---------------------------------------------------------------------------
# LLM-as-judge: SLM response vs gold (notebooks 01 / 05)
# ---------------------------------------------------------------------------

RESPONSE_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for policy-document question answering. "
    "Score model outputs fairly and conservatively."
)


def absolute_response_judge_prompt(sample: QASample, model_answer: str) -> str:
    """Score a model answer against a reference gold answer."""
    return (
        "Evaluate the model answer against the reference.\n"
        "Return JSON with numeric scores from 1 to 5 for:\n"
        "- correctness\n"
        "- coherence\n"
        "- instruction_following\n"
        "- factual_plausibility\n"
        "Also include a short 1-2 sentence max 'reasoning' string, no text outside JSON.\n\n"
        f"Question:\n{sample.question}\n\n"
        f"Reference answer:\n{sample.gold_answer}\n\n"
        f"Model answer:\n{model_answer}\n"
        "JSON format:"
        """
        {
            "correctness": <number>,
            "coherence": <number>,
            "instruction_following": <number>,
            "factual_plausibility": <number>,
            "reasoning": "<reasoning>"
        }
        """
    )


def pairwise_response_judge_prompt(
    sample: QASample,
    candidate_answer: str,
    reference_answer: str,
) -> str:
    """Compare two candidate answers for the same question."""
    return (
        "Compare answer A and answer B for the question below.\n"
        'Return JSON: {"winner": "A"|"B"|"tie", "reasoning": "..."}\n\n'
        f"Question:\n{sample.question}\n\n"
        f"Answer A:\n{candidate_answer}\n\n"
        f"Answer B:\n{reference_answer}\n"
    )


# ---------------------------------------------------------------------------
# LLM-as-judge: synthetic Q&A quality (notebook 03) — no model response yet
# ---------------------------------------------------------------------------

SYNTHETIC_QA_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of synthetic question-answer training data. "
    "Score whether the Q&A pair is high-quality given the source passage. "
    "Be fair and conservative."
)


def synthetic_qa_quality_prompt(sample: QASample) -> str:
    """Score a synthetic Q&A pair against its source context (pre-inference)."""
    context = sample.context.strip() or "(no passage provided)"
    return (
        "Evaluate the quality of this synthetic question-answer pair for training.\n"
        "There is no model response yet — judge only the question and gold answer "
        "relative to the source passage.\n"
        "Return JSON with numeric scores from 1 to 5 for:\n"
        "- correctness: is the gold answer accurate for the question given the passage?\n"
        "- coherence: are question and answer clear and well-formed?\n"
        "- instruction_following: is the question a natural, useful instruction?\n"
        "- factual_plausibility: is the answer grounded in / faithful to the passage?\n"
        "Also include a short 1-2 sentence max 'reasoning' string, no text outside JSON.\n\n"
        f"Passage:\n{context}\n\n"
        f"Question:\n{sample.question}\n\n"
        f"Gold answer:\n{sample.gold_answer}\n"
        "JSON format:"
        """
        {
            "correctness": <number>,
            "coherence": <number>,
            "instruction_following": <number>,
            "factual_plausibility": <number>,
            "reasoning": "<reasoning>"
        }
        """
    )
