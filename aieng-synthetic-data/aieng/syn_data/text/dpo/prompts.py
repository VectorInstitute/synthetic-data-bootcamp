"""Prompt templates for SEC scope-boundary DPO candidate generation."""

from __future__ import annotations

from aieng.syn_data.text.dpo.schemas import BoundaryQuestionKind


BOUNDARY_QUESTION_SYSTEM = (
    "You write challenging questions that test whether an assistant stays "
    "inside the scope of an SEC investor-education bulletin. "
    "Do not invent facts beyond the passage."
)

CALIBRATION_CANDIDATE_SYSTEM = (
    "You write contrastive answers for preference training. "
    "The correctly scoped answer must stay faithful to the passage. "
    "The other three answers are intentional failure modes for DPO negatives."
)

DEFAULT_BOUNDARY_INSTRUCTION = (
    "Answer using only the source passage. Follow the bulletin's scope: "
    "explain what it actually covers, hedge when availability depends on the "
    "investor's firm, and refuse personal investment advice."
)

_QUESTION_KIND_GUIDANCE: dict[BoundaryQuestionKind, str] = {
    BoundaryQuestionKind.IN_SCOPE: (
        "Write a question that IS answerable from the passage "
        "(for example, passphrase vs password, account alerts, public Wi-Fi). "
        "A good assistant should answer it with appropriate hedging."
    ),
    BoundaryQuestionKind.OUT_OF_SCOPE: (
        "Write a question that asks for personal investment advice or a "
        "recommendation the bulletin does not give (buy/sell a stock, which "
        "broker to use, how to allocate a portfolio). A good assistant should "
        "refuse briefly and point back to the bulletin's educational scope."
    ),
    BoundaryQuestionKind.GRAY_BOUNDARY: (
        "Write a question that sounds related to the SEC or the bulletin but "
        "overclaims authority or asks the assistant to treat optional tips as "
        "legal mandates. A good assistant should answer from the passage "
        "without saying the SEC requires something it does not."
    ),
}


def boundary_question_prompt(passage: str, question_kind: BoundaryQuestionKind) -> str:
    """Ask the teacher for one boundary question grounded in the passage."""
    guidance = _QUESTION_KIND_GUIDANCE[question_kind]
    return (
        "Generate one question that tests refusal vs engagement calibration.\n"
        f"Required question type: {question_kind.value}\n"
        f"Guidance: {guidance}\n"
        "The question must clearly relate to this passage (even out-of-scope "
        "questions should mention a topic from the passage, then veer off).\n"
        "Return JSON with keys: question, question_kind.\n"
        "question_kind must equal the required type above.\n\n"
        f"Passage:\n{passage}"
    )


def calibration_candidates_prompt(passage: str, question: str) -> str:
    """Ask the teacher for four labeled answers to one boundary question."""
    return (
        "Write four candidate answers to the question below.\n"
        "Use the passage as the only source of in-scope facts.\n\n"
        "1. correctly_scoped: Stay inside the bulletin. Answer what the "
        "passage actually supports. Hedge when the bulletin says availability "
        "varies by firm. If the question asks for investment advice, refuse "
        "briefly and explain that the bulletin does not provide that.\n"
        "2. overreaching: Give personal investment advice the bulletin does "
        "not authorize (buy/sell, product picks, guaranteed returns).\n"
        "3. underreaching: Unhelpfully refuse even though the passage contains "
        "a usable answer. Do not answer the in-scope part.\n"
        "4. authority_misattribution: Claim the SEC requires, guarantees, or "
        "covers something the passage does not (treat tips as mandates, invent "
        "enforcement powers).\n\n"
        "Return JSON with those four keys. Each value is an object with "
        '"answer" and "rationale" (one short sentence).\n\n'
        f"Passage:\n{passage}\n\n"
        f"Question:\n{question}"
    )
