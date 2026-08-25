"""Data models for refusal-calibration preference pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from aieng.syn_data.text.schemas import (
    DocumentRole,
    FailureMode,
    ParagraphSplit,
    QASample,
)


class CandidateKind(StrEnum):
    """Labels for the four refusal-calibration answer variants."""

    CORRECTLY_SCOPED = "correctly_scoped"
    OVERREACHING = "overreaching"
    UNDERREACHING = "underreaching"
    AUTHORITY_MISATTRIBUTION = "authority_misattribution"


class BoundaryQuestionKind(StrEnum):
    """How a generated question sits relative to the SEC bulletin scope."""

    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    GRAY_BOUNDARY = "gray_boundary"


REJECTED_CANDIDATE_KINDS: tuple[CandidateKind, ...] = (
    CandidateKind.OVERREACHING,
    CandidateKind.UNDERREACHING,
    CandidateKind.AUTHORITY_MISATTRIBUTION,
)


@dataclass
class PreferenceCandidate:
    """One labeled answer variant for a boundary question."""

    kind: CandidateKind
    answer: str
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "kind": self.kind.value,
            "answer": self.answer,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferenceCandidate:
        """Deserialize from a dictionary."""
        return cls(
            kind=CandidateKind(data["kind"]),
            answer=data["answer"],
            rationale=data.get("rationale", ""),
        )


@dataclass
class CalibrationPrompt:
    """A scope-boundary question with four contrastive candidate answers."""

    id: str
    question: str
    doc_id: str
    para_id: str
    context: str
    question_kind: BoundaryQuestionKind
    instruction: str = ""
    candidates: list[PreferenceCandidate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def chosen_answer(self) -> str:
        """Return the correctly scoped candidate answer."""
        for candidate in self.candidates:
            if candidate.kind == CandidateKind.CORRECTLY_SCOPED:
                return candidate.answer
        return ""

    def candidate_by_kind(self, kind: CandidateKind) -> PreferenceCandidate | None:
        """Return the candidate with the given kind, if present."""
        for candidate in self.candidates:
            if candidate.kind == kind:
                return candidate
        return None

    def to_qa_sample(self) -> QASample:
        """Convert to a QASample for eval prompts and the pairwise judge."""
        return QASample(
            id=self.id,
            question=self.question,
            gold_answer=self.chosen_answer(),
            doc_id=self.doc_id,
            para_id=self.para_id,
            context=self.context,
            failure_mode=FailureMode.REFUSAL_CALIBRATION,
            role=DocumentRole.SCOPE_BOUNDARY,
            instruction=self.instruction,
            split=ParagraphSplit.TRAIN,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "doc_id": self.doc_id,
            "para_id": self.para_id,
            "context": self.context,
            "question_kind": self.question_kind.value,
            "instruction": self.instruction,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationPrompt:
        """Deserialize from a dictionary."""
        return cls(
            id=data["id"],
            question=data["question"],
            doc_id=data["doc_id"],
            para_id=data["para_id"],
            context=data.get("context", ""),
            question_kind=BoundaryQuestionKind(data["question_kind"]),
            instruction=data.get("instruction", ""),
            candidates=[
                PreferenceCandidate.from_dict(item)
                for item in data.get("candidates", [])
            ],
            metadata=data.get("metadata", {}),
        )


@dataclass
class PreferencePair:
    """One TRL-style DPO row: prompt, chosen answer, rejected answer."""

    id: str
    prompt: str
    chosen: str
    rejected: str
    rejected_kind: CandidateKind
    prompt_id: str
    doc_id: str
    para_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_trl_row(self) -> dict[str, str]:
        """Return the three fields expected by TRL ``DPOTrainer``."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = asdict(self)
        payload["rejected_kind"] = self.rejected_kind.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferencePair:
        """Deserialize from a dictionary."""
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            chosen=data["chosen"],
            rejected=data["rejected"],
            rejected_kind=CandidateKind(data["rejected_kind"]),
            prompt_id=data["prompt_id"],
            doc_id=data["doc_id"],
            para_id=data["para_id"],
            metadata=data.get("metadata", {}),
        )
