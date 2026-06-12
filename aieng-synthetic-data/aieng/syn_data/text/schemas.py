"""Data models for policy-document QA synthetic data workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class FailureMode(StrEnum):
    """Small-model failure modes targeted by the test set."""

    FORMAT_NON_COMPLIANCE = "format_non_compliance"
    DOMAIN_VOCABULARY_DRIFT = "domain_vocabulary_drift"
    REFUSAL_CALIBRATION = "refusal_calibration"
    MULTI_CONSTRAINT_COLLAPSE = "multi_constraint_collapse"


class DocumentRole(StrEnum):
    """Document archetypes used per domain."""

    POLICY_DENSE = "policy_dense"
    SCOPE_BOUNDARY = "scope_boundary"


class ParagraphSplit(StrEnum):
    """Whether a paragraph is held out for evaluation or reserved for training."""

    TEST = "test"
    TRAIN = "train"


class GenerationStrategy(StrEnum):
    """Synthetic Q&A generation prompting strategies."""

    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"
    FEW_SHOT = "few_shot"
    TOPIC_CONTROLLED = "topic_controlled"
    GROUNDED_RAG = "grounded_rag"


@dataclass
class DocumentSpec:
    """Metadata for a source policy document."""

    doc_id: str
    title: str
    role: DocumentRole
    domain: str
    source_url: str | None = None
    local_path: str | None = None


@dataclass
class Paragraph:
    """A chunk of source document text with train/test assignment."""

    doc_id: str
    para_id: str
    text: str
    role: DocumentRole
    split: ParagraphSplit
    index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "doc_id": self.doc_id,
            "para_id": self.para_id,
            "text": self.text,
            "role": self.role.value,
            "split": self.split.value,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paragraph:
        """Deserialize from a dictionary."""
        return cls(
            doc_id=data["doc_id"],
            para_id=data["para_id"],
            text=data["text"],
            role=DocumentRole(data["role"]),
            split=ParagraphSplit(data["split"]),
            index=data.get("index", 0),
        )


@dataclass
class QASample:
    """A question-answer pair used for evaluation or training."""

    id: str
    question: str
    gold_answer: str
    doc_id: str
    para_id: str
    context: str = ""
    failure_mode: FailureMode | None = None
    role: DocumentRole | None = None
    instruction: str = ""
    split: ParagraphSplit = ParagraphSplit.TEST
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = asdict(self)
        if self.failure_mode is not None:
            payload["failure_mode"] = self.failure_mode.value
        else:
            payload["failure_mode"] = None
        if self.role is not None:
            payload["role"] = self.role.value
        else:
            payload["role"] = None
        payload["split"] = self.split.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QASample:
        """Deserialize from a dictionary."""
        failure_mode = data.get("failure_mode")
        role = data.get("role")
        return cls(
            id=data["id"],
            question=data["question"],
            gold_answer=data["gold_answer"],
            doc_id=data["doc_id"],
            para_id=data["para_id"],
            context=data.get("context", ""),
            failure_mode=FailureMode(failure_mode) if failure_mode else None,
            role=DocumentRole(role) if role else None,
            instruction=data.get("instruction", ""),
            split=ParagraphSplit(data.get("split", ParagraphSplit.TEST.value)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class JudgeScore:
    """LLM-as-judge scores for a single model response."""

    sample_id: str
    correctness: float
    coherence: float
    instruction_following: float
    factual_plausibility: float
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def average(self) -> float:
        """Mean score across the four judge dimensions."""
        return (
            self.correctness
            + self.coherence
            + self.instruction_following
            + self.factual_plausibility
        ) / 4

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)
