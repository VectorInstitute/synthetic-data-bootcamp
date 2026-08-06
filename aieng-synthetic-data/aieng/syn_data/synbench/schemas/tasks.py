"""Task schema: the customer scenario plus how a run is rewarded."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from aieng.syn_data.synbench.schemas.actions import Action


class RewardType(StrEnum):
    """Dimensions a task can be rewarded on."""

    DB = "DB"
    COMMUNICATE = "COMMUNICATE"


class EvaluationCriteria(BaseModel):
    """Oracle actions and required information used to score a run."""

    actions: list[Action] = Field(default_factory=list)
    communicate_info: list[str] = Field(default_factory=list)
    reward_basis: list[RewardType] = Field(
        default_factory=lambda: [RewardType.DB, RewardType.COMMUNICATE]
    )

    @field_validator("reward_basis", mode="before")
    @classmethod
    def _parse_reward_basis(cls, v: Any) -> list[RewardType]:
        if v is None:
            return [RewardType.DB, RewardType.COMMUNICATE]
        out: list[RewardType] = []
        for item in v:
            if isinstance(item, RewardType):
                out.append(item)
            else:
                out.append(RewardType(str(item)))
        return out


class UserScenario(BaseModel):
    """Customer-facing scenario for generation and user simulation.

    ``user_name`` and ``personality_style`` are separate so the same customer
    can be evaluated under different interaction styles. Full clean intent
    (ids, args) lives in ``instructions``; ``initial_message`` is a styled,
    often incomplete opener so multi-turn elicitation is needed.
    """

    user_name: str = ""  # e.g. "Alice Chen"
    personality_style: str = ""  # e.g. "rushed" (catalog key in user_simulator.yaml)
    instructions: str = ""  # full clean goal for the user simulator / oracle grounding
    initial_message: str = ""  # first utterance (style); prefer withholding ids/args

    @model_validator(mode="before")
    # TODO: remove this after migration is complete.
    @classmethod
    def _migrate_legacy_persona(cls, data: Any) -> Any:
        """Map old ``persona`` → ``user_name`` for backward compatibility."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        persona = data.pop("persona", None)
        if not data.get("user_name") and persona:
            data["user_name"] = persona
        return data


class Task(BaseModel):
    """One benchmark task: a user scenario plus its evaluation criteria."""

    id: str
    description: str = ""
    task_type: str = ""
    user_scenario: UserScenario = Field(default_factory=UserScenario)
    evaluation_criteria: EvaluationCriteria = Field(default_factory=EvaluationCriteria)
