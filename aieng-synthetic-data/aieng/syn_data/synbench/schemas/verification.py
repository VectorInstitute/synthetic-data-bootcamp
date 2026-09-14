"""Schema for the outcome of verifying a draft task."""

from pydantic import BaseModel, Field

from aieng.syn_data.synbench.schemas.tasks import Task


class VerificationReport(BaseModel):
    """Pass/fail outcome of verification, with errors and the target DB hash."""

    passed: bool
    errors: list[str] = Field(default_factory=list)
    target_db_hash: str | None = None
    warnings: list[str] = Field(default_factory=list)


class VerifiedTask(BaseModel):
    """A task paired with the report produced when it was verified."""

    task: Task
    verification_report: VerificationReport
