"""Verify draft tasks and filter a batch down to unique, valid ones."""

from __future__ import annotations

from aieng.syn_data.synbench.environment.core import ToolDispatchError, replay_actions
from aieng.syn_data.synbench.environment.hashing import db_hash
from aieng.syn_data.synbench.fsm.validator import (
    FSMValidationError,
    validate_actions_against_fsm,
)
from aieng.syn_data.synbench.schemas.actions import actions_fingerprint
from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import Task
from aieng.syn_data.synbench.schemas.verification import (
    VerificationReport,
    VerifiedTask,
)
from aieng.syn_data.synbench.verification.domain_checks import run_domain_checks
from aieng.syn_data.synbench.verification.errors import VerificationError


def verify_draft(domain: DomainBundle, draft: Task) -> VerifiedTask:
    """Run domain checks, FSM validation, and oracle replay on ``draft``."""
    errors: list[str] = []
    target_hash: str | None = None

    errors.extend(run_domain_checks(domain, draft))

    try:
        validate_actions_against_fsm(
            domain,
            draft.task_type,
            draft.evaluation_criteria.actions,
        )
    except FSMValidationError as e:
        errors.append(str(e))

    if not errors:
        try:
            env = replay_actions(domain, draft.evaluation_criteria.actions)
            target_hash = db_hash(env.db)
        except ToolDispatchError as e:
            errors.append(f"Replay failed: {e}")

    report = VerificationReport(
        passed=len(errors) == 0,
        errors=errors,
        target_db_hash=target_hash,
    )
    return VerifiedTask(task=draft, verification_report=report)


def verify_draft_or_raise(domain: DomainBundle, draft: Task) -> VerifiedTask:
    """Verify ``draft`` and raise ``VerificationError`` if it does not pass."""
    result = verify_draft(domain, draft)
    if not result.verification_report.passed:
        raise VerificationError(
            "Verification failed",
            errors=result.verification_report.errors,
        )
    return result


def filter_verified(
    domain: DomainBundle,
    drafts: list[Task],
    seen_hashes: set[str] | None = None,
) -> tuple[list[VerifiedTask], list[tuple[VerifiedTask, list[str]]], set[str]]:
    """
    Filter verified tasks from a list of draft tasks.

    Deduplicates by ordered oracle action fingerprint (not final DB hash),
    so distinct read-only tasks are not collapsed when they leave the DB unchanged.

    Args:
        domain: The domain bundle.
        drafts: The list of draft tasks to filter.
        seen_hashes: Previously seen action-sequence fingerprints. Defaults to None.

    Returns
    -------
        A tuple of verified tasks, rejected tasks, and seen fingerprints.
    """
    seen = seen_hashes or set()
    verified: list[VerifiedTask] = []
    # Store rejected tasks and the reason for rejection
    rejected: list[tuple[VerifiedTask, list[str]]] = []
    for draft in drafts:
        result = verify_draft(domain, draft)
        if not result.verification_report.passed:
            rejected.append((result, result.verification_report.errors))
            continue
        fp = actions_fingerprint(result.task.evaluation_criteria.actions)
        if fp in seen:
            rejected.append(
                (
                    result,
                    ["Duplicate actions: exact sequence of tool calls and arguments"],
                )
            )
            continue
        seen.add(fp)
        verified.append(result)
    return (verified, rejected, seen)
