"""Drive sampling, prompting, and verification to produce benchmark tasks."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from aieng.syn_data.synbench.generation.llm import (
    call_llm_json,
    is_mock_llm,
    load_mock_response,
)
from aieng.syn_data.synbench.generation.prompt import PromptBuilder
from aieng.syn_data.synbench.generation.sampler import (
    ConstraintSampler,
    SampleConstraints,
)
from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import Task
from aieng.syn_data.synbench.verification.pipeline import filter_verified


class TrajectoryGenerator:
    """Produce one draft ``Task`` per call from sampled domain constraints."""

    def __init__(self, domain: DomainBundle, seed: int = 42):
        self.domain = domain
        self.sampler = ConstraintSampler(domain, seed=seed)
        self.prompt_builder = PromptBuilder()

    def generate_one(
        self, task_id: str | None = None, constraints: SampleConstraints | None = None
    ) -> Task:
        """Generate a single draft task, sampling constraints when none are given."""
        if is_mock_llm():
            draft = load_mock_response()
            if task_id:
                draft = draft.model_copy(update={"id": task_id})
            return draft

        if constraints is None:
            constraints = self.sampler.sample()
        prompt = self.prompt_builder.build(self.domain, constraints)
        data = call_llm_json(prompt)
        if task_id:
            data["id"] = task_id
        if not data.get("task_type"):
            data["task_type"] = constraints.task_type
        return Task.model_validate(data)


class GenerationRun:
    """Batch generation, verification, and persistence of draft tasks."""

    def __init__(self, domain: DomainBundle, seed: int = 42):
        self.domain = domain
        self.generator = TrajectoryGenerator(domain, seed=seed)

    def run(self, n: int) -> list[Task]:
        """Generate ``n`` unverified draft tasks."""
        drafts: list[Task] = []
        for i in range(n):
            tid = f"gen_{uuid4().hex[:8]}_{i}"
            drafts.append(self.generator.generate_one(task_id=tid))
        return drafts

    def run_and_verify(self, n: int) -> tuple[list[Task], list[tuple[Task, list[str]]]]:
        """Generate ``n`` drafts and split them into verified tasks and rejections."""
        drafts = self.run(n)
        verified, rejected_drafts, _ = filter_verified(self.domain, drafts)
        rejections = [(r.task, errors) for r, errors in rejected_drafts]
        verified_tasks = [v.task for v in verified]
        return verified_tasks, rejections

    def verify_drafts(
        self, drafts: list[Task]
    ) -> tuple[list[Task], list[tuple[Task, list[str]]]]:
        """Split already-generated ``drafts`` into verified tasks and rejections."""
        verified, rejected_drafts, _ = filter_verified(self.domain, drafts)
        rejections = [(r.task, errors) for r, errors in rejected_drafts]
        verified_tasks = [v.task for v in verified]
        return verified_tasks, rejections

    def write_tasks(self, tasks: list[Task], out_dir: Path) -> Path:
        """Write ``tasks`` to ``out_dir/tasks.json`` and return that path."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "tasks.json"
        payload = {"tasks": [t.model_dump(mode="json") for t in tasks]}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path
