"""Dispatch to a domain's optional ``verify.py`` draft checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import Task


CHECK_FN = "check_domain_rules"


def _load_verify_module(verify_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"domain_verify_{verify_path.parent.name}",
        verify_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load domain verify module: {verify_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_domain_checks(domain: DomainBundle, draft: Task) -> list[str]:
    """Run optional domain-specific draft checks from ``verify.py``."""
    verify_path = domain.root / "verify.py"
    if not verify_path.exists():
        return []

    module = _load_verify_module(verify_path)
    check_fn = getattr(module, CHECK_FN, None)
    if check_fn is None:
        return []
    return list(check_fn(domain, draft))
