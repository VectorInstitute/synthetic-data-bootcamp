"""Pre-gen prompt novelty: axis cartesian product → shuffle → cycle.

Keeps short batches diverse across *all* axes (not nested-order starvation).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf


@dataclass
class PromptVariation:
    """One resolved prompt after sampling variation axes."""

    prompt: str
    negative_prompt: str
    values: dict[str, str] = field(default_factory=dict)
    index: int = 0
    n_combos: int = 0
    template_used: bool = False


def parse_variation_axes(raw: Any) -> dict[str, list[str]]:
    """Normalize anomaly ``variations`` YAML → ordered axis → value list."""
    if raw is None:
        return {}
    if hasattr(raw, "items"):
        items = list(raw.items())
    elif isinstance(raw, dict):
        items = list(raw.items())
    else:
        return {}
    axes: dict[str, list[str]] = {}
    for key, values in items:
        name = str(key).strip()
        if not name:
            continue
        if values is None:
            continue
        seq = list(values) if not isinstance(values, str) else [values]
        cleaned = [str(v).strip() for v in seq if str(v).strip()]
        if cleaned:
            axes[name] = cleaned
    return axes


def cartesian_combos(axes: dict[str, list[str]]) -> list[dict[str, str]]:
    """Full cartesian product as list of axis→value dicts (stable axis order)."""
    if not axes:
        return [{}]
    keys = list(axes.keys())
    pools = [axes[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*pools)]


def shuffled_combos(
    axes: dict[str, list[str]],
    *,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Cartesian product, then shuffle once (seeded)."""
    combos = cartesian_combos(axes)
    rng = random.Random(int(seed))
    rng.shuffle(combos)
    return combos


def combo_at(
    axes: dict[str, list[str]],
    index: int,
    *,
    seed: int = 42,
) -> tuple[dict[str, str], int]:
    """Pick combo at ``index`` from the shuffled cycle. Returns (values, n_combos)."""
    combos = shuffled_combos(axes, seed=seed)
    n = len(combos)
    if n == 0:
        return {}, 0
    # Full wraps keep cycling; reshuffle each full pass so wrap ≠ identical order.
    pass_idx = int(index) // n
    within = int(index) % n
    if pass_idx == 0:
        return dict(combos[within]), n
    # Deterministic reshuffle per pass.
    return dict(shuffled_combos(axes, seed=int(seed) + pass_idx)[within]), n


def render_template(template: str, values: dict[str, str], *, extras: dict[str, str] | None = None) -> str:
    """Format ``template`` with variation values (+ optional extras like description)."""
    mapping = {**(extras or {}), **values}
    # Prefer str.format_map so missing keys can be left alone if needed.
    class _Safe(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    rendered = template.format_map(_Safe(**mapping))
    # Collapse whitespace from YAML multiline blocks.
    return " ".join(rendered.split())


def _as_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        return dict(OmegaConf.to_container(cfg, resolve=True) or {})
    except Exception:
        return {}


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_prompt_variation(
    anomaly_cfg: Any,
    *,
    method: str | None = None,
    base_prompt: str = "",
    base_negative: str = "",
    variation_index: int = 0,
    seed: int = 42,
) -> PromptVariation:
    """Resolve prompt for one edit attempt.

    If the anomaly defines ``variations`` + a ``prompt_template`` (anomaly-level
    or method block), sample the shuffled cycle and render. Otherwise return the
    static ``base_prompt`` / ``base_negative`` unchanged.
    """
    anom = anomaly_cfg
    method_key = str(method or "").lower() or None
    methods = _get(anom, "methods")
    block = None
    if method_key and methods is not None:
        block = _get(methods, method_key)

    template = None
    if block is not None:
        raw_t = _get(block, "prompt_template")
        if raw_t:
            template = str(raw_t).strip()
    if not template:
        raw_t = _get(anom, "prompt_template")
        if raw_t:
            template = str(raw_t).strip()

    axes = parse_variation_axes(_get(anom, "variations"))
    if not template or not axes:
        return PromptVariation(
            prompt=str(base_prompt or "").strip(),
            negative_prompt=str(base_negative or "").strip(),
            values={},
            index=int(variation_index),
            n_combos=0,
            template_used=False,
        )

    values, n_combos = combo_at(axes, int(variation_index), seed=int(seed))
    extras: dict[str, str] = {}
    description = _get(anom, "description")
    if description:
        extras["description"] = str(description).strip()
    display = _get(anom, "display_name")
    if display:
        extras["display_name"] = str(display).strip()
    anomaly_id = _get(anom, "id")
    if anomaly_id:
        extras["id"] = str(anomaly_id).strip()

    prompt = render_template(template, values, extras=extras)
    # Optional negative template (rare); else keep method/anomaly negative as-is.
    neg_template = None
    if block is not None:
        raw_n = _get(block, "negative_prompt_template")
        if raw_n:
            neg_template = str(raw_n).strip()
    if not neg_template:
        raw_n = _get(anom, "negative_prompt_template")
        if raw_n:
            neg_template = str(raw_n).strip()
    if neg_template:
        negative = render_template(neg_template, values, extras=extras)
    else:
        negative = str(base_negative or "").strip()

    return PromptVariation(
        prompt=prompt,
        negative_prompt=negative,
        values=values,
        index=int(variation_index),
        n_combos=n_combos,
        template_used=True,
    )
