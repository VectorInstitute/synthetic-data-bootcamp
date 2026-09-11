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
    if hasattr(raw, "items") or isinstance(raw, dict):
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
    """Format a template with variation values and optional extras."""
    mapping = {**(extras or {}), **values}

    # Prefer str.format_map so missing keys can be left alone if needed.
    class _Safe(dict[str, str]):
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
        container = OmegaConf.to_container(cfg, resolve=True)
        return {str(key): value for key, value in container.items()} if isinstance(container, dict) else {}
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


def _variation_template(anom: Any, block: Any) -> str | None:
    """Resolve the method-level or anomaly-level prompt template."""
    raw = _get(block, "prompt_template") if block is not None else None
    if not raw:
        raw = _get(anom, "prompt_template")
    return str(raw).strip() if raw else None


def _variation_extras(anom: Any) -> dict[str, str]:
    """Collect optional anomaly fields exposed to prompt templates."""
    extras: dict[str, str] = {}
    for key in ("description", "display_name", "id"):
        value = _get(anom, key)
        if value:
            extras[key] = str(value).strip()
    return extras


def _variation_negative(
    anom: Any,
    block: Any,
    values: dict[str, str],
    extras: dict[str, str],
    base_negative: str,
) -> str:
    """Render a negative template or preserve the static negative prompt."""
    raw = _get(block, "negative_prompt_template") if block is not None else None
    if not raw:
        raw = _get(anom, "negative_prompt_template")
    if raw:
        return render_template(str(raw).strip(), values, extras=extras)
    return str(base_negative or "").strip()


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

    template = _variation_template(anom, block)
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
    extras = _variation_extras(anom)
    prompt = render_template(template, values, extras=extras)
    negative = _variation_negative(anom, block, values, extras, base_negative)

    return PromptVariation(
        prompt=prompt,
        negative_prompt=negative,
        values=values,
        index=int(variation_index),
        n_combos=n_combos,
        template_used=True,
    )
