"""Hydra configuration helpers + dataset package loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from edgecase_synthesis.data import project_root


def configs_dir(start: Path | None = None) -> Path:
    return project_root(start) / "configs"


def datasets_dir(start: Path | None = None) -> Path:
    return configs_dir(start) / "datasets"


def merge_skip_null(base: Any, overlay: Any) -> Any:
    """Deep-merge overlay into base; ``null`` / ``None`` means keep base value."""
    base_cfg = OmegaConf.create(
        OmegaConf.to_container(base, resolve=False) if OmegaConf.is_config(base) else base
    )
    OmegaConf.set_struct(base_cfg, False)
    if overlay is None:
        return base_cfg
    if OmegaConf.is_config(overlay):
        over_obj = OmegaConf.to_container(overlay, resolve=False)
    else:
        over_obj = overlay
    if not isinstance(over_obj, dict):
        return OmegaConf.create(over_obj)

    for key, val in over_obj.items():
        if val is None:
            continue
        if (
            key in base_cfg
            and OmegaConf.is_dict(base_cfg[key])
            and isinstance(val, dict)
        ):
            base_cfg[key] = merge_skip_null(base_cfg[key], val)
        else:
            base_cfg[key] = val
    return base_cfg


def _load_yaml(path: Path) -> DictConfig:
    if not path.exists():
        return OmegaConf.create({})
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg  # type: ignore[return-value]


def list_dataset_names(*, start: Path | None = None) -> list[str]:
    root = datasets_dir(start)
    if not root.exists():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "_template"
    )


def load_dataset_package(dataset_name: str, *, start: Path | None = None) -> DictConfig:
    """Load configs/datasets/<name>/{dataset,data,annotation,generation/default}.yaml."""
    root = datasets_dir(start) / dataset_name
    if not root.is_dir():
        available = ", ".join(list_dataset_names(start=start)) or "(none)"
        raise FileNotFoundError(
            f"Unknown dataset package {dataset_name!r}. Available: {available}. "
            f"Copy configs/datasets/_template to configs/datasets/{dataset_name}."
        )
    return OmegaConf.create(
        {
            "dataset": _load_yaml(root / "dataset.yaml"),
            "data": _load_yaml(root / "data.yaml"),
            "annotation": _load_yaml(root / "annotation.yaml"),
            "generation": _load_yaml(root / "generation" / "default.yaml"),
        }
    )


def load_config(
    overrides: list[str] | None = None,
    *,
    start: Path | None = None,
) -> DictConfig:
    """Compose shared Hydra config, then merge the selected dataset package.

    Examples
    --------
    >>> cfg = load_config()
    >>> cfg = load_config(overrides=["hardware=gpu_l4"])
    >>> cfg = load_config(overrides=["dataset_name=rdd2022"])
    >>> cfg = load_config(overrides=["dataset_name=nordland_hf", "hardware=gpu_l4"])
    """
    root = project_root(start)
    GlobalHydra.instance().clear()

    with initialize_config_dir(
        config_dir=str(configs_dir(root)),
        version_base=None,
    ):
        cfg = compose(config_name="config", overrides=overrides or [])

    OmegaConf.set_struct(cfg, False)
    # Drop nulls introduced by hardware "documentation" nulls.
    cfg = _prune_nulls(cfg)

    dataset_name = str(cfg.get("dataset_name", "mapillary_vistas"))
    pkg = load_dataset_package(dataset_name, start=root)
    cfg.dataset = pkg.dataset
    cfg.data = merge_skip_null(cfg.get("data", {}), pkg.data)
    cfg.annotation = merge_skip_null(cfg.get("annotation", {}), pkg.annotation)
    cfg.generation = merge_skip_null(cfg.get("generation", {}), pkg.generation)

    cfg.paths.project_root = str(root)
    # Re-resolve after dataset_name / paths are final.
    OmegaConf.resolve(cfg)
    cfg = _prune_nulls(cfg)
    return cfg


def _prune_nulls(cfg: Any) -> Any:
    """Remove keys whose value is None so str(cfg.x) never sees explicit nulls."""
    if not OmegaConf.is_config(cfg):
        return cfg
    container = OmegaConf.to_container(cfg, resolve=False)
    cleaned = _prune_nulls_obj(container)
    out = OmegaConf.create(cleaned)
    OmegaConf.set_struct(out, False)
    return out


def _prune_nulls_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _prune_nulls_obj(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_prune_nulls_obj(v) for v in obj]
    return obj


def anomalies_dir(dataset: str, *, start: Path | None = None) -> Path:
    """``configs/datasets/<dataset>/generation/anomalies/``."""
    return datasets_dir(start) / dataset / "generation" / "anomalies"


def list_anomalies(dataset: str, *, start: Path | None = None) -> list[str]:
    root = anomalies_dir(dataset, start=start)
    if not root.exists():
        return []
    return sorted(path.stem for path in root.glob("*.yaml") if not path.name.startswith("_"))


def load_anomaly(
    dataset: str,
    anomaly_id: str,
    *,
    start: Path | None = None,
) -> DictConfig:
    path = anomalies_dir(dataset, start=start) / f"{anomaly_id}.yaml"
    if not path.exists():
        available = ", ".join(list_anomalies(dataset, start=start)) or "(none)"
        raise FileNotFoundError(
            f"Unknown anomaly {dataset}/{anomaly_id!r}. Available: {available}"
        )
    return _load_yaml(path)


def merge_generation_anomaly(
    generation: DictConfig | dict,
    anomaly: DictConfig | dict,
    *,
    method: str | None = None,
) -> DictConfig:
    """Merge shared generation + anomaly + optional method block (null-safe)."""
    base = merge_skip_null(OmegaConf.create({}), generation)
    anom = OmegaConf.create(OmegaConf.to_container(anomaly, resolve=True))
    OmegaConf.set_struct(anom, False)

    # Shared anomaly fields (excluding nested methods / metadata-only).
    shared_keys = (
        "prompt",
        "negative_prompt",
        "strength",
        "controlnet_strength",
        "guidance_scale",
        "num_inference_steps",
        "seed",
        "max_side",
        "family",
        "vae_id",
        "default_anomaly_method",
        "instruct_image_guidance",
        "instruct_guidance_scale",
        "instruct_num_inference_steps",
        "inpaint_num_inference_steps",
        "inpaint_guidance_scale",
        "padding_mask_crop",
        "controlnet_scale",
        "edit_mask",
    )
    shared = OmegaConf.create({})
    OmegaConf.set_struct(shared, False)
    for key in shared_keys:
        if key in anom and anom[key] is not None:
            shared[key] = anom[key]
    base = merge_skip_null(base, shared)

    method_key = str(method or "").lower() or None
    methods = anom.get("methods")
    if method_key and methods is not None and method_key in methods:
        block = methods[method_key]
        if block is not None:
            base = merge_skip_null(base, block)

    base.anomaly = anom
    return base  # type: ignore[return-value]


def resolve_method_prompt(merged: DictConfig | dict, method: str) -> tuple[str, str]:
    """Return (prompt, negative_prompt) after method merge; apply CN fidelity if needed."""
    prompt = str(merged.get("prompt") or "").strip()
    negative = str(merged.get("negative_prompt") or "").strip()
    method = str(method).lower()
    if method == "controlnet_dual":
        # If method block did not supply a full prompt, append shared fidelity suffix.
        anom = merged.get("anomaly") or {}
        methods = anom.get("methods") if hasattr(anom, "get") else None
        block = methods.get(method) if methods is not None else None
        block_prompt = None
        if block is not None and "prompt" in block:
            block_prompt = block.get("prompt")
        if block_prompt is None:
            suffix = str(merged.get("controlnet_fidelity_suffix") or "").strip()
            if suffix:
                prompt = f"{prompt}, {suffix}" if prompt else suffix
            anti = str(merged.get("controlnet_anti_restyle") or "").strip()
            if anti:
                negative = f"{negative}, {anti}" if negative else anti
    return prompt, negative


def config_to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
