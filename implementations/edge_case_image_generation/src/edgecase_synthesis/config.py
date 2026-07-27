"""Hydra configuration helpers."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from edgecase_synthesis.data import project_root


def configs_dir(start: Path | None = None) -> Path:
    return project_root(start) / "configs"


def load_config(
    overrides: list[str] | None = None,
    *,
    start: Path | None = None,
) -> DictConfig:
    """Compose the pipeline config with Hydra.

    Examples
    --------
    >>> cfg = load_config()
    >>> cfg = load_config(overrides=["hardware=gpu_l4"])  # NVIDIA L4 / g2-standard-8
    >>> cfg = load_config(overrides=["data/source@data=wikimedia"])
    >>> cfg = load_config(overrides=["data.max_images=3"])
    """
    root = project_root(start)
    GlobalHydra.instance().clear()

    with initialize_config_dir(
        config_dir=str(configs_dir(root)),
        version_base=None,
    ):
        cfg = compose(config_name="config", overrides=overrides or [])

    OmegaConf.set_struct(cfg, False)
    cfg.paths.project_root = str(root)
    OmegaConf.resolve(cfg)
    return cfg


def anomalies_dir(dataset: str, *, start: Path | None = None) -> Path:
    """``configs/generation/anomalies/<dataset>/``."""
    return configs_dir(start) / "generation" / "anomalies" / dataset


def list_anomalies(dataset: str, *, start: Path | None = None) -> list[str]:
    """Return anomaly ids (yaml stems) available for a dataset."""
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
    """Load one anomaly YAML (prompts, edit mask, optional hyperparam overrides)."""
    path = anomalies_dir(dataset, start=start) / f"{anomaly_id}.yaml"
    if not path.exists():
        available = ", ".join(list_anomalies(dataset, start=start)) or "(none)"
        raise FileNotFoundError(
            f"Unknown anomaly {dataset}/{anomaly_id!r}. Available: {available}"
        )
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def merge_generation_anomaly(generation: DictConfig | dict, anomaly: DictConfig | dict) -> DictConfig:
    """Merge shared generation settings with anomaly-specific overrides.

    Anomaly fields win when set. Nested ``controlnet_scale`` is merged key-wise.
    """
    base = OmegaConf.create(OmegaConf.to_container(generation, resolve=True))
    anom = OmegaConf.create(OmegaConf.to_container(anomaly, resolve=True))
    OmegaConf.set_struct(base, False)

    for key in (
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
    ):
        if key in anom and anom[key] is not None:
            base[key] = anom[key]

    if "controlnet_scale" in anom and anom.controlnet_scale is not None:
        scales = OmegaConf.create(
            OmegaConf.to_container(base.get("controlnet_scale", {"depth": 0.55, "seg": 0.40}), resolve=True)
        )
        OmegaConf.set_struct(scales, False)
        override = anom.controlnet_scale
        if isinstance(override, (int, float)):
            scales.depth = float(override)
            scales.seg = float(override)
        else:
            for sk in ("depth", "seg"):
                if sk in override and override[sk] is not None:
                    scales[sk] = float(override[sk])
        base.controlnet_scale = scales

    base.anomaly = anom
    return base


def config_to_dict(cfg: DictConfig) -> dict:
    """Convert a composed config to a plain dict (paths resolved)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
