"""Flight pre-check helpers for Notebook 0 (and optional reuse elsewhere).

Checks env, hardware, samples, and API reachability without loading heavy
diffusion / detector weights. Safe to run on a laptop before NB1–NB3.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Status = Literal["pass", "warn", "fail", "skip"]


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str
    fix: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "warn", "skip"}


@dataclass
class PreflightReport:
    results: list[CheckResult] = field(default_factory=list)
    recommended_hardware: str = "cpu"

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "warn"]

    @property
    def ready(self) -> bool:
        return not self.failed

    def print_table(self) -> None:
        width = max((len(r.name) for r in self.results), default=8)
        icon = {"pass": "OK", "warn": "!!", "fail": "XX", "skip": "--"}
        print(f"{'STATUS':<6}  {'CHECK':<{width}}  DETAIL")
        print(f"{'-' * 6}  {'-' * width}  {'-' * 40}")
        for r in self.results:
            print(f"{icon[r.status]:<6}  {r.name:<{width}}  {r.detail}")
            if r.fix and r.status in {"fail", "warn"}:
                print(f"{'':6}  {'':<{width}}  → {r.fix}")
        print()
        if self.ready:
            print(
                f"Preflight READY — recommended HARDWARE=\"{self.recommended_hardware}\""
            )
        else:
            print("Preflight BLOCKED — fix the XX rows before Notebook 1.")


def _mask_key(key: str | None) -> str:
    if not key:
        return "(missing)"
    if len(key) <= 8:
        return key[:2] + "…"
    return f"{key[:4]}…{key[-4:]} (len={len(key)})"


def check_imports() -> CheckResult:
    missing: list[str] = []
    for mod in ("torch", "transformers", "PIL", "omegaconf", "hydra", "openai"):
        try:
            __import__(mod if mod != "PIL" else "PIL.Image")
        except ImportError:
            missing.append(mod)
    if missing:
        return CheckResult(
            "python packages",
            "fail",
            f"missing: {', '.join(missing)}",
            "From repo root: uv sync --dev --group edge-case-image-generation",
        )
    import torch

    return CheckResult(
        "python packages",
        "pass",
        f"torch {torch.__version__}, transformers/openai/hydra OK",
    )


def check_env_file(project_root: Path) -> CheckResult:
    env_path = project_root / ".env"
    example = project_root / ".env.example"
    if env_path.is_file():
        return CheckResult("env file", "pass", str(env_path))
    return CheckResult(
        "env file",
        "fail",
        f"no .env at {env_path}",
        f"cp {example.name} .env  then paste your vp_… key into OPENAI_API_KEY",
    )


def check_proxy_key() -> CheckResult:
    key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("VECTOR_PROXY_API_KEY")
        or ""
    ).strip()
    if not key:
        return CheckResult(
            "Vector API key",
            "fail",
            "OPENAI_API_KEY / VECTOR_PROXY_API_KEY not set",
            "Put your Vector proxy key (vp_…) in implementations/edge_case_image_generation/.env",
        )
    if not key.startswith("vp_") and not key.startswith("sk-"):
        return CheckResult(
            "Vector API key",
            "warn",
            f"unexpected prefix: {_mask_key(key)}",
            "Workshop keys usually start with vp_",
        )
    return CheckResult("Vector API key", "pass", _mask_key(key))


def check_proxy_chat(
    *,
    model: str = "gemini-3-flash-preview",
    api_base_url: str | None = None,
) -> CheckResult:
    """Tiny text-only call — proves the Vector proxy key works."""
    try:
        from edgecase_synthesis.vlm_api import make_openai_client, resolve_proxy_base_url
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Vector proxy chat", "fail", f"import error: {exc}")

    base = resolve_proxy_base_url(api_base_url)
    try:
        client = make_openai_client(base_url=base)
        # Prefer a models list if available; fall back to a tiny chat.
        try:
            models = client.models.list()
            ids = [getattr(m, "id", "") for m in getattr(models, "data", []) or []]
            sample = ", ".join(ids[:5]) if ids else "(empty list)"
            return CheckResult(
                "Vector proxy chat",
                "pass",
                f"{base} reachable; sample models: {sample}",
            )
        except Exception:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=8,
            )
            text = (resp.choices[0].message.content or "").strip()
            return CheckResult(
                "Vector proxy chat",
                "pass",
                f"{base} chat OK ({model}) → {text!r}",
            )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Vector proxy chat",
            "fail",
            f"{base}: {type(exc).__name__}: {exc}",
            "Check OPENAI_API_KEY and network access to proxy.vectorinstitute.ai",
        )


def probe_hardware() -> tuple[CheckResult, str, dict[str, Any]]:
    """Return (check, recommended_hardware_name, facts)."""
    import torch

    facts: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if not torch.cuda.is_available():
        return (
            CheckResult(
                "GPU / CUDA",
                "warn",
                "CUDA not available — use HARDWARE=\"cpu\" (slow edits)",
                "For workshop speed, run on a Vertex / cloud L4 notebook",
            ),
            "cpu",
            facts,
        )

    names: list[str] = []
    vrams: list[float] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gb = float(props.total_memory) / (1024**3)
        names.append(props.name)
        vrams.append(gb)
        facts["devices"].append({"index": i, "name": props.name, "vram_gb": round(gb, 1)})

    n = len(names)
    min_vram = min(vrams) if vrams else 0.0
    label = ", ".join(f"{nm} ({gb:.0f} GB)" for nm, gb in zip(names, vrams))
    if n >= 2 and min_vram >= 20:
        hw = "gpu_l4x2"
    elif min_vram >= 20:
        hw = "gpu_l4"
    else:
        hw = "gpu_l4" if min_vram >= 12 else "cpu"
        return (
            CheckResult(
                "GPU / CUDA",
                "warn",
                f"{n}× GPU: {label} — VRAM may be tight for Klein-4B",
                f'Try HARDWARE="{hw}" or fall back to "cpu"',
            ),
            hw if min_vram >= 12 else "cpu",
            facts,
        )

    return (
        CheckResult("GPU / CUDA", "pass", f"{n}× GPU: {label} → recommend {hw}"),
        hw,
        facts,
    )


def check_disk(path: Path, *, min_free_gb: float = 15.0) -> CheckResult:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        return CheckResult(
            "disk space",
            "warn",
            f"{free_gb:.1f} GB free under {path}",
            f"Klein / Depth / YOLO weights need headroom (≥{min_free_gb:.0f} GB recommended)",
        )
    return CheckResult("disk space", "pass", f"{free_gb:.1f} GB free")


def check_samples(project_root: Path, *, dataset_name: str = "mapillary_vistas") -> CheckResult:
    from edgecase_synthesis.config import load_config
    from edgecase_synthesis.data import list_sample_images

    cfg = load_config(
        start=project_root,
        overrides=[f"dataset_name={dataset_name}", "hardware=cpu"],
    )
    samples_dir = Path(cfg.paths.samples_dir)
    paths = list_sample_images(samples_dir)
    scenes = [p for p in paths if p.stem.startswith("scene_")]
    tagged = [p for p in paths if not p.stem.startswith("scene_")]
    if not paths:
        return CheckResult(
            "sample images",
            "fail",
            f"empty: {samples_dir}",
            "Run the data cell in Notebook 0 (ensure_mapillary_samples) "
            "or: uv run python scripts/extract_mapillary_toy.py",
        )
    return CheckResult(
        "sample images",
        "pass",
        f"{len(paths)} images in {samples_dir.name}/ "
        f"({len(scenes)} scene_ seeds, {len(tagged)} tagged)",
    )


def ensure_workshop_data(
    project_root: Path,
    *,
    dataset_name: str = "mapillary_vistas",
    clean: bool = False,
    min_images: int = 1,
) -> list[Path]:
    """Download / verify workshop samples (Mapillary toy extract with tqdm)."""
    if dataset_name != "mapillary_vistas":
        # Other datasets still use prepare_sample_images / local folders.
        from edgecase_synthesis.config import load_config
        from edgecase_synthesis.data import list_sample_images, prepare_sample_images

        cfg = load_config(
            start=project_root,
            overrides=[f"dataset_name={dataset_name}", "hardware=cpu"],
        )
        prepare_sample_images(cfg=cfg, force=clean)
        return list_sample_images(Path(cfg.paths.samples_dir))

    from edgecase_synthesis.mapillary_extract import ensure_mapillary_samples

    return ensure_mapillary_samples(
        project_root, clean=clean, min_images=min_images
    )


def check_hf_token() -> CheckResult:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    home = Path.home() / ".cache" / "huggingface" / "token"
    try:
        from huggingface_hub import get_token

        hub_token = get_token()
    except Exception:
        hub_token = None
    if token or hub_token or home.is_file():
        return CheckResult(
            "Hugging Face login",
            "pass",
            "token present (needed for gated Mapillary + some model weights)",
        )
    return CheckResult(
        "Hugging Face login",
        "warn",
        "no HF token detected",
        "Set HF_TOKEN in .env, or run huggingface-cli login / Notebook 0 interactive login",
    )


def _hub_cached(repo_id: str) -> bool | None:
    """True if any snapshot exists locally; None if hub helpers unavailable."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    # Prefer checking config.json / model_index.json presence in cache.
    for filename in ("model_index.json", "config.json", "preprocessor_config.json"):
        try:
            path = try_to_load_from_cache(repo_id, filename)
            if path is not None and path != "None":
                return True
        except Exception:
            continue
    # Fallback: look under HF_HOME hub folder.
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    safe = "models--" + repo_id.replace("/", "--")
    snap = hub / safe / "snapshots"
    if snap.is_dir() and any(snap.iterdir()):
        return True
    return False


def check_model_availability(project_root: Path, *, hardware: str) -> CheckResult:
    from edgecase_synthesis.config import load_config

    cfg = load_config(
        start=project_root,
        overrides=[f"dataset_name=mapillary_vistas", f"hardware={hardware}"],
    )
    repos = [
        ("depth", str(cfg.conditioning.depth.model_id)),
        ("seg", str(cfg.conditioning.segmentation.model_name)),
        ("instruct", str(cfg.generation.instruct_model_id)),
        ("judge", str(cfg.judge.model_id)),
        ("embed", str((cfg.judge.get("embedding_gate") or {}).get("model_id") or "openai/clip-vit-base-patch32")),
    ]
    lines: list[str] = []
    cached_n = 0
    unknown = 0
    for role, repo in repos:
        if role == "judge" and str(cfg.judge.backend).lower() == "api":
            lines.append(f"{role}: {repo} (API — no local weights)")
            continue
        if "/" not in repo and repo.endswith(".pt"):
            # Ultralytics weight name — check common cache locations lightly.
            ultra = Path.home() / ".cache" / "ultralytics" / repo
            cwd_pt = Path.cwd() / repo
            hit = ultra.is_file() or cwd_pt.is_file()
            cached_n += int(hit)
            lines.append(f"{role}: {repo} [{'cached' if hit else 'will download on first use'}]")
            continue
        hit = _hub_cached(repo)
        if hit is True:
            cached_n += 1
            lines.append(f"{role}: {repo} [cached]")
        elif hit is False:
            lines.append(f"{role}: {repo} [not cached — downloads on first use]")
        else:
            unknown += 1
            lines.append(f"{role}: {repo} [cache status unknown]")

    detail = f"profile={hardware}; " + "; ".join(lines)
    status: Status = "pass" if unknown == 0 else "warn"
    return CheckResult(
        "models (config)",
        status,
        detail,
        "First NB1 run downloads missing HF / Ultralytics weights into the cache",
    )


def check_yolo_weight() -> CheckResult:
    name = "yolov8s-worldv2.pt"
    candidates = [
        Path.cwd() / name,
        Path.home() / ".cache" / "ultralytics" / name,
    ]
    hit = next((p for p in candidates if p.is_file()), None)
    if hit:
        return CheckResult("YOLO-World weight", "pass", f"found {hit}")
    return CheckResult(
        "YOLO-World weight",
        "warn",
        f"{name} not cached yet",
        "Ultralytics downloads it automatically on first annotate call",
    )


def run_preflight(
    project_root: Path,
    *,
    dataset_name: str = "mapillary_vistas",
    ping_proxy: bool = True,
    hardware_override: str | None = None,
) -> PreflightReport:
    """Run the full checklist and return a printable report."""
    from edgecase_synthesis.config import load_env

    load_env(project_root)
    report = PreflightReport()

    report.add(check_imports())
    report.add(check_env_file(project_root))
    report.add(check_proxy_key())
    report.add(check_hf_token())
    report.add(check_disk(project_root))

    gpu_check, recommended, _facts = probe_hardware()
    report.add(gpu_check)
    report.recommended_hardware = hardware_override or recommended

    report.add(check_samples(project_root, dataset_name=dataset_name))
    report.add(
        check_model_availability(project_root, hardware=report.recommended_hardware)
    )
    report.add(check_yolo_weight())

    if ping_proxy and not any(r.name == "Vector API key" and r.status == "fail" for r in report.results):
        report.add(check_proxy_chat())
    elif ping_proxy:
        report.add(
            CheckResult(
                "Vector proxy chat",
                "skip",
                "skipped — fix API key first",
            )
        )

    return report
