"""Load and prepare real images for the pipeline."""

from __future__ import annotations

import shutil
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from edgecase_synthesis.image_quality import is_usable_sample_image

if TYPE_CHECKING:
    from collections.abc import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
USER_AGENT = "EdgeCaseSynthesis-Bootcamp/0.1 (educational)"
ZENODO_RECORD_API = "https://zenodo.org/api/records/{record_id}"
NORDLAND_HF_ROWS_API = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True)
class ImageSample:
    """A single real image ready for structure extraction."""

    path: Path
    image: Image.Image
    name: str

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


@dataclass(frozen=True)
class DataSourceInfo:
    """Resolved metadata for the active image source."""

    name: str
    label: str
    license: str
    attribution: str


def project_root(start: Path | None = None) -> Path:
    """Walk up from *start* until we find pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root (no pyproject.toml found). "
        "Run notebooks from the repo or set PROJECT_ROOT."
    )


def default_samples_dir(root: Path | None = None) -> Path:
    return project_root(root) / "data" / "samples"


def _as_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def get_data_source_info(cfg: DictConfig | dict[str, Any]) -> DataSourceInfo:
    data = _as_dict(cfg)["data"]
    return DataSourceInfo(
        name=data["name"],
        label=data.get("label", data["name"]),
        license=data.get("license", ""),
        attribution=data.get("attribution", ""),
    )


def prepare_sample_images(
    samples_dir: Path | None = None,
    *,
    cfg: DictConfig | dict[str, Any] | None = None,
    force: bool = False,
) -> list[Path]:
    """Populate samples_dir from the Hydra-configured image source."""
    if cfg is None:
        from edgecase_synthesis.config import load_config

        cfg = load_config()

    config = _as_dict(cfg)
    target = (samples_dir or Path(config["paths"]["samples_dir"])).resolve()
    target.mkdir(parents=True, exist_ok=True)

    source_name = config["data"]["name"]
    source_cfg = config["data"]

    if not force and _source_samples_complete(target, source_name, source_cfg):
        return list_sample_images(target)

    if source_name == "local":
        existing = list_sample_images(target)
        if existing:
            return existing
        raise FileNotFoundError(
            f"No images in {target}. Add files manually or switch source with "
            "load_config(overrides=['data/source@data=wikimedia'])."
        )
    if source_name == "l4r_nlb":
        return _prepare_l4r_samples(target, source_cfg, force=force)
    if source_name == "nordland_hf":
        return _prepare_nordland_hf_samples(target, source_cfg, force=force)
    if source_name == "railsem19":
        return _prepare_railsem19_samples(target, source_cfg, force=force)
    if source_name == "wikimedia":
        return _prepare_wikimedia_samples(target, source_cfg, force=force)

    raise ValueError(f"Unsupported image source: {source_name}")


def _source_samples_complete(
    target: Path,
    source_name: str,
    source_cfg: dict[str, Any],
) -> bool:
    if source_name == "local":
        return bool(list_sample_images(target))
    if source_name == "wikimedia":
        expected = [target / name for name in source_cfg.get("urls", {})]
        return bool(expected) and all(path.exists() for path in expected)
    if source_name == "l4r_nlb":
        expected = [target / frame["save_as"] for frame in source_cfg.get("frames", [])]
        return bool(expected) and all(path.exists() for path in expected)
    if source_name == "nordland_hf":
        expected = [target / frame["save_as"] for frame in source_cfg.get("frames", [])]
        if not expected or not all(path.exists() for path in expected):
            return False
        quality_cfg = source_cfg.get("quality", {})
        quality_kwargs = {
            "min_luminance_std": float(quality_cfg.get("min_luminance_std", 12.0)),
            "min_luminance_mean": float(quality_cfg.get("min_luminance_mean", 18.0)),
            "min_luminance_range": float(quality_cfg.get("min_luminance_range", 25.0)),
        }
        return all(
            is_usable_sample_image(load_image(path), **quality_kwargs)
            for path in expected
        )
    if source_name == "railsem19":
        expected = [target / frame["save_as"] for frame in source_cfg.get("frames", [])]
        return bool(expected) and all(path.exists() for path in expected)
    return False


def _clear_stale_sample_images(target: Path, keep: set[str]) -> None:
    """Remove images in samples_dir that are not part of the active source."""
    for path in list_sample_images(target):
        if path.name not in keep:
            path.unlink(missing_ok=True)


def _prepare_railsem19_samples(
    target: Path,
    source_cfg: dict[str, Any],
    *,
    force: bool,
) -> list[Path]:
    frames = source_cfg.get("frames", [])
    if not frames:
        raise ValueError("railsem19 source has no frames configured.")

    dataset_root = Path(source_cfg["dataset_root"])
    images_dir = dataset_root / source_cfg.get("images_subdir", "jpgs/rs19_val")
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"RailSem19 images not found at {images_dir}. "
            "Place the rs19_val release under data/rs19_val/."
        )

    expected_names = {frame["save_as"] for frame in frames}
    _clear_stale_sample_images(target, expected_names)

    copied: list[Path] = []
    missing: list[str] = []

    for frame in frames:
        frame_id = frame["frame_id"]
        save_as = frame["save_as"]
        dest = target / save_as
        src = images_dir / f"{frame_id}.jpg"
        if not src.exists():
            # Some releases use .JPG — keep a cheap fallback.
            src_alt = images_dir / f"{frame_id}.JPG"
            src = src_alt if src_alt.exists() else src

        if not src.exists():
            missing.append(frame_id)
            continue

        if dest.exists() and not force:
            copied.append(dest)
            continue

        shutil.copy2(src, dest)
        copied.append(dest)
        print(f"Copied {save_as} ← {src.name}")

    if missing:
        raise FileNotFoundError(
            "Missing RailSem19 frames: "
            + ", ".join(missing)
            + f"\nLooked under {images_dir}"
        )

    _write_attribution(target, source_cfg)
    return sorted(
        p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _prepare_nordland_hf_samples(
    target: Path,
    source_cfg: dict[str, Any],
    *,
    force: bool,
) -> list[Path]:
    frames = source_cfg.get("frames", [])
    if not frames:
        raise ValueError("nordland_hf source has no frames configured.")

    dataset_id = source_cfg.get("dataset_id", "Somayeh-h/Nordland")
    quality_cfg = source_cfg.get("quality", {})
    min_std = float(quality_cfg.get("min_luminance_std", 12.0))
    min_mean = float(quality_cfg.get("min_luminance_mean", 18.0))
    min_range = float(quality_cfg.get("min_luminance_range", 25.0))
    max_offset_skip = int(quality_cfg.get("max_offset_skip", 100))
    quality_kwargs = {
        "min_luminance_std": min_std,
        "min_luminance_mean": min_mean,
        "min_luminance_range": min_range,
    }

    downloaded: list[Path] = []

    for frame in frames:
        save_as = frame["save_as"]
        dest = target / save_as
        if dest.exists() and not force:
            cached = load_image(dest)
            if is_usable_sample_image(cached, **quality_kwargs):
                downloaded.append(dest)
                continue
            print(f"Replacing low-contrast cached image {save_as}")

        start_offset = int(frame["offset"])
        saved = False
        for step in range(max_offset_skip + 1):
            offset = start_offset + step
            key, image_bytes = _fetch_nordland_hf_image(dataset_id, offset)
            candidate = Image.open(BytesIO(image_bytes)).convert("RGB")
            if not is_usable_sample_image(candidate, **quality_kwargs):
                print(
                    f"Skipping low-contrast frame {key} (offset {offset}), trying next…"
                )
                continue

            dest.write_bytes(image_bytes)
            downloaded.append(dest)
            print(f"Saved {save_as} ({key}, offset {offset})")
            saved = True
            break

        if not saved:
            raise RuntimeError(
                f"Could not find a usable Nordland frame for {save_as} "
                f"starting at offset {start_offset} (searched {max_offset_skip + 1} rows)."
            )

    _write_attribution(
        target,
        {
            **source_cfg,
            "attribution": source_cfg.get("attribution", "")
            + f"\nDataset: https://huggingface.co/datasets/{dataset_id}",
        },
    )
    return sorted(
        p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _fetch_nordland_hf_image(dataset_id: str, offset: int) -> tuple[str, bytes]:
    params = {
        "dataset": dataset_id,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": 1,
    }
    response = _fetch_with_retries(
        NORDLAND_HF_ROWS_API,
        headers={"User-Agent": USER_AGENT},
        params=params,
    )
    rows = response.json().get("rows", [])
    if not rows:
        raise FileNotFoundError(
            f"No Nordland row at offset {offset} in {dataset_id}. "
            "Check offsets in configs/data/source/nordland_hf.yaml."
        )

    row = rows[0]["row"]
    key = row.get("__key__", f"offset-{offset}")
    png = row.get("png") or {}
    image_url = png.get("src")
    if not image_url:
        raise FileNotFoundError(f"Nordland row {key} has no image URL.")

    image_response = _fetch_with_retries(
        image_url,
        headers={"User-Agent": USER_AGENT},
    )
    return key, image_response.content


def _prepare_wikimedia_samples(
    target: Path,
    source_cfg: dict[str, Any],
    *,
    force: bool,
) -> list[Path]:
    headers = {"User-Agent": USER_AGENT}
    downloaded: list[Path] = []

    for filename, url in source_cfg.get("urls", {}).items():
        dest = target / filename
        if dest.exists() and not force:
            downloaded.append(dest)
            continue
        print(f"Downloading {filename} …")
        response = _fetch_with_retries(url, headers=headers)
        dest.write_bytes(response.content)
        downloaded.append(dest)

    return sorted(
        p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _prepare_l4r_samples(
    target: Path,
    source_cfg: dict[str, Any],
    *,
    force: bool,
) -> list[Path]:
    frames = source_cfg.get("frames", [])
    if not frames:
        raise ValueError("l4r_nlb source has no frames configured.")

    archive_dir = Path(source_cfg["archive_dir"])
    extract_dir = Path(source_cfg["extract_dir"])
    archive_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    seasons_needed = sorted({frame["season"] for frame in frames})
    if source_cfg.get("auto_download", True):
        for season in seasons_needed:
            _ensure_l4r_archive(season, source_cfg, archive_dir)

    copied: list[Path] = []
    missing: list[str] = []

    for frame in frames:
        season = frame["season"]
        frame_id = frame["frame_id"]
        save_as = frame["save_as"]
        dest = target / save_as

        if dest.exists() and not force:
            copied.append(dest)
            continue

        src = _locate_l4r_frame(extract_dir, season, frame_id)
        if src is None:
            src = _extract_l4r_frame_from_archive(
                archive_dir, extract_dir, season, frame_id
            )

        if src is None or not src.exists():
            missing.append(f"{season}/{frame_id}")
            continue

        shutil.copy2(src, dest)
        copied.append(dest)

    if missing:
        raise FileNotFoundError(
            "Could not extract L4R_NLB frames: "
            + ", ".join(missing)
            + f"\nCheck archives in {archive_dir} and frame IDs in configs/data/source/l4r_nlb.yaml."
        )

    _write_attribution(target, source_cfg)
    return sorted(
        p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _ensure_l4r_archive(
    season: str,
    source_cfg: dict[str, Any],
    archive_dir: Path,
) -> Path:
    zip_name = f"L4R_NLB_{season}.zip"
    zip_path = archive_dir / zip_name
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        return zip_path

    record_id = source_cfg["zenodo_record"]
    download_url, size_bytes = _zenodo_file_info(record_id, zip_name)
    size_gb = size_bytes / 1e9 if size_bytes else 0.0

    print(
        f"Downloading {zip_name} from Zenodo (~{size_gb:.1f} GB). "
        "This is a one-time download; only curated frames are extracted."
    )
    _download_file(download_url, zip_path, expected_size=size_bytes)
    return zip_path


def _zenodo_file_info(record_id: int | str, filename: str) -> tuple[str, int]:
    api_url = ZENODO_RECORD_API.format(record_id=record_id)
    response = _fetch_with_retries(api_url, headers={"User-Agent": USER_AGENT})
    payload = response.json()
    for entry in payload.get("files", []):
        if entry.get("key") == filename:
            return entry["links"]["self"], int(entry.get("size", 0))
    known = ", ".join(entry.get("key", "?") for entry in payload.get("files", []))
    raise FileNotFoundError(
        f"{filename} not found in Zenodo record {record_id}. Available: {known}"
    )


def _download_file(url: str, dest: Path, *, expected_size: int = 0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": USER_AGENT}

    with requests.get(url, stream=True, headers=headers, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", expected_size))
        downloaded = 0
        last_report = time.monotonic()

        with temp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if total and now - last_report >= 5:
                    pct = 100 * downloaded / total
                    print(f"  … {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.0f}%)")
                    last_report = now

    temp.replace(dest)


def _locate_l4r_frame(extract_dir: Path, season: str, frame_id: str) -> Path | None:
    candidates = [
        extract_dir / f"L4R_NLB_{season}" / "images" / f"{frame_id}.png",
        extract_dir / season / "images" / f"{frame_id}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _extract_l4r_frame_from_archive(
    archive_dir: Path,
    extract_dir: Path,
    season: str,
    frame_id: str,
) -> Path | None:
    zip_path = archive_dir / f"L4R_NLB_{season}.zip"
    if not zip_path.exists():
        return None

    member_suffix = f"images/{frame_id}.png"
    with zipfile.ZipFile(zip_path) as archive:
        member = next(
            (name for name in archive.namelist() if name.endswith(member_suffix)),
            None,
        )
        if member is None:
            return None

        out_dir = extract_dir / f"L4R_NLB_{season}" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{frame_id}.png"
        if not out_path.exists():
            with archive.open(member) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        return out_path


def _write_attribution(target: Path, source_cfg: dict[str, Any]) -> None:
    attribution_path = target / "ATTRIBUTION.txt"
    lines = [
        source_cfg.get("label", "L4R_NLB"),
        f"License: {source_cfg.get('license', 'CC BY 3.0')}",
        source_cfg.get("attribution", ""),
        f"Zenodo: {source_cfg.get('zenodo_url', '')}",
        f"GitHub annotations: {source_cfg.get('github_url', '')}",
    ]
    attribution_path.write_text("\n".join(line for line in lines if line), encoding="utf-8")


def download_sample_images(
    samples_dir: Path | None = None,
    *,
    force: bool = False,
    cfg: DictConfig | dict[str, Any] | None = None,
) -> list[Path]:
    """Backward-compatible alias for prepare_sample_images()."""
    return prepare_sample_images(samples_dir, cfg=cfg, force=force)


def _fetch_with_retries(
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, str | int] | None = None,
    attempts: int = 3,
) -> requests.Response:
    last_error: requests.RequestException | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=120,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def list_sample_images(samples_dir: Path | None = None) -> list[Path]:
    directory = (samples_dir or default_samples_dir()).resolve()
    if not directory.exists():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path: Path | str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_sample_images(
    samples_dir: Path | None = None,
    *,
    max_images: int | None = None,
    download_if_missing: bool = True,
    cfg: DictConfig | dict[str, Any] | None = None,
) -> list[ImageSample]:
    if cfg is None:
        from edgecase_synthesis.config import load_config

        cfg = load_config()

    config = _as_dict(cfg)
    directory = Path(samples_dir) if samples_dir else Path(config["paths"]["samples_dir"])
    paths = list_sample_images(directory)

    if not paths and download_if_missing:
        paths = prepare_sample_images(directory, cfg=cfg)

    if not paths:
        raise FileNotFoundError(
            f"No images found in {directory}. "
            "Add images manually or call prepare_sample_images()."
        )

    # Prefer curated frame order when the active source lists save_as names.
    frame_names = [
        frame["save_as"]
        for frame in config.get("data", {}).get("frames", [])
        if isinstance(frame, dict) and "save_as" in frame
    ]
    if frame_names:
        by_name = {path.name: path for path in paths}
        ordered = [by_name[name] for name in frame_names if name in by_name]
        if ordered:
            paths = ordered

    if max_images is None:
        max_images = config.get("data", {}).get("max_images", 5)

    if max_images is not None:
        paths = paths[:max_images]

    return [
        ImageSample(path=path, image=load_image(path), name=path.stem)
        for path in paths
    ]
