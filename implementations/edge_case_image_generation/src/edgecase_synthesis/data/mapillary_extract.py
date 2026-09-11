"""Mapillary Vistas toy-subset extract (Range GETs — not the full ~29GB zip).

Public entry points:
  - ``ensure_mapillary_samples`` — check / extract for Notebook 0
  - ``extract_mapillary_toy`` — always run extract (CLI / force refresh)

Requires a Hugging Face token (``HF_TOKEN`` / ``huggingface-cli login``) and
access to ``candylion/mapillary-vistas-v2``.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import time
import urllib.request
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError

from huggingface_hub import get_token, hf_hub_url
from PIL import Image
from tqdm.auto import tqdm

from edgecase_synthesis.config import load_config
from edgecase_synthesis.data.loader import list_sample_images, project_root


# Transient HF / CDN failures — retry with backoff instead of aborting mid-extract.
_RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}

_READABLE_DEFAULTS = {
    "object--traffic-cone": "Traffic Cone",
    "object--trash-can": "Trash Can",
    "animal--ground-animal": "Ground Animal",
    "object--pothole": "Pothole",
}


@dataclass(frozen=True)
class _ExtractPaths:
    repo: str
    zip_name: str
    cache: Path
    samples: Path
    index: Path
    target: dict[str, str]
    label_aliases: dict[str, str]
    max_per: int
    max_generic: int
    thumb: int


def _load_extract_paths(start: Path | None = None) -> _ExtractPaths:
    root = project_root(start)
    cfg = load_config(overrides=["dataset_name=mapillary_vistas"], start=root)
    data = cfg.data
    target = {str(k): str(v) for k, v in dict(data.get("extract_label_map") or {}).items()}
    readable = {k: str(_READABLE_DEFAULTS.get(k) or v.replace("_", " ").title()) for k, v in target.items()}
    aliases: dict[str, str] = {}
    for internal in target:
        aliases[internal] = internal
        aliases[internal.lower()] = internal
        aliases[readable[internal]] = internal
        aliases[readable[internal].lower()] = internal
    cache = Path(str(cfg.paths.dataset_cache_dir))
    return _ExtractPaths(
        repo=str(data.get("hf_repo") or "candylion/mapillary-vistas-v2"),
        zip_name=str(data.get("hf_zip_name") or "mapillary-vistas-dataset_public_v2.0.zip"),
        cache=cache,
        samples=Path(str(cfg.paths.samples_dir)),
        index=cache / "_zip_index.jsonl",
        target=target,
        label_aliases=aliases,
        max_per=int(data.get("extract_max_per_class") or 6),
        max_generic=int(data.get("extract_max_generic") or 10),
        thumb=int(data.get("extract_thumb_max_side") or 1280),
    )


HttpRange = Callable[[int, int | None], bytes]


def _find_all(hay: bytes, needle: bytes) -> list[int]:
    out: list[int] = []
    i = 0
    while True:
        j = hay.find(needle, i)
        if j < 0:
            break
        out.append(j)
        i = j + 1
    return out


def _zip_size(url: str, token: str) -> int:
    auth = {"Authorization": f"Bearer {token}", "User-Agent": "edgecase-synthesis"}
    req = urllib.request.Request(url, method="HEAD", headers=auth)
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers.get("Content-Length") or 0)
    if size:
        return size
    req = urllib.request.Request(url, headers={**auth, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range") or ""
    if "/" in cr:
        return int(cr.rsplit("/", 1)[-1])
    raise RuntimeError("Could not determine remote zip size (Content-Length / Content-Range).")


def _central_directory_location(tail: bytes, http_range: HttpRange) -> tuple[int, int]:
    """Resolve central-directory size and offset from ZIP metadata."""
    locs = _find_all(tail, b"PK\x06\x07")
    eocds = _find_all(tail, b"PK\x05\x06")
    if locs:
        locator = tail[locs[-1] :]
        z64_offset = struct.unpack_from("<Q", locator, 8)[0]
        z64 = http_range(z64_offset, z64_offset + 128)
        if z64[:4] != b"PK\x06\x06":
            raise RuntimeError("ZIP64 EOCD signature mismatch")
        return struct.unpack_from("<Q", z64, 40)[0], struct.unpack_from("<Q", z64, 48)[0]
    if eocds:
        eocd = tail[eocds[-1] :]
        return struct.unpack_from("<I", eocd, 12)[0], struct.unpack_from("<I", eocd, 16)[0]
    raise RuntimeError("Could not find EOCD / ZIP64 locator in zip tail")


def _expand_zip64_values(
    extra: bytes,
    comp_size: int,
    uncomp_size: int,
    local_offset: int,
) -> tuple[int, int, int]:
    """Expand sentinel sizes and offsets from a ZIP64 extra field."""
    position = 0
    while position + 4 <= len(extra):
        extra_id, extra_size = struct.unpack_from("<HH", extra, position)
        data = extra[position + 4 : position + 4 + extra_size]
        if extra_id == 1:
            offset = 0
            if uncomp_size == 0xFFFFFFFF:
                uncomp_size = struct.unpack_from("<Q", data, offset)[0]
                offset += 8
            if comp_size == 0xFFFFFFFF:
                comp_size = struct.unpack_from("<Q", data, offset)[0]
                offset += 8
            if local_offset == 0xFFFFFFFF:
                local_offset = struct.unpack_from("<Q", data, offset)[0]
        position += 4 + extra_size
    return comp_size, uncomp_size, local_offset


def _parse_central_directory(directory: bytes) -> list[dict[str, Any]]:
    """Parse central-directory records into the slim index schema."""
    entries: list[dict[str, Any]] = []
    position = 0
    while position + 46 <= len(directory):
        if directory[position : position + 4] != b"PK\x01\x02":
            next_position = directory.find(b"PK\x01\x02", position + 1)
            if next_position < 0:
                break
            position = next_position
            continue
        method = struct.unpack_from("<H", directory, position + 10)[0]
        comp_size = struct.unpack_from("<I", directory, position + 20)[0]
        uncomp_size = struct.unpack_from("<I", directory, position + 24)[0]
        name_len = struct.unpack_from("<H", directory, position + 28)[0]
        extra_len = struct.unpack_from("<H", directory, position + 30)[0]
        comment_len = struct.unpack_from("<H", directory, position + 32)[0]
        local_offset = struct.unpack_from("<I", directory, position + 42)[0]
        name = directory[position + 46 : position + 46 + name_len]
        extra = directory[position + 46 + name_len : position + 46 + name_len + extra_len]
        if comp_size == 0xFFFFFFFF or uncomp_size == 0xFFFFFFFF or local_offset == 0xFFFFFFFF:
            comp_size, uncomp_size, local_offset = _expand_zip64_values(extra, comp_size, uncomp_size, local_offset)
        try:
            decoded_name = name.decode("utf-8")
        except UnicodeDecodeError:
            decoded_name = name.decode("latin1")
        entries.append(
            {
                "name": decoded_name,
                "method": method,
                "comp": comp_size,
                "uncomp": uncomp_size,
                "local_off": local_offset,
            },
        )
        position += 46 + name_len + extra_len + comment_len
    return entries


def build_zip_index(
    http_range: HttpRange,
    *,
    url: str,
    token: str,
    dest: Path,
    cache_dir: Path,
) -> Path:
    """Parse ZIP64 EOCD + central directory via Range GETs; cache a slim jsonl index."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    size = _zip_size(url, token)
    tqdm.write(f"remote zip size={size / 1e9:.2f} GB — fetching EOCD tail…")

    tail_len = min(1024 * 1024, size)
    tail = http_range(size - tail_len, size - 1)

    cd_size, cd_offset = _central_directory_location(tail, http_range)
    tqdm.write(f"central directory offset={cd_offset} size={cd_size / 1e6:.1f} MB — downloading…")
    cd = http_range(cd_offset, cd_offset + cd_size - 1)
    tqdm.write(f"cd bytes={len(cd):,} — parsing…")

    entries = _parse_central_directory(cd)

    # Slim index: only paths the toy extractor needs (+ config for debugging).
    useful = [
        e
        for e in entries
        if e["name"].startswith("validation/images/")
        or e["name"].startswith("validation/v2.0/polygons/")
        or e["name"] == "validation/v2.0/panoptic/panoptic_2020.json"
        or e["name"].startswith("config")
    ]
    tqdm.write(f"entries={len(entries):,} useful={len(useful):,} → {dest}")
    with dest.open("w", encoding="utf-8") as fh:
        for e in useful:
            fh.write(json.dumps(e) + "\n")
    return dest


def _make_http_range(url: str, token: str, *, max_retries: int = 8) -> HttpRange:
    def http_range(start: int, end: int | None = None) -> bytes:
        range_val = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
        req = urllib.request.Request(
            url,
            headers={
                "Range": range_val,
                "Authorization": f"Bearer {token}",
                "User-Agent": "edgecase-synthesis",
            },
        )
        last_err: Exception | None = None
        for attempt in range(1, max(1, int(max_retries)) + 1):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return cast(bytes, resp.read())
            except HTTPError as exc:
                last_err = exc
                if exc.code not in _RETRYABLE_HTTP or attempt >= max_retries:
                    raise
                delay = min(90.0, (2 ** (attempt - 1)) + random.uniform(0, 1.5))
                tqdm.write(f"  HTTP {exc.code} on Range GET — retry {attempt}/{max_retries} in {delay:.1f}s…")
                time.sleep(delay)
            except (URLError, TimeoutError, ConnectionError) as exc:
                last_err = exc
                if attempt >= max_retries:
                    raise
                delay = min(90.0, (2 ** (attempt - 1)) + random.uniform(0, 1.5))
                tqdm.write(f"  network error ({exc}) — retry {attempt}/{max_retries} in {delay:.1f}s…")
                time.sleep(delay)
        raise RuntimeError(f"Range GET failed after retries: {last_err}")

    return http_range


def _extract_zip_entry(http_range: HttpRange, entry: dict[str, Any]) -> bytes:
    """Fetch and decompress one indexed ZIP entry."""
    local_offset = int(entry["local_off"])
    compressed_size = int(entry["comp"])
    method = int(entry["method"])
    header = http_range(local_offset, local_offset + 512)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {entry['name']}")
    name_len = struct.unpack_from("<H", header, 26)[0]
    extra_len = struct.unpack_from("<H", header, 28)[0]
    data_start = local_offset + 30 + name_len + extra_len
    data = http_range(data_start, data_start + compressed_size - 1)
    if method == 0:
        return data
    if method == 8:
        return zlib.decompress(data, -15)
    raise RuntimeError(f"unsupported zip method {method}")


def _load_mapillary_index(
    paths: _ExtractPaths,
    http_range: HttpRange,
    *,
    url: str,
    token: str,
) -> dict[str, dict[str, Any]]:
    """Build when needed and load useful Mapillary archive entries."""
    if not paths.index.exists():
        tqdm.write(f"missing {paths.index.name} — building via EOCD Range parse (one-time)…")
        build_zip_index(http_range, url=url, token=token, dest=paths.index, cache_dir=paths.cache)
    by_name: dict[str, dict[str, Any]] = {}
    with paths.index.open() as handle:
        for line in handle:
            entry = json.loads(line)
            name = entry["name"]
            if (
                name.startswith("validation/images/")
                or name.startswith("validation/v2.0/polygons/")
                or name == "validation/v2.0/panoptic/panoptic_2020.json"
            ):
                by_name[name] = entry
    return by_name


def _resolve_internal(label: str, aliases: dict[str, str]) -> str | None:
    """Resolve a readable or internal category label."""
    if label in aliases:
        return aliases[label]
    return aliases.get(label.lower())


def _internals_in_aliases(aliases: set[str], label_aliases: dict[str, str]) -> set[str]:
    """Resolve every known internal category in an alias set."""
    return {internal for alias in aliases if (internal := _resolve_internal(alias, label_aliases)) is not None}


def _mapillary_categories_by_image(
    panoptic: dict[str, Any],
    label_aliases: dict[str, str],
) -> dict[str, set[str]]:
    """Map image stems to target internal category names."""
    categories = panoptic.get("categories") or []
    annotations = panoptic.get("annotations") or []
    images_by_id = {image["id"]: image for image in (panoptic.get("images") or [])}
    id_to_aliases = {
        category.get("id"): {
            str(category[key]) for key in ("name", "title", "supercategory", "readable") if category.get(key)
        }
        for category in categories
    }
    per_image: dict[str, set[str]] = {}
    if annotations and "segments_info" in annotations[0]:
        for annotation in annotations:
            file_name = annotation.get("file_name") or images_by_id.get(annotation.get("image_id"), {}).get("file_name")
            found: set[str] = set()
            for segment in annotation.get("segments_info") or []:
                category_id = segment.get("category_id")
                found |= _internals_in_aliases(
                    id_to_aliases.get(category_id, {str(category_id)}),
                    label_aliases,
                )
            if file_name:
                per_image[Path(file_name).stem] = found
        return per_image
    for annotation in annotations:
        image = images_by_id[annotation["image_id"]]
        stem = Path(image["file_name"]).stem
        category_id = annotation["category_id"]
        found = _internals_in_aliases(id_to_aliases.get(category_id, {str(category_id)}), label_aliases)
        per_image.setdefault(stem, set()).update(found)
    return per_image


def _select_mapillary_stems(
    paths: _ExtractPaths,
    per_image: dict[str, set[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Select per-target and generic image stems within configured caps."""
    counts = Counter(target for categories in per_image.values() for target in paths.target if target in categories)
    tqdm.write("target counts in val: " + str({paths.target[k]: int(counts.get(k, 0)) for k in paths.target}))
    found: dict[str, list[str]] = {target: [] for target in paths.target}
    for stem, categories in per_image.items():
        for target in paths.target:
            if target in categories and len(found[target]) < paths.max_per:
                found[target].append(stem)
        if all(len(stems) >= paths.max_per for stems in found.values()):
            break
    picked = {stem for stems in found.values() for stem in stems}
    generic: list[str] = []
    for stem in per_image:
        if stem in picked:
            continue
        generic.append(stem)
        picked.add(stem)
        if len(generic) >= paths.max_generic:
            break
    return found, generic


def _bbox_from_mapillary_polygon(
    stem: str,
    internal_names: set[str],
    *,
    scale: float,
    paths: _ExtractPaths,
    by_name: dict[str, dict[str, Any]],
    http_range: HttpRange,
) -> list[dict[str, Any]]:
    """Extract scaled target boxes from one polygon annotation."""
    polygon_path = f"validation/v2.0/polygons/{stem}.json"
    if polygon_path not in by_name:
        return []
    payload = json.loads(_extract_zip_entry(http_range, by_name[polygon_path]))
    boxes: list[dict[str, Any]] = []
    for obj in payload.get("objects") or []:
        internal = _resolve_internal(obj.get("label") or "", paths.label_aliases)
        points = obj.get("polygon") or []
        if internal is None or internal not in internal_names or not points:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        boxes.append(
            {
                "label": paths.target[internal],
                "bbox_xyxy": [min(xs) * scale, min(ys) * scale, max(xs) * scale, max(ys) * scale],
            },
        )
    return boxes


def _load_existing_labels(labels_path: Path, *, clean: bool) -> dict[str, list[Any]]:
    """Load existing labels unless a clean extraction was requested."""
    if not labels_path.exists() or clean:
        return {}
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): list(v) if isinstance(v, list) else [] for k, v in payload.items()}


def _extract_mapillary_jobs(
    paths: _ExtractPaths,
    jobs: list[tuple[str, str, set[str] | None]],
    *,
    clean: bool,
    by_name: dict[str, dict[str, Any]],
    http_range: HttpRange,
) -> list[dict[str, Any]]:
    """Extract selected images and incrementally persist their labels."""
    labels_path = paths.samples / "labels.json"
    labels = _load_existing_labels(labels_path, clean=clean)
    metadata: list[dict[str, Any]] = []
    for stem, tag, wanted in tqdm(jobs, desc="Extracting samples", unit="img"):
        image_key = f"validation/images/{stem}.jpg"
        if image_key not in by_name:
            tqdm.write(f"missing {image_key}")
            continue
        output_name = f"{tag}_{stem}.jpg"
        output_path = paths.samples / output_name
        if output_path.exists() and not clean and output_name in labels:
            boxes = labels[output_name]
        else:
            image = Image.open(BytesIO(_extract_zip_entry(http_range, by_name[image_key]))).convert("RGB")
            original_width, _ = image.size
            image.thumbnail((paths.thumb, paths.thumb))
            scale = image.size[0] / original_width
            image.save(output_path, quality=90)
            boxes = (
                _bbox_from_mapillary_polygon(
                    stem,
                    wanted,
                    scale=scale,
                    paths=paths,
                    by_name=by_name,
                    http_range=http_range,
                )
                if wanted
                else []
            )
            labels[output_name] = boxes
            labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        metadata.append({"file": output_name, "stem": stem, "tag": tag, "n_boxes": len(boxes)})
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return metadata


def _write_mapillary_metadata(paths: _ExtractPaths, metadata: list[dict[str, Any]]) -> None:
    """Write attribution and extraction metadata files."""
    (paths.samples / "ATTRIBUTION.txt").write_text(
        "Mapillary Vistas Dataset v2.0 — CC BY-NC-SA.\n"
        "https://www.mapillary.com/dataset/vistas\n"
        f"HF: https://huggingface.co/datasets/{paths.repo}\n"
        "Toy subset for educational bootcamp use only.\n",
        encoding="utf-8",
    )
    (paths.samples / "source_meta.json").write_text(
        json.dumps(
            {
                "dataset": "mapillary_vistas_v2",
                "hf_id": paths.repo,
                "target_classes": list(paths.target.values()),
                "n_images": len(metadata),
                "rows": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.cache.mkdir(parents=True, exist_ok=True)
    (paths.cache / "toy_meta.json").write_text(
        json.dumps({"rows": metadata, "targets": paths.target}, indent=2),
        encoding="utf-8",
    )


def extract_mapillary_toy(
    *,
    start: Path | None = None,
    clean: bool = False,
    max_retries: int = 8,
) -> list[Path]:
    """Extract the Mapillary toy subset into ``paths.samples_dir`` (with tqdm)."""
    paths = _load_extract_paths(start)
    token = (
        get_token()
        or __import__("os").environ.get("HF_TOKEN")
        or __import__("os").environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "No Hugging Face token. Set HF_TOKEN in .env or run `huggingface-cli login` "
            "(and accept Mapillary Vistas terms once in the browser).",
        )

    url = hf_hub_url(paths.repo, paths.zip_name, repo_type="dataset")
    http_range = _make_http_range(url, token, max_retries=max_retries)
    tqdm.write("HF auth ok — preparing Mapillary toy extract (Range GETs, not full zip)")

    by_name = _load_mapillary_index(paths, http_range, url=url, token=token)

    pano_e = by_name["validation/v2.0/panoptic/panoptic_2020.json"]
    tqdm.write(f"fetching panoptic json ({pano_e['comp'] / 1e6:.1f} MB compressed)…")
    panoptic = json.loads(_extract_zip_entry(http_range, pano_e))
    per_image = _mapillary_categories_by_image(panoptic, paths.label_aliases)
    found, generic = _select_mapillary_stems(paths, per_image)

    paths.samples.mkdir(parents=True, exist_ok=True)
    if clean:
        removed = 0
        for old in list(paths.samples.glob("*.jpg")) + list(paths.samples.glob("*.png")):
            old.unlink()
            removed += 1
        tqdm.write(f"clean: removed {removed} existing sample images")

    jobs: list[tuple[str, str, set[str] | None]] = []
    for internal, slug in paths.target.items():
        for stem in found[internal]:
            jobs.append((stem, slug, {internal}))
    for stem in generic:
        jobs.append((stem, "scene", set(paths.target)))

    meta = _extract_mapillary_jobs(
        paths,
        jobs,
        clean=clean,
        by_name=by_name,
        http_range=http_range,
    )
    _write_mapillary_metadata(paths, meta)
    tqdm.write(f"DONE {len(meta)} images → {paths.samples}")
    return list_sample_images(paths.samples)


def ensure_mapillary_samples(
    start: Path | None = None,
    *,
    clean: bool = False,
    min_images: int = 1,
    max_retries: int = 8,
) -> list[Path]:
    """Return sample paths, extracting the toy subset when missing or ``clean``.

    Safe to call from Notebook 0. Does not download Klein / diffusion weights.
    """
    paths = _load_extract_paths(start)
    existing = list_sample_images(paths.samples)
    if not clean and len(existing) >= min_images:
        scenes = sum(1 for p in existing if p.stem.startswith("scene_"))
        tagged = len(existing) - scenes
        tqdm.write(f"Samples OK: {len(existing)} images ({scenes} scene_ seeds, {tagged} tagged) in {paths.samples}")
        return existing
    if existing and not clean:
        tqdm.write(f"Only {len(existing)} sample(s) found (need ≥{min_images}) — extracting…")
    elif clean:
        tqdm.write("Re-extracting Mapillary toy subset (clean=True)…")
    else:
        tqdm.write(f"No samples in {paths.samples} — extracting Mapillary toy subset…")
    return extract_mapillary_toy(start=start, clean=clean, max_retries=max_retries)


def main(argv: list[str] | None = None) -> None:
    """CLI entry (also used by ``scripts/extract_mapillary_toy.py``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing sample images before extracting.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Retries per Range GET on HTTP 429/5xx (default: 8).",
    )
    args = parser.parse_args(argv)
    extract_mapillary_toy(clean=bool(args.clean), max_retries=int(args.max_retries))


if __name__ == "__main__":
    main()
