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
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import urllib.request
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
    readable = {
        k: str(_READABLE_DEFAULTS.get(k) or v.replace("_", " ").title())
        for k, v in target.items()
    }
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

    locs = _find_all(tail, b"PK\x06\x07")
    eocds = _find_all(tail, b"PK\x05\x06")
    if locs:
        loc = tail[locs[-1] :]
        z64_eocd_off = struct.unpack_from("<Q", loc, 8)[0]
        z64 = http_range(z64_eocd_off, z64_eocd_off + 128)
        if z64[:4] != b"PK\x06\x06":
            raise RuntimeError("ZIP64 EOCD signature mismatch")
        cd_size = struct.unpack_from("<Q", z64, 40)[0]
        cd_offset = struct.unpack_from("<Q", z64, 48)[0]
    elif eocds:
        eocd = tail[eocds[-1] :]
        cd_size = struct.unpack_from("<I", eocd, 12)[0]
        cd_offset = struct.unpack_from("<I", eocd, 16)[0]
    else:
        raise RuntimeError("Could not find EOCD / ZIP64 locator in zip tail")

    tqdm.write(f"central directory offset={cd_offset} size={cd_size / 1e6:.1f} MB — downloading…")
    cd = http_range(cd_offset, cd_offset + cd_size - 1)
    tqdm.write(f"cd bytes={len(cd):,} — parsing…")

    entries: list[dict] = []
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            nxt = cd.find(b"PK\x01\x02", pos + 1)
            if nxt < 0:
                break
            pos = nxt
            continue
        method = struct.unpack_from("<H", cd, pos + 10)[0]
        comp_size = struct.unpack_from("<I", cd, pos + 20)[0]
        uncomp_size = struct.unpack_from("<I", cd, pos + 24)[0]
        name_len = struct.unpack_from("<H", cd, pos + 28)[0]
        extra_len = struct.unpack_from("<H", cd, pos + 30)[0]
        comment_len = struct.unpack_from("<H", cd, pos + 32)[0]
        local_off = struct.unpack_from("<I", cd, pos + 42)[0]
        name = cd[pos + 46 : pos + 46 + name_len]
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        if comp_size == 0xFFFFFFFF or uncomp_size == 0xFFFFFFFF or local_off == 0xFFFFFFFF:
            epos = 0
            while epos + 4 <= len(extra):
                eid, esz = struct.unpack_from("<HH", extra, epos)
                edata = extra[epos + 4 : epos + 4 + esz]
                if eid == 1:
                    off = 0
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size = struct.unpack_from("<Q", edata, off)[0]
                        off += 8
                    if comp_size == 0xFFFFFFFF:
                        comp_size = struct.unpack_from("<Q", edata, off)[0]
                        off += 8
                    if local_off == 0xFFFFFFFF:
                        local_off = struct.unpack_from("<Q", edata, off)[0]
                epos += 4 + esz
        try:
            name_s = name.decode("utf-8")
        except UnicodeDecodeError:
            name_s = name.decode("latin1")
        entries.append(
            {
                "name": name_s,
                "method": method,
                "comp": comp_size,
                "uncomp": uncomp_size,
                "local_off": local_off,
            }
        )
        pos += 46 + name_len + extra_len + comment_len

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
        if end is None:
            range_val = f"bytes={start}-"
        else:
            range_val = f"bytes={start}-{end}"
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
                    return resp.read()
            except HTTPError as exc:
                last_err = exc
                if exc.code not in _RETRYABLE_HTTP or attempt >= max_retries:
                    raise
                delay = min(90.0, (2 ** (attempt - 1)) + random.uniform(0, 1.5))
                tqdm.write(
                    f"  HTTP {exc.code} on Range GET — retry {attempt}/{max_retries} "
                    f"in {delay:.1f}s…"
                )
                time.sleep(delay)
            except (URLError, TimeoutError, ConnectionError) as exc:
                last_err = exc
                if attempt >= max_retries:
                    raise
                delay = min(90.0, (2 ** (attempt - 1)) + random.uniform(0, 1.5))
                tqdm.write(
                    f"  network error ({exc}) — retry {attempt}/{max_retries} "
                    f"in {delay:.1f}s…"
                )
                time.sleep(delay)
        raise RuntimeError(f"Range GET failed after retries: {last_err}")

    return http_range


def extract_mapillary_toy(
    *,
    start: Path | None = None,
    clean: bool = False,
    max_retries: int = 8,
) -> list[Path]:
    """Extract the Mapillary toy subset into ``paths.samples_dir`` (with tqdm)."""
    paths = _load_extract_paths(start)
    token = get_token() or __import__("os").environ.get("HF_TOKEN") or __import__(
        "os"
    ).environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "No Hugging Face token. Set HF_TOKEN in .env or run `huggingface-cli login` "
            "(and accept Mapillary Vistas terms once in the browser)."
        )

    url = hf_hub_url(paths.repo, paths.zip_name, repo_type="dataset")
    http_range = _make_http_range(url, token, max_retries=max_retries)
    tqdm.write("HF auth ok — preparing Mapillary toy extract (Range GETs, not full zip)")

    def extract_entry(entry: dict) -> bytes:
        local_off = int(entry["local_off"])
        comp = int(entry["comp"])
        method = int(entry["method"])
        hdr = http_range(local_off, local_off + 512)
        if hdr[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {entry['name']}")
        name_len = struct.unpack_from("<H", hdr, 26)[0]
        extra_len = struct.unpack_from("<H", hdr, 28)[0]
        data_start = local_off + 30 + name_len + extra_len
        data = http_range(data_start, data_start + comp - 1)
        if method == 0:
            return data
        if method == 8:
            return zlib.decompress(data, -15)
        raise RuntimeError(f"unsupported zip method {method}")

    if not paths.index.exists():
        tqdm.write(
            f"missing {paths.index.name} — building via EOCD Range parse (one-time)…"
        )
        build_zip_index(
            http_range,
            url=url,
            token=token,
            dest=paths.index,
            cache_dir=paths.cache,
        )

    by_name: dict[str, dict] = {}
    with paths.index.open() as fh:
        for line in fh:
            entry = json.loads(line)
            name = entry["name"]
            if (
                name.startswith("validation/images/")
                or name.startswith("validation/v2.0/polygons/")
                or name == "validation/v2.0/panoptic/panoptic_2020.json"
            ):
                by_name[name] = entry

    pano_e = by_name["validation/v2.0/panoptic/panoptic_2020.json"]
    tqdm.write(f"fetching panoptic json ({pano_e['comp'] / 1e6:.1f} MB compressed)…")
    pano = json.loads(extract_entry(pano_e))
    categories = pano.get("categories") or []
    annotations = pano.get("annotations") or []
    images = pano.get("images") or []

    id_to_aliases: dict[Any, set[str]] = {}
    for cat in categories:
        cid = cat.get("id")
        aliases: set[str] = set()
        for key in ("name", "title", "supercategory", "readable"):
            val = cat.get(key)
            if val:
                aliases.add(str(val))
        id_to_aliases[cid] = aliases

    def resolve_internal(label: str) -> str | None:
        if label in paths.label_aliases:
            return paths.label_aliases[label]
        return paths.label_aliases.get(label.lower())

    def internals_in_aliases(aliases: set[str]) -> set[str]:
        found_keys: set[str] = set()
        for alias in aliases:
            internal = resolve_internal(alias)
            if internal is not None:
                found_keys.add(internal)
        return found_keys

    img_by_id = {im["id"]: im for im in images}
    per_image: dict[str, set[str]] = {}
    if annotations and "segments_info" in annotations[0]:
        for ann in annotations:
            file_name = ann.get("file_name") or img_by_id.get(ann.get("image_id"), {}).get(
                "file_name"
            )
            cats: set[str] = set()
            for seg in ann.get("segments_info") or []:
                cats |= internals_in_aliases(
                    id_to_aliases.get(seg.get("category_id"), {str(seg.get("category_id"))})
                )
            if file_name:
                per_image[Path(file_name).stem] = cats
    else:
        for ann in annotations:
            im = img_by_id[ann["image_id"]]
            stem = Path(im["file_name"]).stem
            cats = internals_in_aliases(
                id_to_aliases.get(ann["category_id"], {str(ann["category_id"])})
            )
            per_image.setdefault(stem, set()).update(cats)

    cnt: Counter[str] = Counter()
    for cats in per_image.values():
        for target in paths.target:
            if target in cats:
                cnt[target] += 1
    tqdm.write(
        "target counts in val: "
        + str({paths.target[k]: int(cnt.get(k, 0)) for k in paths.target})
    )

    found: dict[str, list[str]] = {t: [] for t in paths.target}
    for stem, cats in per_image.items():
        for target in paths.target:
            if target in cats and len(found[target]) < paths.max_per:
                found[target].append(stem)
        if all(len(v) >= paths.max_per for v in found.values()):
            break

    picked: set[str] = set()
    for stems in found.values():
        picked.update(stems)
    generic: list[str] = []
    for stem in per_image:
        if stem in picked:
            continue
        generic.append(stem)
        picked.add(stem)
        if len(generic) >= paths.max_generic:
            break

    paths.samples.mkdir(parents=True, exist_ok=True)
    if clean:
        removed = 0
        for old in list(paths.samples.glob("*.jpg")) + list(paths.samples.glob("*.png")):
            old.unlink()
            removed += 1
        tqdm.write(f"clean: removed {removed} existing sample images")

    def bbox_from_poly(stem: str, internal_names: set[str], *, scale: float = 1.0) -> list[dict]:
        poly_path = f"validation/v2.0/polygons/{stem}.json"
        if poly_path not in by_name:
            return []
        payload = json.loads(extract_entry(by_name[poly_path]))
        boxes: list[dict] = []
        for obj in payload.get("objects") or []:
            lab = obj.get("label") or ""
            internal = resolve_internal(lab)
            if internal is None or internal not in internal_names:
                continue
            pts = obj.get("polygon") or []
            if not pts:
                continue
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            boxes.append(
                {
                    "label": paths.target[internal],
                    "bbox_xyxy": [
                        min(xs) * scale,
                        min(ys) * scale,
                        max(xs) * scale,
                        max(ys) * scale,
                    ],
                }
            )
        return boxes

    labels_path = paths.samples / "labels.json"
    labels_out: dict[str, list] = {}
    if labels_path.exists() and not clean:
        try:
            labels_out = json.loads(labels_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            labels_out = {}
    meta: list[dict] = []

    def flush_labels() -> None:
        labels_path.write_text(json.dumps(labels_out, indent=2), encoding="utf-8")

    jobs: list[tuple[str, str, set[str] | None]] = []
    for internal, slug in paths.target.items():
        for stem in found[internal]:
            jobs.append((stem, slug, {internal}))
    for stem in generic:
        jobs.append((stem, "scene", set(paths.target)))

    for stem, tag, want_internal in tqdm(jobs, desc="Extracting samples", unit="img"):
        img_key = f"validation/images/{stem}.jpg"
        if img_key not in by_name:
            tqdm.write(f"missing {img_key}")
            continue
        out_name = f"{tag}_{stem}.jpg"
        out_path = paths.samples / out_name
        if out_path.exists() and not clean and out_name in labels_out:
            boxes = labels_out[out_name]
            meta.append({"file": out_name, "stem": stem, "tag": tag, "n_boxes": len(boxes)})
            continue

        raw = extract_entry(by_name[img_key])
        im = Image.open(BytesIO(raw)).convert("RGB")
        w0, _h0 = im.size
        im.thumbnail((paths.thumb, paths.thumb))
        scale = im.size[0] / w0
        im.save(out_path, quality=90)
        boxes = bbox_from_poly(stem, want_internal, scale=scale) if want_internal else []
        labels_out[out_name] = boxes
        meta.append({"file": out_name, "stem": stem, "tag": tag, "n_boxes": len(boxes)})
        flush_labels()

    flush_labels()
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
                "n_images": len(meta),
                "rows": meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.cache.mkdir(parents=True, exist_ok=True)
    (paths.cache / "toy_meta.json").write_text(
        json.dumps({"rows": meta, "targets": paths.target}, indent=2),
        encoding="utf-8",
    )
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
        tqdm.write(
            f"Samples OK: {len(existing)} images "
            f"({scenes} scene_ seeds, {tagged} tagged) in {paths.samples}"
        )
        return existing
    if existing and not clean:
        tqdm.write(
            f"Only {len(existing)} sample(s) found (need ≥{min_images}) — extracting…"
        )
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
