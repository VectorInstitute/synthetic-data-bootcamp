#!/usr/bin/env python3
"""Extract a small Mapillary Vistas toy subset via authenticated zip Range GETs.

Does NOT download the full ~29GB archive. Pulls:
  - ZIP central-directory index via EOCD/ZIP64 Range GETs (cached under data/mapillary_vistas/)
  - validation panoptic JSON (~3MB compressed) to find rare-class frames
  - ~30 RGB images + matching polygon bboxes for bootcamp seeds

Targets: pothole, traffic_cone, ground_animal (+ generic street scenes).
Requires: `huggingface-cli login` and access to candylion/mapillary-vistas-v2.
"""
from __future__ import annotations

import json
import struct
import zlib
from collections import Counter
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import urllib.request
from huggingface_hub import get_token, hf_hub_url
from PIL import Image

REPO = "candylion/mapillary-vistas-v2"
FNAME = "mapillary-vistas-dataset_public_v2.0.zip"
ROOT = Path(__file__).resolve().parents[1] / "data" / "mapillary_vistas"
SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
INDEX = ROOT / "_zip_index.jsonl"

TARGET = {
    "object--pothole": "pothole",
    "object--traffic-cone": "traffic_cone",
    "animal--ground-animal": "ground_animal",
}
READABLE = {
    "object--pothole": "Pothole",
    "object--traffic-cone": "Traffic Cone",
    "animal--ground-animal": "Ground Animal",
}
MAX_PER = 6
MAX_GENERIC = 10
THUMB = 1280

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
    # Some CDNs omit Content-Length on HEAD; probe with a 1-byte range.
    req = urllib.request.Request(url, headers={**auth, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range") or ""
    if "/" in cr:
        return int(cr.rsplit("/", 1)[-1])
    raise RuntimeError("Could not determine remote zip size (Content-Length / Content-Range).")


def build_zip_index(http_range: HttpRange, *, url: str, token: str, dest: Path = INDEX) -> Path:
    """Parse ZIP64 EOCD + central directory via Range GETs; cache a slim jsonl index."""
    ROOT.mkdir(parents=True, exist_ok=True)
    size = _zip_size(url, token)
    print(f"remote zip size={size / 1e9:.2f} GB — fetching EOCD tail…", flush=True)

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

    print(
        f"central directory offset={cd_offset} size={cd_size / 1e6:.1f} MB — downloading…",
        flush=True,
    )
    cd = http_range(cd_offset, cd_offset + cd_size - 1)
    print(f"cd bytes={len(cd):,} — parsing…", flush=True)

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
    print(f"entries={len(entries):,} useful={len(useful):,} → {dest}", flush=True)
    with dest.open("w", encoding="utf-8") as fh:
        for e in useful:
            fh.write(json.dumps(e) + "\n")
    return dest


def main() -> None:
    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face token — run `huggingface-cli login` first.")
    url = hf_hub_url(REPO, FNAME, repo_type="dataset")
    print("auth ok", flush=True)

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
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read()

    def extract_entry(e: dict) -> bytes:
        local_off = int(e["local_off"])
        comp = int(e["comp"])
        method = int(e["method"])
        hdr = http_range(local_off, local_off + 512)
        if hdr[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {e['name']}")
        name_len = struct.unpack_from("<H", hdr, 26)[0]
        extra_len = struct.unpack_from("<H", hdr, 28)[0]
        data_start = local_off + 30 + name_len + extra_len
        data = http_range(data_start, data_start + comp - 1)
        if method == 0:
            return data
        if method == 8:
            return zlib.decompress(data, -15)
        raise RuntimeError(f"unsupported zip method {method}")

    if not INDEX.exists():
        print(f"missing {INDEX.name} — building via EOCD Range parse (one-time, ~tens of MB)…", flush=True)
        build_zip_index(http_range, url=url, token=token, dest=INDEX)

    print("loading index…", flush=True)
    by_name: dict[str, dict] = {}
    with INDEX.open() as f:
        for line in f:
            e = json.loads(line)
            n = e["name"]
            if (
                n.startswith("validation/images/")
                or n.startswith("validation/v2.0/polygons/")
                or n == "validation/v2.0/panoptic/panoptic_2020.json"
            ):
                by_name[n] = e
    print("index subset", len(by_name), flush=True)

    pano_e = by_name["validation/v2.0/panoptic/panoptic_2020.json"]
    print(f"fetching panoptic json ({pano_e['comp'] / 1e6:.1f} MB compressed)…", flush=True)
    pano = json.loads(extract_entry(pano_e))
    categories = pano.get("categories") or []
    annotations = pano.get("annotations") or []
    images = pano.get("images") or []
    print(
        f"n categories={len(categories)} n_ann={len(annotations)} n_images={len(images)}",
        flush=True,
    )
    id_to_name = {c["id"]: (c.get("name") or c.get("title")) for c in categories}
    img_by_id = {im["id"]: im for im in images}

    per_image: dict[str, set[str]] = {}
    if annotations and "segments_info" in annotations[0]:
        for ann in annotations:
            file_name = ann.get("file_name") or img_by_id.get(ann.get("image_id"), {}).get(
                "file_name"
            )
            cats: set[str] = set()
            for seg in ann.get("segments_info") or []:
                cats.add(id_to_name.get(seg.get("category_id"), str(seg.get("category_id"))))
            if file_name:
                per_image[Path(file_name).stem] = cats
    else:
        for ann in annotations:
            im = img_by_id[ann["image_id"]]
            stem = Path(im["file_name"]).stem
            per_image.setdefault(stem, set()).add(id_to_name.get(ann["category_id"], ""))

    cnt: Counter[str] = Counter()
    for cats in per_image.values():
        for t in TARGET:
            if t in cats:
                cnt[t] += 1
    print("target counts in val:", dict(cnt), flush=True)

    found: dict[str, list[str]] = {t: [] for t in TARGET}
    for stem, cats in per_image.items():
        for t in TARGET:
            if t in cats and len(found[t]) < MAX_PER:
                found[t].append(stem)
        if all(len(v) >= MAX_PER for v in found.values()):
            break
    print("picked:", {k: v for k, v in found.items()}, flush=True)

    picked: set[str] = set()
    for stems in found.values():
        picked.update(stems)
    generic: list[str] = []
    for stem in per_image:
        if stem in picked:
            continue
        generic.append(stem)
        picked.add(stem)
        if len(generic) >= MAX_GENERIC:
            break

    SAMPLES.mkdir(parents=True, exist_ok=True)
    for old in list(SAMPLES.glob("*.jpg")) + list(SAMPLES.glob("*.png")):
        old.unlink()

    def bbox_from_poly(stem: str, internal_names: set[str]) -> list[dict]:
        poly_path = f"validation/v2.0/polygons/{stem}.json"
        if poly_path not in by_name:
            return []
        d = json.loads(extract_entry(by_name[poly_path]))
        boxes: list[dict] = []
        rev = {v: k for k, v in READABLE.items()}
        for o in d.get("objects") or []:
            lab = o.get("label") or ""
            if lab in internal_names:
                internal = lab
            elif lab in rev and rev[lab] in internal_names:
                internal = rev[lab]
            else:
                continue
            pts = o.get("polygon") or []
            if not pts:
                continue
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            boxes.append(
                {
                    "label": TARGET[internal],
                    "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)],
                }
            )
        return boxes

    labels_out: dict[str, list] = {}
    meta: list[dict] = []

    def save(stem: str, tag: str, want_internal: set[str] | None) -> None:
        img_key = f"validation/images/{stem}.jpg"
        if img_key not in by_name:
            print("missing", img_key, flush=True)
            return
        print(f"extract {tag}_{stem}.jpg …", flush=True)
        raw = extract_entry(by_name[img_key])
        im = Image.open(BytesIO(raw)).convert("RGB")
        w0, _h0 = im.size
        im.thumbnail((THUMB, THUMB))
        scale = im.size[0] / w0
        out_name = f"{tag}_{stem}.jpg"
        im.save(SAMPLES / out_name, quality=90)
        boxes: list[dict] = []
        if want_internal:
            boxes = bbox_from_poly(stem, want_internal)
            for b in boxes:
                x1, y1, x2, y2 = b["bbox_xyxy"]
                b["bbox_xyxy"] = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        labels_out[out_name] = boxes
        meta.append({"file": out_name, "stem": stem, "tag": tag, "n_boxes": len(boxes)})
        print(f"  saved {out_name} {im.size} boxes={len(boxes)}", flush=True)

    for internal, slug in TARGET.items():
        for stem in found[internal]:
            save(stem, slug, {internal})
    for stem in generic:
        save(stem, "scene", set(TARGET))

    (SAMPLES / "labels.json").write_text(json.dumps(labels_out, indent=2))
    (SAMPLES / "ATTRIBUTION.txt").write_text(
        "Mapillary Vistas Dataset v2.0 — CC BY-NC-SA.\n"
        "https://www.mapillary.com/dataset/vistas\n"
        "HF: https://huggingface.co/datasets/candylion/mapillary-vistas-v2\n"
        "Toy subset for educational bootcamp use only.\n"
    )
    (SAMPLES / "source_meta.json").write_text(
        json.dumps(
            {
                "dataset": "mapillary_vistas_v2",
                "hf_id": REPO,
                "target_classes": list(TARGET.values()),
                "n_images": len(meta),
                "rows": meta,
            },
            indent=2,
        )
    )
    (ROOT / "toy_meta.json").write_text(
        json.dumps({"rows": meta, "targets": TARGET}, indent=2)
    )
    print("DONE", len(meta), "images ->", SAMPLES, flush=True)


if __name__ == "__main__":
    main()
