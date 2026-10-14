#!/usr/bin/env python3
"""Download/reuse the four Japanese stress pages; never overwrite existing results."""

import argparse
import io
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pymupdf
from PIL import Image

from benchmark.main import atomic_write, digest, read_json, valid_pdf, write_json
from benchmark.datagen.render import image_pdf

HERE = Path(__file__).resolve().parent
DEFAULT_IO = HERE / "cache" / "japanese-stress"
SHARED_RAW = HERE.parent / "io" / "raw"
CACHE_NAMES = {"jcs2023_p031_table": "jcs2023_guideline.pdf",
               "nta2025_tax_form": "nta_corporate_tax.pdf"}


def download(url, destination, expected_hash):
    if not destination.exists():
        print(f"Download {destination.name}", flush=True)
        with urlopen(Request(url, headers={"User-Agent": "pdf-parser-benchmark/1"}), timeout=90) as response:
            data = response.read()
        import hashlib
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"Downloaded source changed: {url}")
        atomic_write(destination, data)
    if digest(destination) != expected_hash:
        raise ValueError(f"Cached source changed: {destination}; preserve it and use a fresh --io-dir")
    return destination


def setup(root):
    catalog = read_json(HERE / "japanese_stress_sources.json")
    for folder in ("raw", "input", "output", "sources", "ground_truth"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    base = ("https://huggingface.co/datasets/stockmark/OmniDocBench-JASyn/resolve/"
            + catalog["jasyn_revision_observed"] + "/")
    annotations = download(base + "OmniDocBench_JASyn.json", root / "sources/jasyn_annotations.json",
                           catalog["jasyn_annotation_sha256"])
    by_image = {entry["page_info"]["image_path"]: entry for entry in read_json(annotations)}
    for name, specification in catalog["documents"].items():
        image_name = specification.get("image")
        url = base + "images/" + image_name if image_name else specification["url"]
        cached = download(url, root / "sources" / (image_name or CACHE_NAMES[name]),
                          specification["source_sha256"])
        target = root / "raw" / f"{name}.pdf"
        if not target.exists():
            if image_name:
                with Image.open(cached) as image:
                    data = image_pdf(image.convert("RGB"), 595.2756, 841.8898)
            else:
                with pymupdf.open(cached) as source, pymupdf.open() as selected:
                    page = specification["pdf_page_1_based"] - 1
                    if name == "nta2025_tax_form":
                        selected.new_page(width=595.2756, height=841.8898).show_pdf_page(
                            pymupdf.Rect(0, 0, 595.2756, 841.8898), source, page)
                    else:
                        selected.insert_pdf(source, from_page=page, to_page=page)
                    data = selected.tobytes(garbage=4, deflate=True)
            valid_pdf(data)
            atomic_write(target, data)
            print(f"Created {target}", flush=True)
        else:
            valid_pdf(target.read_bytes())
            print(f"Keep existing {target}", flush=True)
        if image_name:
            ground_truth = root / specification["ground_truth"]
            if not ground_truth.exists():
                write_json(ground_truth, by_image[image_name])
    metadata = root / "sources.json"
    if not metadata.exists():
        write_json(metadata, catalog)
    print(f"Ready: {root}\nPrepare: benchmark/.venv/bin/python benchmark/main.py --prepare --io-dir {root}")
    return root


def publish(root, target_dir):
    """Copy the corpus PDFs into the shared raw corpus; existing files win."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((root / "raw").glob("*.pdf")):
        target = target_dir / source.name
        if not target.exists():
            atomic_write(target, source.read_bytes())
            print(f"Published {target}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io-dir", type=Path, default=DEFAULT_IO)
    parser.add_argument("--no-publish", action="store_true",
                        help=f"build the corpus without copying its PDFs into {SHARED_RAW}")
    args = parser.parse_args(argv)
    root = setup(args.io_dir.resolve())
    if not args.no_publish:
        publish(root, SHARED_RAW)


if __name__ == "__main__":
    main()
