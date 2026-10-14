#!/usr/bin/env python3
"""Download the pinned public Japanese PDFs into the shared io/raw corpus; never overwrite."""

import argparse
import hashlib
import ssl
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.main import atomic_write, read_json, valid_pdf

HERE = Path(__file__).resolve().parent
DEFAULT_IO = HERE.parent / "io"


def fetch(url):
    request = Request(url, headers={"User-Agent": "pdf-parser-benchmark/1"})
    try:
        with urlopen(request, timeout=180) as response:
            return response.read()
    except URLError:
        # shortcut: some proxies break archive.org's TLS chain; the pinned SHA-256 still proves integrity.
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with urlopen(request, timeout=180, context=context) as response:
            return response.read()


def setup(root):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, document in read_json(HERE / "japanese_public_sources.json")["documents"].items():
        target = raw / name
        if target.exists():
            valid_pdf(target.read_bytes())
            print(f"Keep existing {target}", flush=True)
            continue
        data = fetch(document["snapshot"])
        valid_pdf(data)
        if hashlib.sha256(data).hexdigest() != document["sha256"]:
            raise ValueError(f"Archived PDF no longer matches the pinned hash: {document['snapshot']}")
        atomic_write(target, data)
        print(f"Created {target}", flush=True)
    print(f"Ready: {raw}\nPrepare: benchmark/.venv/bin/python benchmark/main.py --prepare --io-dir {root}",
          flush=True)
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io-dir", type=Path, default=DEFAULT_IO)
    args = parser.parse_args(argv)
    setup(args.io_dir.resolve())


if __name__ == "__main__":
    main()
