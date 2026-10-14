"""DOCX -> document.pdf + one DOCX per page: the Word server first, local LibreOffice second.

Both write the same folder layout (see server.py). Used by the benchmark and the pipeline.
"""

import io
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
NAMES = re.compile(r"document\.pdf|pages\.json|pages/page_\d{4}\.docx")


def request_word(url, source, token=None, timeout=600):
    with urlopen(urljoin(url, "/health"), timeout=5):
        pass  # Fail fast when the Windows machine is off, before uploading anything.
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="input.docx"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n").encode() + \
        Path(source).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, body, headers), timeout=timeout) as response:
        return response.read()


def unpack(data, destination):
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        names = bundle.namelist()
        if "document.pdf" not in names or not all(NAMES.fullmatch(name) for name in names):
            raise ValueError(f"Unexpected converter archive entries: {names[:5]}")
        bundle.extractall(destination)


def convert(source, destination, url=None, token=None, timeout=600, python=None, office=None):
    """Write document.pdf, pages/page_NNNN.docx and pages.json into a new `destination`.

    Returns "word" when the server at `url` converted it, otherwise "libreoffice".
    `python` must have LibreOffice's `uno` module; `office` is the soffice binary.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    partial = destination.with_name(destination.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    if url:
        try:
            unpack(request_word(url, source, token, timeout), partial)
            partial.rename(destination)
            return "word"
        except Exception as error:
            shutil.rmtree(partial, ignore_errors=True)
            print(f"[convert] Word server unavailable or failed ({error}); using LibreOffice", flush=True)
    env = {**os.environ, "LIBREOFFICE_BIN": office} if office else None
    subprocess.run([python or os.environ.get("LIBREOFFICE_PYTHON") or "/usr/bin/python3",
                    str(HERE / "libreoffice.py"), str(source), str(partial)],
                   check=True, timeout=timeout, env=env)
    partial.rename(destination)
    return "libreoffice"
