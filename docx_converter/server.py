"""Word-on-Windows DOCX converter service.

POST /convert with a DOCX (multipart field "file", or the raw bytes as the body) returns a zip:
    document.pdf            Word's own rendering of the whole document
    pages/page_NNNN.docx    the content of each page as Word laid it out (styling not kept)
    pages.json              {"converter": "word", "pages": [{"page", "header", "footer"}]}
GET /health returns "ok". Run on the Windows machine with Word and Japanese fonts:
    pip install pywin32
    python server.py --host 0.0.0.0 --port 8765 --token SECRET
Requests run one at a time in one hidden Word: Word is a single, stateful app.
"""

import argparse
from email.parser import BytesParser
from email.policy import HTTP
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

WD_PAGES, WD_GOTO_PAGE, WD_ABSOLUTE, WD_DOCX, WD_PDF, WD_NO_SAVE = 2, 1, 1, 16, 17, 0
WD_PAGE_NUMBER, WD_ADJUSTED_PAGE_NUMBER = 3, 1
WD_PRIMARY, WD_FIRST_PAGE, WD_EVEN_PAGES = 1, 2, 3
MACROS_DISABLED = 3  # msoAutomationSecurityForceDisable
LIMIT = 200 * 1024 * 1024

_word = None


def word():
    """One hidden Word, restarted if it died. Macros never run."""
    global _word
    if _word is not None:
        try:
            _word.Documents.Count
            return _word
        except Exception:
            _word = None
    import win32com.client

    _word = win32com.client.DispatchEx("Word.Application")
    _word.Visible = False
    _word.DisplayAlerts = 0
    _word.AutomationSecurity = MACROS_DISABLED
    return _word


def flatten(document):
    """Freeze what a page would show differently once it stands alone."""
    document.TrackRevisions = False
    document.AcceptAllRevisions()
    if document.Comments.Count:
        document.DeleteAllComments()
    document.ConvertNumbersToText()  # 第7条, （3）, ⑤ keep their real numbers instead of restarting at 1
    document.Fields.Unlink()  # TOC, dates, SEQ and links become the text Word shows


def furniture(document, start):
    """Header and footer text of the page that starts at `start` (text only, not shapes)."""
    position = document.Range(start, start)
    section = position.Sections(1)
    setup = section.PageSetup
    first = document.Range(section.Range.Start, section.Range.Start).Information(WD_PAGE_NUMBER)
    if setup.DifferentFirstPageHeaderFooter and position.Information(WD_PAGE_NUMBER) == first:
        kind = WD_FIRST_PAGE
    elif setup.OddAndEvenPagesHeaderFooter and position.Information(WD_ADJUSTED_PAGE_NUMBER) % 2 == 0:
        kind = WD_EVEN_PAGES
    else:
        kind = WD_PRIMARY
    return {"header": section.Headers(kind).Range.Text.strip(),
            "footer": section.Footers(kind).Range.Text.strip()}


def convert(source, output):
    app = word()
    # A dummy password makes protected files fail at once instead of opening a prompt.
    document = app.Documents.Open(str(source), ConfirmConversions=False, ReadOnly=True,
                                  AddToRecentFiles=False, PasswordDocument="docx-converter",
                                  Visible=False, OpenAndRepair=False, NoEncodingDialog=True)
    try:
        flatten(document)
        (output / "pages").mkdir(parents=True)
        document.ExportAsFixedFormat(str(output / "document.pdf"), WD_PDF)
        # Word lays pages out when it renders; the same layout gives each page's start.
        document.Repaginate()
        count = document.ComputeStatistics(WD_PAGES)
        starts = [document.GoTo(WD_GOTO_PAGE, WD_ABSOLUTE, number).Start for number in range(1, count + 1)]
        rows = []
        for number, (start, end) in enumerate(zip(starts, starts[1:] + [document.Content.End]), 1):
            page = app.Documents.Add(Visible=False)
            try:
                if end > start:
                    page.Range().FormattedText = document.Range(start, end).FormattedText
                page.SaveAs2(str(output / "pages" / f"page_{number:04d}.docx"), FileFormat=WD_DOCX)
            finally:
                page.Close(WD_NO_SAVE)
            rows.append({"page": number, **furniture(document, start)})
    finally:
        document.Close(WD_NO_SAVE)
    (output / "pages.json").write_text(json.dumps({"converter": "word", "pages": rows},
                                                  ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def upload(content_type, body):
    """The DOCX bytes from multipart field "file", or the raw body."""
    if not content_type.startswith("multipart/form-data"):
        return body
    message = BytesParser(policy=HTTP).parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") == "file":
            return part.get_payload(decode=True)
    raise ValueError('multipart field "file" is missing')


def archive(folder):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(folder).as_posix())
    return stream.getvalue()


class Handler(BaseHTTPRequestHandler):
    token = None
    convert = staticmethod(convert)

    def reply(self, status, body, kind="text/plain; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, b"ok") if self.path == "/health" else self.reply(404, b"not found")

    def do_POST(self):
        if self.path != "/convert":
            return self.reply(404, b"not found")
        if self.token and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {self.token}"):
            return self.reply(401, b"unauthorized")
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= LIMIT:
            return self.reply(413, b"body must be between 1 byte and 200 MB")
        body = self.rfile.read(length)
        try:
            with tempfile.TemporaryDirectory(prefix="docx-converter-", ignore_cleanup_errors=True) as folder:
                source = Path(folder) / "input.docx"
                source.write_bytes(upload(self.headers.get("Content-Type", ""), body))
                self.convert(source, Path(folder) / "output")
                payload = archive(Path(folder) / "output")
        except Exception as error:
            return self.reply(500, f"{type(error).__name__}: {error}".encode())
        self.reply(200, payload, "application/zip")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=os.environ.get("DOCX_CONVERTER_TOKEN"),
                        help="required as 'Authorization: Bearer <token>' (default: $DOCX_CONVERTER_TOKEN)")
    args = parser.parse_args()
    Handler.token = args.token
    if not args.token:
        print("Warning: no --token; anyone on the network can make this Word open files.", flush=True)
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Listening on http://{args.host}:{args.port}/convert", flush=True)
    try:
        server.serve_forever()
    finally:
        if _word is not None:
            _word.Quit()


if __name__ == "__main__":
    main()
