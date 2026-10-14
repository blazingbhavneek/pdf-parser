"""LibreOffice fallback with the Word server's output layout.

Writes <output>/document.pdf, <output>/pages/page_NNNN.docx (one per laid-out page) and
<output>/pages.json. Needs a Python that has LibreOffice's `uno` module (Arch: /usr/bin/python3):
    /usr/bin/python3 docx_converter/libreoffice.py <input.docx> <output_folder>
"""

import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import uno
from com.sun.star.beans import PropertyValue
from com.sun.star.connection import NoConnectException
from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK


def prop(name, value):
    result = PropertyValue()
    result.Name, result.Value = name, value
    return result


HIDDEN = (prop("Hidden", True),)


def paragraphs(text):
    """Body paragraphs in document order, descending into table cells."""
    enumeration = text.createEnumeration()
    while enumeration.hasMoreElements():
        element = enumeration.nextElement()
        if element.supportsService("com.sun.star.text.TextTable"):
            for name in element.getCellNames():
                yield from paragraphs(element.getCellByName(name))
        else:
            yield element


def flatten(context, document):
    """Freeze tracked changes and list labels so each page keeps the text it shows."""
    helper = context.ServiceManager.createInstanceWithContext("com.sun.star.frame.DispatchHelper", context)
    for command in (".uno:AcceptAllTrackedChanges", ".uno:DeleteAllNotes"):
        helper.executeDispatch(document.CurrentController.Frame, command, "", 0, ())
    # Read every label first: un-numbering one paragraph renumbers the ones after it.
    numbered = [(paragraph, paragraph.ListLabelString) for paragraph in paragraphs(document.Text)
                if paragraph.supportsService("com.sun.star.text.Paragraph") and paragraph.ListLabelString]
    for paragraph, label in numbered:
        paragraph.NumberingStyleName = ""
        if not paragraph.ListLabelString:  # Outline numbering can survive; never write it twice.
            paragraph.getText().insertString(paragraph.getStart(), label + "\t", False)


def furniture(document, cursor):
    style = document.StyleFamilies.getByName("PageStyles").getByName(cursor.PageStyleName)
    return {part: getattr(style, part.title() + "Text").String if getattr(style, part.title() + "IsOn") else ""
            for part in ("header", "footer")}


def column(letters):
    """Writer column order: A..Z, a..z, then longer names."""
    return len(letters), letters


def pieces(document, cursor, starts):
    """Per page, the selectable pieces shown on it: whole or partial paragraphs and table row runs.

    A selection can never cross into or out of a table, so pages are rebuilt from pieces that
    each stay inside one text: a paragraph split by a page break becomes two partial pieces.
    """
    def page(position):
        cursor.gotoRange(position, False)
        return cursor.getPage()

    result = [[] for _ in starts]
    enumeration = document.Text.createEnumeration()
    while enumeration.hasMoreElements():
        element = enumeration.nextElement()
        if element.supportsService("com.sun.star.text.TextTable"):
            # Positions and column counts break on merged cells; cell names ("A1", "C2") do not.
            last_cell = {}
            for name in element.getCellNames():
                letters, row = re.fullmatch(r"([A-Za-z]+)(\d+)", name).groups()
                last_cell[int(row)] = max(last_cell.get(int(row), letters), letters, key=column)
            row_pages = {row: page(element.getCellByName(f"A{row}").getStart()) for row in last_cell}
            for number, run in itertools.groupby(sorted(last_cell), key=row_pages.get):
                run = list(run)
                right = max((last_cell[row] for row in run), key=column)
                # A non-rectangular range (merged cells) comes back as None, not an error.
                block = element.getCellRangeByName(f"A{run[0]}:{right}{run[-1]}")
                if block is not None:
                    result[number - 1].append((block, False))
                    continue
                # ponytail: then one table per row keeps every cell's text but not the merges;
                # fine for a fallback, Word handles it.
                for row in run:
                    if last_cell[row] == "A":  # A one-cell range copies nothing; take its text.
                        cell = element.getCellByName(f"A{row}")
                        text = cell.createTextCursorByRange(cell.getStart())
                        text.gotoRange(cell.getEnd(), True)
                        result[number - 1].append((text, True))
                    else:
                        result[number - 1].append(
                            (element.getCellRangeByName(f"A{row}:{last_cell[row]}{row}"), False))
            continue
        first, last = page(element.getStart()), page(element.getEnd())
        for number in range(first, last + 1):
            if first == last:
                result[number - 1].append((element, True))
                continue
            text = element.getText()
            piece = text.createTextCursorByRange(element.getStart() if number == first else starts[number - 1])
            piece.gotoRange(element.getEnd() if number == last else starts[number], True)
            result[number - 1].append((piece, number == last))
    return result


def split(context, desktop, document, output):
    flatten(context, document)
    document.storeToURL((output / "document.pdf").as_uri(), (prop("FilterName", "writer_pdf_Export"),))
    controller = document.CurrentController
    cursor = controller.ViewCursor
    cursor.jumpToLastPage()
    starts, rows = [], []
    for number in range(1, cursor.getPage() + 1):
        cursor.jumpToPage(number)
        cursor.jumpToStartOfPage()
        starts.append(cursor.getStart())
        rows.append({"page": number, **furniture(document, cursor)})
    for number, selections in enumerate(pieces(document, cursor, starts), 1):
        page = desktop.loadComponentFromURL("private:factory/swriter", "_blank", 0, HIDDEN)
        try:
            for selection, paragraph_end in selections:
                controller.select(selection)
                page.CurrentController.select(page.Text.getEnd())
                page.CurrentController.insertTransferable(controller.getTransferable())
                if paragraph_end:
                    page.Text.insertControlCharacter(page.Text.getEnd(), PARAGRAPH_BREAK, False)
            page.storeToURL((output / "pages" / f"page_{number:04d}.docx").as_uri(),
                            (prop("FilterName", "MS Word 2007 XML"),))
        finally:
            page.close(True)
    return rows


def convert(source, output):
    (output / "pages").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="docx-converter-") as temporary:
        # A private copy keeps LibreOffice's lock file out of the caller's folder.
        copy = Path(temporary) / ("input" + source.suffix.lower())
        copy.write_bytes(source.read_bytes())
        pipe = f"docx_converter_{os.getpid()}"
        office = subprocess.Popen(
            [os.environ.get("LIBREOFFICE_BIN") or "soffice", "--headless", "--invisible", "--nologo",
             "--norestore", f"-env:UserInstallation={(Path(temporary) / 'profile').as_uri()}",
             f"--accept=pipe,name={pipe};urp;"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            local = uno.getComponentContext()
            resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
            for _ in range(300):
                try:
                    context = resolver.resolve(f"uno:pipe,name={pipe};urp;StarOffice.ComponentContext")
                    break
                except NoConnectException:
                    time.sleep(0.1)
            else:
                raise RuntimeError("LibreOffice did not start within 30 seconds")
            desktop = context.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", context)
            document = desktop.loadComponentFromURL(copy.as_uri(), "_blank", 0, HIDDEN)
            if document is None:
                raise RuntimeError(f"LibreOffice could not open {source}")
            try:
                rows = split(context, desktop, document, output)
            finally:
                document.close(True)
        finally:
            office.terminate()
            try:
                office.wait(timeout=30)
            except subprocess.TimeoutExpired:
                office.kill()
    (output / "pages.json").write_text(json.dumps({"converter": "libreoffice", "pages": rows},
                                                  ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    convert(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
