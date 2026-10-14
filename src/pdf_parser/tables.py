"""Tables: OTSL (from PaddleOCR-VL) to cells, OCR reconciliation, HTML templates and parsing.

Standard library only, so the benchmark scorer can reuse `parse_html`.
"""

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import html
from html.parser import HTMLParser
import re

from pdf_parser import lattice

TAG = re.compile(r"<fcel>|<ecel>|<lcel>|<ucel>|<xcel>|<nl>")
_ATTRS = '( (rowspan|colspan)="[1-9][0-9]*")*'
# Between non-empty cells the filling model may end rows and add empty cells, nothing else.
_GAP = "([ \\n]|</tr><tr>|<td" + _ATTRS + "></td>)*"


@dataclass
class Cell:
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    chars: list = field(default_factory=list)  # lattice.Char


@dataclass
class Table:
    rows: int
    cols: int
    cells: list


def parse_otsl(content, chars=None):
    """Cells from OTSL text; `chars` (one lattice.Char per character of `content`) are sliced per cell.

    Rows are padded or trimmed to the width needing the fewest changes, never cutting a filled
    cell, and spans follow <lcel>/<ucel>/<xcel> exactly as PaddleX's converter does.
    """
    matches = list(TAG.finditer(content))
    rows, row = [], []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        if match.group() == "<nl>":
            if row:
                rows.append(row)
            row = []
        else:
            row.append((match.group(), match.end(), end))
    if row:
        rows.append(row)
    if not rows:
        return None
    least = max(max((k + 1 for k, (tag, _, _) in enumerate(r) if tag == "<fcel>"), default=0) for r in rows)
    widest = max(len(r) for r in rows)
    width = min(range(least, max(least, widest) + 1), key=lambda w: (sum(abs(len(r) - w) for r in rows), w))
    rows = [r[:width] + [("<ecel>", 0, 0)] * (width - len(r)) for r in rows]
    cells = []
    for r, row in enumerate(rows):
        for c, (tag, start, end) in enumerate(row):
            if tag not in ("<fcel>", "<ecel>"):
                continue
            colspan = 1
            while c + colspan < width and row[c + colspan][0] in ("<lcel>", "<xcel>"):
                colspan += 1
            rowspan = 1
            while r + rowspan < len(rows) and rows[r + rowspan][c][0] in ("<ucel>", "<xcel>"):
                rowspan += 1
            if tag == "<ecel>":
                start = end
            while start < end and content[start].isspace():
                start += 1
            while end > start and content[end - 1].isspace():
                end -= 1
            cell_chars = chars[start:end] if chars is not None else lattice.locked(content[start:end])
            cells.append(Cell(r, c, rowspan, colspan, list(cell_chars)))
    return Table(len(rows), width, cells)


def coverage(table, lines):
    """Share of the letters and digits OCR found in the table area that the table's cells contain."""
    found = Counter(c.c for line in lines for c in line if c.c.isalnum())
    held = Counter(c.c for cell in table.cells for c in cell.chars if c.c.isalnum())
    return sum((found & held).values()) / sum(found.values()) if found else 1.0


def from_lines(lines):
    """A plain grid (no spans) from OCR line positions: rows by vertical overlap, columns by left edge.

    The fallback when the table model drops content; every OCR line keeps its lattice.
    """
    rows = []
    for line in sorted(lines, key=lambda l: l["box"][1]):
        box = line["box"]
        if rows:
            first = rows[-1][0]["box"]
            if min(first[3], box[3]) - max(first[1], box[1]) > 0.5 * min(first[3] - first[1], box[3] - box[1]):
                rows[-1].append(line)
                continue
        rows.append([line])
    heights = sorted(l["box"][3] - l["box"][1] for l in lines)
    gap = heights[len(heights) // 2] if heights else 0
    starts = []
    for left in sorted(l["box"][0] for l in lines):
        if not starts or left - starts[-1] > gap:
            starts.append(left)
    cells = {}
    for r, row in enumerate(rows):
        for line in sorted(row, key=lambda l: l["box"][0]):
            c = max(k for k, start in enumerate(starts) if start <= line["box"][0] + gap / 2)
            cell = cells.setdefault((r, c), Cell(r, c))
            if cell.chars:
                cell.chars = cell.chars + [lattice.Char(" ", 1.0)]
            cell.chars = cell.chars + list(line["chars"])
    return Table(len(rows), len(starts), [cells.get((r, c), Cell(r, c)) for r in range(len(rows))
                                          for c in range(len(starts))])


def from_grid(rows, cols, cells):
    """A table from (row, col, rowspan, colspan, text) cells, e.g. a ruled table in the text layer."""
    return Table(rows, cols, [Cell(r, c, rs, cs, lattice.locked(text)) for r, c, rs, cs, text in sorted(cells)])


def _letters(chars):
    return Counter(c.c for c in chars if c.c.isalnum())


def reconcile(table, lines, trusted=False):
    """Confirm or contest each cell's characters with the lines ({"box", "chars"}) that hold them.

    Cells wrap over several lines and short texts (低, なし) repeat down a column, so a line lying
    within one cell goes to the first cell, in reading order, that still lacks its characters, and a
    cell's further lines must sit under its first one. A line running across several cells (OCR
    merging a borderless row) is shared: each cell takes its own span of it.
    `trusted` lines (a verified text layer) replace the table reader's characters outright: the
    table model then only decides where the exact text goes.
    """
    lines = sorted(lines, key=lambda l: (l["box"][1], l["box"][0]))
    bags = [_letters(line["chars"]) for line in lines]
    free = set(range(len(lines)))
    for cell in sorted(table.cells, key=lambda c: (c.row, c.col)):
        need = _letters(cell.chars)
        size = sum(need.values())
        parts, area = [], None
        for k, line in enumerate(lines):
            if not need:
                break
            x0, y0, x1, y1 = line["box"]
            if area and (y0 > area[3] + 2 * (y1 - y0) or min(x1, area[2]) <= max(x0, area[0])):
                continue  # Not under this cell's lines.
            if k in free and bags[k] and sum((bags[k] & need).values()) >= 0.6 * sum(bags[k].values()):
                free.discard(k)
                parts.append(line["chars"])
                need -= bags[k]
                area = line["box"] if area is None else (min(area[0], x0), area[1], max(area[2], x1), max(area[3], y1))
        if not parts and size:
            # No line lies within this cell (the table reader misread it, or a longer line runs across
            # several cells): the line holding most of the cell's letters.
            held = [sum((bag & need).values()) for bag in bags]
            k = max(range(len(lines)), key=held.__getitem__, default=None)
            if k is not None and held[k] >= 0.5 * size:
                text, cell_text = lattice.text(lines[k]["chars"]), lattice.text(cell.chars)
                blocks = [b for b in SequenceMatcher(None, text, cell_text, autojunk=False).get_matching_blocks()
                          if b.size]
                span = lines[k]["chars"][blocks[0].a:blocks[-1].a + blocks[-1].size]
                if trusted and k in free and lattice.agreement(text, cell_text) >= 0.5:
                    free.discard(k)
                    span = lines[k]["chars"]  # The line is this cell: keep what the table reader dropped.
                parts.append(span)
        if parts:
            joined = lattice.join_lines([list(part) for part in parts])
            cell.chars = joined if trusted else lattice.merge(cell.chars, joined)


def _open(cell):
    spans = (f' rowspan="{cell.rowspan}"' if cell.rowspan > 1 else "") + \
            (f' colspan="{cell.colspan}"' if cell.colspan > 1 else "")
    return f"<td{spans}>"


def render(table, contents):
    """HTML with one string per cell (already escaped)."""
    by_row = {}
    for cell, content in zip(table.cells, contents):
        by_row.setdefault(cell.row, []).append(_open(cell) + content + "</td>")
    return "<table>" + "".join("<tr>" + "".join(by_row.get(r, [])) + "</tr>" for r in range(table.rows)) + "</table>"


def escape(string):
    return html.escape(string, quote=False)


def template_and_regex(table, cell_segments):
    """Prompt HTML (slots shown as ⟦A|B⟧) and the constraint letting a model change structure only."""
    shown = render(table, [lattice.template(segs, escape) for segs in cell_segments])
    parts = ["<table><tr>"]
    for segs in cell_segments:
        if segs:
            parts += [_GAP, "<td" + _ATTRS + ">", lattice.regex(segs, escape), "</td>"]
    parts += [_GAP, "</tr></table>"]
    return shown, "".join(parts)


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.stack.append([])
        elif tag == "tr" and self.stack:
            self.stack[-1].append([])
        elif tag in ("td", "th") and self.stack:
            if not self.stack[-1]:
                self.stack[-1].append([])

            def span(name):
                value = attrs.get(name) or "1"
                return int(value) if value.isdigit() and int(value) > 0 else 1

            self.stack[-1][-1].append([[], span("rowspan"), span("colspan"), tag == "th"])

    def handle_endtag(self, tag):
        if tag == "table" and self.stack:
            self.tables.append([[("".join(text).strip(), rowspan, colspan, header)
                                 for text, rowspan, colspan, header in row] for row in self.stack.pop()])

    def handle_data(self, data):
        if self.stack and self.stack[-1] and self.stack[-1][-1]:
            self.stack[-1][-1][-1][0].append(data)


def parse_html(markup):
    """Every <table> in `markup` as rows of (text, rowspan, colspan, is_header)."""
    parser = _TableParser()
    parser.feed(markup)
    parser.close()
    return parser.tables


def to_html(rows):
    def cell(text, rowspan, colspan, header):
        tag = "th" if header else "td"
        spans = (f' rowspan="{rowspan}"' if rowspan > 1 else "") + (f' colspan="{colspan}"' if colspan > 1 else "")
        return f"<{tag}{spans}>{escape(text)}</{tag}>"
    return "<table>" + "".join("<tr>" + "".join(cell(*c) for c in row) + "</tr>" for row in rows) + "</table>"


def width(rows):
    return sum(colspan for _, _, colspan, _ in rows[0]) if rows else 0


def merge_split(first_html, second_html):
    """One table from a table that continues on the next page, or None if they do not fit.

    Same column count is required; a repeated header row at the top of the second part is dropped.
    """
    first, second = parse_html(first_html), parse_html(second_html)
    if len(first) != 1 or len(second) != 1 or not first[0] or not second[0]:
        return None
    first, second = first[0], second[0]
    if width(first) != width(second):
        return None
    if [c[0] for c in second[0]] == [c[0] for c in first[0]]:
        second = second[1:]
    return to_html(first + second)
