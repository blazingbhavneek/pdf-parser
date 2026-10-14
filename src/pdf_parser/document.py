"""PDF pages to images and text-layer characters (runs in worker processes)."""

import pymupdf

from pdf_parser import lattice

GARBLED = 0.02  # More unmapped / private-use glyphs than this: the text layer is not trusted.


def page_count(path):
    with pymupdf.open(path) as document:
        return len(document)


def render(path, index, dpi):
    """PNG, pixel size, point size, text-layer characters and ruled tables of one page.

    Characters are (c, x0, y0, x1, y1, line_id, vertical) in page pixels, already rotated like the image.
    """
    with pymupdf.open(path) as document:
        page = document[index]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
        matrix = page.rotation_matrix * pymupdf.Matrix(dpi / 72, dpi / 72)
        chars = []
        for b, block in enumerate(page.get_text("rawdict")["blocks"]):
            for l, line in enumerate(block.get("lines", ())):
                dx, dy = line["dir"]
                vertical = line.get("wmode", 0) == 1 or abs(dy) > abs(dx)
                for span in line["spans"]:
                    for char in span["chars"]:
                        box = pymupdf.Rect(char["bbox"]) * matrix
                        chars.append((char["c"], box.x0, box.y0, box.x1, box.y1, (b, l), vertical))
        return (pixmap.tobytes("png"), (pixmap.width, pixmap.height), (page.rect.width, page.rect.height), chars,
                ruled_tables(page, matrix) if chars else [])


def _edges(values, tolerance=1.0):
    edges = []
    for value in sorted(values):
        if not edges or value - edges[-1] > tolerance:
            edges.append(value)
    return edges


def ruled_tables(page, matrix):
    """Tables drawn with ruling lines, read from the text layer: exact text and merged cells.

    Each is {"box": page pixels, "rows", "cols", "cells": [(row, col, rowspan, colspan, text)]}.
    """
    try:
        found = page.find_tables().tables
    except Exception:  # A drawing PyMuPDF cannot analyse: the table model reads the table instead.
        return []
    result = []
    for table in found:
        boxes = [(r, c, box) for r, row in enumerate(table.rows) for c, box in enumerate(row.cells) if box]
        texts = table.extract()
        xs = _edges([v for _, _, box in boxes for v in (box[0], box[2])])
        ys = _edges([v for _, _, box in boxes for v in (box[1], box[3])])
        nearest = lambda edges, v: min(range(len(edges)), key=lambda k: abs(edges[k] - v))
        cells = []
        for r, c, (x0, y0, x1, y1) in boxes:
            col, row = nearest(xs, x0), nearest(ys, y0)
            text = lattice.join_text((texts[r][c] or "").split("\n"))
            cells.append((row, col, max(1, nearest(ys, y1) - row), max(1, nearest(xs, x1) - col), text))
        if len(cells) > 1:  # A lone frame is not a table.
            box = pymupdf.Rect(table.bbox) * matrix
            result.append({"box": (box.x0, box.y0, box.x1, box.y1), "rows": len(ys) - 1, "cols": len(xs) - 1,
                           "cells": sorted(cells)})
    return result


def usable(chars):
    """A real text layer: enough characters and almost no unmapped or private-use glyphs."""
    visible = [c[0] for c in chars if not c[0].isspace()]
    if len(visible) < 10:
        return False
    bad = sum(c == "�" or 0xE000 <= ord(c) <= 0xF8FF or ord(c) < 0x20 for c in visible)
    return bad / len(visible) <= GARBLED
