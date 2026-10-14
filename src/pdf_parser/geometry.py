"""Page geometry: block reading order, assigning lines/characters to blocks, line order, crops."""

from dataclasses import dataclass, field
import io
import math

from PIL import Image


@dataclass
class Block:
    label: str
    box: tuple  # x0, y0, x1, y1 in page pixels
    order: float = None  # layout model's reading order; None for tables, figures, captions, ...
    lines: list = field(default_factory=list)  # OCR lines: {"poly", "box", "chars"}
    chars: list = field(default_factory=list)  # text-layer characters: (c, x0, y0, x1, y1, line, vertical)
    ruled: dict = None  # the text layer's ruled table at this table block (document.ruled_tables)


def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def inside(point, box, pad=2.0):
    return box[0] - pad <= point[0] <= box[2] + pad and box[1] - pad <= point[1] <= box[3] + pad


def padded(box, pad, size):
    return max(0, box[0] - pad), max(0, box[1] - pad), min(size[0], box[2] + pad), min(size[1], box[3] + pad)


def bounds(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def reading_order(blocks, vertical=False):
    """The layout model's order where it gave one; every other block goes right after the last
    ordered block above it (overlapping columns), else after the last one higher on the page.
    On a `vertical` page (columns right to left) "above" is "to the right"."""
    ordered = [b for b in blocks if b.order is not None]
    # Flip a vertical page so the right edge reads as the top.
    flip = (lambda box: (box[1], -box[2], box[3], -box[0])) if vertical else (lambda box: box)

    def key(block):
        if block.order is not None:
            return block.order, flip(block.box)[1]
        box = flip(block.box)
        above = [o for o in ordered if (b := flip(o.box))[3] <= box[1] + 5 and min(b[2], box[2]) > max(b[0], box[0])]
        if not above:
            above = [o for o in ordered if center(flip(o.box))[1] < center(box)[1]]
        return max((o.order for o in above), default=-1) + 0.5, box[1]

    return sorted(blocks, key=key)


def assign(items, blocks, box_of, attribute):
    """Append each item to `attribute` of the smallest block containing its center; return leftovers."""
    by_size = sorted(blocks, key=lambda b: area(b.box))
    leftovers = []
    for item in items:
        point = center(box_of(item))
        owner = next((b for b in by_size if inside(point, b.box)), None)
        if owner is None:
            leftovers.append(item)
        else:
            getattr(owner, attribute).append(item)
    return leftovers


def attach_ruled(blocks, ruled):
    """Give each layout table its ruled text-layer table; blocks inside that grid (a header row the
    layout model split off, say) become part of the table. Returns the remaining blocks."""
    for found in ruled:
        box = found["box"]
        table = next((b for b in blocks if b.label == "table" and not b.ruled
                      and area(overlap(b.box, box)) >= 0.5 * max(area(b.box), area(box))), None)
        if table is None:
            continue  # The layout model sees no table here (a framed note, say): leave it.
        table.ruled, table.box = found, (min(table.box[0], box[0]), min(table.box[1], box[1]),
                                         max(table.box[2], box[2]), max(table.box[3], box[3]))
        for other in blocks:
            if other is not table and inside(center(other.box), box, 0):
                table.lines += other.lines
                other.label = "absorbed"
    return [b for b in blocks if b.label != "absorbed"]


def overlap(a, b):
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def cluster(lines):
    """Group OCR lines no layout block claimed into paragraph-like blocks."""
    groups = []
    for line in sorted(lines, key=lambda l: l["box"][1]):
        x0, y0, x1, y1 = line["box"]
        for group in groups:
            gx0, gy0, gx1, gy1 = group.box
            height = y1 - y0
            if min(x1, gx1) > max(x0, gx0) and 0 <= y0 - gy1 < 1.5 * height:
                group.box = (min(x0, gx0), gy0, max(x1, gx1), max(y1, gy1))
                group.lines.append(line)
                break
        else:
            groups.append(Block("unclaimed", (x0, y0, x1, y1), None, [line]))
    return groups


def is_vertical(block):
    if block.label == "vertical_text":
        return True
    tall = sum((b := l["box"])[3] - b[1] > 1.5 * (b[2] - b[0]) for l in block.lines)
    return bool(block.lines) and tall > len(block.lines) / 2


def skew(lines):
    """Median slope of wide OCR lines (radians, positive = falling to the right)."""
    angles = sorted(math.atan2(l["poly"][1][1] - l["poly"][0][1], l["poly"][1][0] - l["poly"][0][0])
                    for l in lines if (b := l["box"])[2] - b[0] > 2 * (b[3] - b[1]))
    return angles[len(angles) // 2] if angles else 0.0


def order_lines(lines, points_of, vertical, angle=0.0):
    """Reading order of lines: right-to-left columns when vertical, else rows top-down, left-right.

    Positions are measured in a frame rotated by -`angle`, so lines of a skewed scan stay apart.
    """
    cos, sin = math.cos(angle), math.sin(angle)

    def box(line):
        points = points_of(line)
        xs = [x * cos + y * sin for x, y in points]
        ys = [y * cos - x * sin for x, y in points]
        return min(xs), min(ys), max(xs), max(ys)

    boxes = {id(line): box(line) for line in lines}
    if vertical:
        return [line for tier in tiers(lines, lambda l: boxes[id(l)])
                for line in sorted(tier, key=lambda l: (-center(boxes[id(l)])[0], boxes[id(l)][1]))]
    rows = []
    for line in sorted(lines, key=lambda l: boxes[id(l)][1]):
        current = boxes[id(line)]
        if rows:
            first = boxes[id(rows[-1][0])]
            overlap = min(first[3], current[3]) - max(first[1], current[1])
            if overlap > 0.5 * min(first[3] - first[1], current[3] - current[1]):
                rows[-1].append(line)
                continue
        rows.append([line])
    return [line for row in rows for line in sorted(row, key=lambda l: boxes[id(l)][0])]


def tiers(lines, box_of):
    """Vertical lines grouped into tiers, top-down: lines that share no height are read separately,
    so a headline column above the body is not interleaved with the body's columns."""
    groups = []
    for line in sorted(lines, key=lambda l: box_of(l)[1]):
        y0, y1 = box_of(line)[1], box_of(line)[3]
        if groups and y0 < groups[-1][0]:
            groups[-1][0] = max(groups[-1][0], y1)
            groups[-1][1].append(line)
        else:
            groups.append([y1, [line]])
    return [group for _, group in groups]


def corners(box):
    x0, y0, x1, y1 = box
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def text_layer_boxes(chars, vertical):
    """Text-layer characters regrouped into lines in geometric order (never stream order): [(text, box)]."""
    lines = {}
    for char in chars:
        lines.setdefault(char[5], []).append(char)
    groups = [sorted(group, key=lambda c: (c[2], c[1]) if vertical else (c[1], c[2])) for group in lines.values()]
    box = lambda group: (min(c[1] for c in group), min(c[2] for c in group),
                         max(c[3] for c in group), max(c[4] for c in group))
    return [("".join(c[0] for c in group), box(group))
            for group in order_lines(groups, lambda group: corners(box(group)), vertical)]


def text_layer_lines(chars, vertical):
    return [text for text, _ in text_layer_boxes(chars, vertical)]


def crop_quad(image, poly):
    """Rectified crop of a 4-point text line (top-left, top-right, bottom-right, bottom-left)."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    width = max(1, round(max(math.dist((x0, y0), (x1, y1)), math.dist((x3, y3), (x2, y2)))))
    height = max(1, round(max(math.dist((x0, y0), (x3, y3)), math.dist((x1, y1), (x2, y2)))))
    return image.transform((width, height), Image.Transform.QUAD, (x0, y0, x3, y3, x2, y2, x1, y1),
                           Image.Resampling.BICUBIC)


def encode(image, kind="PNG", **options):
    stream = io.BytesIO()
    image.save(stream, format=kind, **options)
    return stream.getvalue()
