"""Exact scores against gold Markdown (Word pages), separate from the LLM-judge `--score`.

    benchmark/.venv/bin/python benchmark/gold.py [--pipelines ours paddle] [--variants chars_rem]

Reads gold/<doc>/page_NNNN.md (+ pages.json) and output/<pipeline>/<doc>/page_NNNN_<type>_attemptN.md,
writes score/gold.json (every page plus the summary) and prints one row per pipeline and variant.

Text (the whole page, tables row by row, so text placed in the wrong kind of block still counts):
only kana, kanji and Latin letters count; digits, symbols and spaces are ignored.
  cer       character edits / gold characters (micro-averaged over pages)
  lcs       longest common subsequence / gold characters: how much text arrived in order
  seq       share of gold runs (2+ letters between symbols or digits) found verbatim
  leak/10k  predicted kanji outside the Japanese set and absent from gold, per 10,000 predicted letters
Tables (pipe or HTML; the k-th predicted table against the k-th gold table, digits count):
  teds, teds_s (structure only), cell_cer (cell texts aligned in reading order)
"strict" keeps only pages whose gold is complete: PDF agreement >= 0.98 and no images.
Page headers/footers recorded by the Word converter are removed from predictions before scoring.
"""

import argparse
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import unicodedata

from apted import APTED, Config
from markdown_it import MarkdownIt
from rapidfuzz.distance import LCSseq, Levenshtein

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "src")]

from benchmark.main import VARIANTS, markdown_type, now, read_json, variant_of, write_json  # noqa: E402
from pdf_parser.charset import is_kana, is_kanji, japanese  # noqa: E402
from pdf_parser.tables import parse_html  # noqa: E402

STRICT_AGREEMENT = 0.98
MARKDOWN = MarkdownIt("commonmark", {"html": True}).enable("table")


def normalize(text):
    """NFC, plus full-/half-width folding (Ａ→A, ｶ→カ) only inside the width-forms block."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[＀-￯]+", lambda m: unicodedata.normalize("NFKC", m.group()), text)


def is_letter(c):
    return is_kana(c) or is_kanji(c) or (c.isascii() and c.isalpha())


def letters(text):
    return "".join(c for c in normalize(text) if is_letter(c))


def runs(text):
    """Letter sequences between symbols, digits and spaces, two characters or longer."""
    return [run for run in "".join(c if is_letter(c) else " " for c in normalize(text)).split() if len(run) >= 2]


class _Split(HTMLParser):
    """Rendered Markdown → all text in reading order (tables row by row), table markup, image count."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text, self.tables, self.images, self.depth = [], [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.tables.append([])
        if tag == "img":
            self.images += 1
        if self.depth:
            self.tables[-1].append(self.get_starttag_text())
        if tag in ("p", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "pre"):
            self.text.append("\n")

    def handle_endtag(self, tag):
        if self.depth:
            self.tables[-1].append(f"</{tag}>")
        if tag == "table" and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        self.text.append(data)
        if self.depth:
            self.tables[-1].append(data)


def split(markdown):
    parser = _Split()
    parser.feed(MARKDOWN.render(markdown))
    parser.close()
    tables = [rows for markup in parser.tables for rows in parse_html("".join(markup))]
    return "".join(parser.text), tables, parser.images


class _Node:
    def __init__(self, tag, rowspan=1, colspan=1, content="", children=()):
        self.tag, self.rowspan, self.colspan, self.content, self.children = tag, rowspan, colspan, content, list(children)


class _TedsConfig(Config):
    valuecls = float

    def rename(self, a, b):
        if a.tag != b.tag or a.rowspan != b.rowspan or a.colspan != b.colspan:
            return 1.0
        if a.tag == "td" and (a.content or b.content):
            return Levenshtein.normalized_distance(a.content, b.content)
        return 0.0

    def children(self, node):
        return node.children


def cell_text(text):
    return "".join(normalize(text).split())


def _tree(rows, structure):
    return _Node("table", children=[_Node("tr", children=[
        _Node("td", rowspan, colspan, "" if structure else cell_text(text)) for text, rowspan, colspan, _ in row])
        for row in rows])


def _size(node):
    return 1 + sum(_size(child) for child in node.children)


def teds(gold, predicted, structure=False):
    a, b = _tree(predicted, structure), _tree(gold, structure)
    return 1 - APTED(a, b, _TedsConfig()).compute_edit_distance() / max(_size(a), _size(b))


def cell_edits(gold_rows, pred_rows):
    """Edit cost between cell texts aligned in row-major order (a missing row costs only its text)."""
    gold = [cell_text(text) for row in gold_rows for text, *_ in row]
    pred = [cell_text(text) for row in pred_rows for text, *_ in row]
    previous = [0]
    for b in pred:
        previous.append(previous[-1] + len(b))
    for a in gold:
        current = [previous[0] + len(a)]
        for j, b in enumerate(pred, 1):
            current.append(min(previous[j] + len(a), current[j - 1] + len(b),
                               previous[j - 1] + Levenshtein.distance(a, b)))
        previous = current
    return previous[-1], sum(len(a) for a in gold)


def score_page(gold_markdown, predicted_markdown, furniture, allowed):
    gold_text, gold_tables, gold_images = split(gold_markdown)
    pred_text, pred_tables, pred_images = split(predicted_markdown)
    gold, pred = letters(gold_text), letters(pred_text)
    for edge in furniture:  # Page header/footer lines are not gold content.
        edge = letters(edge)
        if edge and edge in pred and edge not in gold:
            pred = pred.replace(edge, "", 1)
    gold_runs = runs(gold_text)
    gold_chars = set(gold)
    record = {
        "gold_letters": len(gold), "pred_letters": len(pred),
        "edits": Levenshtein.distance(gold, pred), "lcs": LCSseq.similarity(gold, pred),
        "runs": len(gold_runs), "runs_found": sum(run in pred for run in gold_runs),
        "leaks": [c for c in pred if is_kanji(c) and c not in allowed and c not in gold_chars],
        "substitutions": [[gold[op.src_pos], pred[op.dest_pos]]
                          for op in Levenshtein.editops(gold, pred) if op.tag == "replace"],
        "gold_images": gold_images, "pred_images": pred_images,
        "missing_tables": max(0, len(gold_tables) - len(pred_tables)),
        "extra_tables": max(0, len(pred_tables) - len(gold_tables)), "tables": []}
    for k, gold_rows in enumerate(gold_tables):
        pred_rows = pred_tables[k] if k < len(pred_tables) else []
        edits, chars = cell_edits(gold_rows, pred_rows)
        record["tables"].append({
            "teds": teds(gold_rows, pred_rows) if pred_rows else 0.0,
            "teds_s": teds(gold_rows, pred_rows, True) if pred_rows else 0.0,
            "cell_edits": edits, "cell_chars": chars})
    return record


def summarize(records):
    def ratio(a, b):
        return round(a / b, 4) if b else None

    tables = [t for r in records for t in r["tables"]]
    substitutions = Counter(f"{g}→{p}" for r in records for g, p in r["substitutions"])
    return {
        "pages": len(records),
        "cer": ratio(sum(r["edits"] for r in records), sum(r["gold_letters"] for r in records)),
        "lcs": ratio(sum(r["lcs"] for r in records), sum(r["gold_letters"] for r in records)),
        "seq": ratio(sum(r["runs_found"] for r in records), sum(r["runs"] for r in records)),
        "leak_per_10k": ratio(10000 * sum(len(r["leaks"]) for r in records), sum(r["pred_letters"] for r in records)),
        "leaked": "".join(sorted({c for r in records for c in r["leaks"]})),
        "tables": len(tables),
        "teds": ratio(sum(t["teds"] for t in tables), len(tables)),
        "teds_s": ratio(sum(t["teds_s"] for t in tables), len(tables)),
        "cell_cer": ratio(sum(t["cell_edits"] for t in tables), sum(t["cell_chars"] for t in tables)),
        "missing_tables": sum(r["missing_tables"] for r in records),
        "extra_tables": sum(r["extra_tables"] for r in records),
        "top_substitutions": substitutions.most_common(15)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--io-dir", type=Path, default=HERE / "datagen" / "io")
    parser.add_argument("--pipelines", nargs="+")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--documents", nargs="+")
    args = parser.parse_args(argv)
    root = args.io_dir.expanduser().resolve()
    allowed = japanese()
    records = []
    for gold_dir in sorted(p for p in (root / "gold").glob("*") if p.is_dir()):
        document = gold_dir.name
        if args.documents and document not in args.documents:
            continue
        pages = {row["page"]: row for row in read_json(gold_dir / "pages.json", [])}
        for pipeline_dir in sorted(p for p in (root / "output").glob("*") if p.is_dir()):
            if args.pipelines and pipeline_dir.name not in args.pipelines:
                continue
            for predicted in sorted((pipeline_dir / document).glob("page_*_attempt*.md")):
                variant = variant_of(markdown_type(predicted.stem))
                if args.variants and variant not in args.variants:
                    continue
                number = int(predicted.stem.split("_")[1])
                gold_page = gold_dir / f"page_{number:04d}.md"
                if not gold_page.is_file():
                    continue
                info = pages.get(number, {})
                record = score_page(gold_page.read_text(encoding="utf-8"), predicted.read_text(encoding="utf-8"),
                                    [info.get("header") or "", info.get("footer") or ""], allowed)
                strict = info.get("agreement", 1.0) >= STRICT_AGREEMENT and not info.get("images")
                records.append({"pipeline": pipeline_dir.name, "document": document, "page": number,
                                "variant": variant, "file": predicted.name, "strict": strict, **record})
    groups = defaultdict(list)
    for record in records:
        groups[(record["pipeline"], record["variant"])].append(record)
    summary = [{"pipeline": pipeline, "variant": variant, "all": summarize(group),
                "strict": summarize([r for r in group if r["strict"]])}
               for (pipeline, variant), group in sorted(groups.items())]
    write_json(root / "score" / "gold.json", {"created_at": now(), "summary": summary, "records": records})
    columns = ("pages", "cer", "lcs", "seq", "leak_per_10k", "tables", "teds", "teds_s", "cell_cer")
    print(f"{'pipeline':<10} {'variant':<10} {'set':<6} " + " ".join(f"{c:>12}" for c in columns))
    for row in summary:
        for subset in ("strict", "all"):
            values = row[subset]
            print(f"{row['pipeline']:<10} {row['variant']:<10} {subset:<6} " +
                  " ".join(f"{'-' if values[c] is None else values[c]:>12}" for c in columns))
        if row["all"]["leaked"]:
            print(f"{'':<28}leaked: {row['all']['leaked'][:60]}")
    print(f"Wrote {root / 'score' / 'gold.json'} ({len(records)} pages)")


if __name__ == "__main__":
    main()
