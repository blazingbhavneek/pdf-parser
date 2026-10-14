"""Japanese PDF -> Markdown (see pipeline-design.md).

Start the servers first:
  model server: PYTHONPATH=src ~/.venvs/paddlepaddle_venv/bin/python -m pdf_parser.model_server
  PaddleOCR-VL: vllm serve ~/.paddlex/official_models/PaddleOCR-VL-1.6 --served-model-name PaddleOCR-VL-1.6 \\
                    --port 8118 --max-logprobs 20 --gpu-memory-utilization 0.4
  optional fill-in / judge VLM: any OpenAI-compatible server with structured outputs (vLLM)
Then:   python -m pdf_parser report.pdf [more.pdf ...] -o out/
Writes  out/<name>.md (tables as HTML), out/<name>.json (what each block went through), out/images/.
"""

import argparse
import asyncio
import base64
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

import httpx
from PIL import Image

from pdf_parser import charset, document, geometry, lattice, tables, vlm
from pdf_parser.geometry import Block

TEXT = {"text", "unclaimed", "paragraph_title", "doc_title", "abstract", "content", "reference", "reference_content",
        "footnote", "vision_footnote", "figure_title", "aside_text", "vertical_text", "algorithm"}
FURNITURE = {"header", "footer", "number", "formula_number"}
PICTURES = {"image", "chart", "seal", "header_image", "footer_image"}
FORMULAS = {"display_formula", "inline_formula"}
TRUST_TEXT_LAYER = 0.9  # agreement with an OCR reading needed before the text layer is used as-is
LINES_PER_READ = 8  # OCR lines per VL request for tall blocks
REPORTED = ("source", "slots", "filled", "template", "answer", "error")
MAX_SKEW_DEGREES = 0.5  # straighten scans tilted more than this
MIN_TABLE_COVERAGE = 0.9  # share of OCR text a table reading must keep
TINY_POINTS = 7.0  # text lines thinner than this (line box, in points) that no layout block claims are
                   # lettering inside a figure the layout model missed: unreadable, so skipped


@dataclass
class Settings:
    model_server: str
    vl_url: str
    vl_model: str
    fill_url: str = None
    fill_model: str = None
    fill_key: str = None
    judge_url: str = None
    judge_model: str = None
    judge_key: str = None
    dpi: int = 200
    threshold: float = 0.95
    top_k: int = 4
    pages: int = 8
    vl_concurrency: int = 32
    fill_concurrency: int = 16
    workers: int = 4
    text_layer: bool = True
    furniture: bool = False


KATAKANA = re.compile("[ァ-ヺー]{4,}")


def katakana_fixes(texts):
    """{misread: word} for long katakana words that differ from a far more frequent word of the same
    document only by one voicing mark (ボ/ポ/ホ, ブ/プ/フ, ...): the classic OCR slip that both
    readers can make with confidence."""
    counts = Counter(m.group() for text in texts for m in KATAKANA.finditer(text))
    bare = lambda word: "".join(c for c in unicodedata.normalize("NFD", word) if c not in "\u3099\u309a")
    fixes = {}
    for word, n in counts.items():
        for other, m in counts.items():
            if m >= max(3, 3 * n) and len(other) == len(word) and bare(other) == bare(word) \
                    and sum(a != b for a, b in zip(word, other)) == 1:
                fixes[word] = other
    return fixes


def b64(data):
    return base64.b64encode(data).decode()


def markdown(label, body):
    if label == "doc_title":
        return "# " + body
    if label == "paragraph_title":
        return "## " + body
    if label == "algorithm":
        return "```\n" + body + "\n```"
    return body


class ModelServer:
    def __init__(self, url, timeout=600):
        self.http = httpx.AsyncClient(base_url=url, timeout=timeout, limits=httpx.Limits(max_connections=64))

    async def post(self, path, payload):
        response = await self.http.post(path, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"model server {path}: HTTP {response.status_code}: {response.text[:300]}")
        return response.json()


class Renderer:
    """Markdown -> page image for the judge, via the benchmark's Playwright renderer on one thread."""

    def __init__(self):
        self.pool = ThreadPoolExecutor(1)
        self.renderer = None

    def _render(self, markdown_text, assets, size_pt):
        if self.renderer is None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from benchmark.datagen.render import MarkdownRenderer

            self.renderer = MarkdownRenderer().__enter__()
        with tempfile.TemporaryDirectory(prefix="judge-") as folder:
            destination = Path(folder) / "page"
            self.renderer.render(markdown_text, assets, assets, size_pt[0], size_pt[1], 100, destination)
            return destination.with_suffix(".png").read_bytes()

    async def render(self, markdown_text, assets, size_pt):
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._render, markdown_text, assets, size_pt)

    def close(self):
        if self.renderer is not None:
            self.pool.submit(self.renderer.__exit__, None, None, None).result()
        self.pool.shutdown()


def small(png, longest=1600):
    image = Image.open(io.BytesIO(png))
    image.thumbnail((longest, longest))
    return geometry.encode(image)


class Parser:
    def __init__(self, settings):
        self.settings = settings
        self.models = ModelServer(settings.model_server)
        self.vl = vlm.Chat(settings.vl_url, settings.vl_model, concurrency=settings.vl_concurrency)
        # Slots are filled only by a capable instruction-following VLM; an OCR model chooses worse
        # than the readers' own best guess, which is the result when none is configured.
        self.filler = vlm.Chat(settings.fill_url, settings.fill_model, settings.fill_key, settings.fill_concurrency) \
            if settings.fill_url and settings.fill_model else None
        self.judge = vlm.Chat(settings.judge_url, settings.judge_model, settings.judge_key, 4) \
            if settings.judge_url and settings.judge_model else None
        self.allowed = charset.japanese()
        allowed = charset.regex_class(self.allowed)
        self.text_constraint = allowed + "*"
        self.table_constraint = "(" + "|".join(vlm.OTSL_TAGS) + "|" + allowed + ")*"
        # Spawned, not forked: the event loop and HTTP clients already run threads.
        self.pool = ProcessPoolExecutor(settings.workers, mp_context=multiprocessing.get_context("spawn"))
        self.pages = asyncio.Semaphore(settings.pages)
        self.renderer = Renderer() if self.judge else None

    async def close(self):
        for chat in {self.vl, self.filler, self.judge} - {None}:
            await chat.close()
        await self.models.http.aclose()
        self.pool.shutdown()
        if self.renderer:
            self.renderer.close()

    async def run(self, path, out_dir):
        """Parse one PDF into out_dir/<stem>.md and out_dir/<stem>.json."""
        out_dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(self.pool, document.page_count, str(path))
        pages = await asyncio.gather(*(self.page(path, index, out_dir) for index in range(count)))
        pieces = []
        for page in pages:
            for position, (label, body, exact) in enumerate(page["blocks"]):
                if label == "table" and position == 0 and pieces and pieces[-1][0] == "table":
                    merged = tables.merge_split(pieces[-1][1], body)
                    if merged:  # A table continuing from the previous page.
                        pieces[-1] = ("table", merged, exact and pieces[-1][2])
                        continue
                pieces.append((label, body, exact))
        fixes = katakana_fixes([body for _, body, _ in pieces])
        bodies = [body if exact else KATAKANA.sub(lambda m: fixes.get(m.group(), m.group()), body)
                  for _, body, exact in pieces]
        (out_dir / f"{path.stem}.md").write_text("\n\n".join(bodies) + "\n", encoding="utf-8")
        report = {"source": str(path), "katakana_fixes": fixes, "pages": [page["report"] for page in pages]}
        (out_dir / f"{path.stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")

    async def page(self, path, index, out_dir):
        async with self.pages:
            loop = asyncio.get_running_loop()
            png, _, size_pt, chars, ruled = await loop.run_in_executor(
                self.pool, document.render, str(path), index, self.settings.dpi)
            image = Image.open(io.BytesIO(png)).convert("RGB")
            use_layer = self.settings.text_layer and document.usable(chars)
            # Scans may be tilted; straighten them (never pages whose text layer must stay aligned).
            image, blocks, angle = await self.analyse(image, straighten=not use_layer)
            if use_layer:
                blocks = geometry.attach_ruled(blocks, ruled)
                geometry.assign(chars, blocks, lambda c: c[1:5], "chars")
            name = f"{path.stem}_p{index + 1:04d}"
            results, report = await self.blocks(image, blocks, use_layer, name, out_dir)
            verdict = None
            if self.judge and any(entry["source"].startswith("text_layer") for entry in report):
                rendered = await self.renderer.render("\n\n".join(body for _, body, _ in results), out_dir, size_pt)
                verdict = await vlm.judge(self.judge, small(png), small(rendered))
                if not verdict["faithful"]:  # The text layer misled us: read the page by OCR only.
                    results, report = await self.blocks(image, blocks, False, name, out_dir)
            return {"blocks": results, "report": {"page": index + 1, "text_layer": use_layer,
                                                  "deskew_degrees": round(angle, 2), "judge": verdict,
                                                  "blocks": report}}

    async def analyse(self, image, straighten=False):
        """(image, layout blocks in reading order holding their OCR line lattices, degrees straightened)."""
        jpeg = await asyncio.to_thread(geometry.encode, image, "JPEG", quality=95)
        layout, det = await asyncio.gather(self.models.post("/layout", {"image": b64(jpeg)}),
                                           self.models.post("/det", {"image": b64(jpeg)}))
        lines = [{"poly": poly, "box": geometry.bounds(poly)} for poly in det["polys"]]
        angle = math.degrees(geometry.skew(lines)) if straighten else 0.0
        if abs(angle) > MAX_SKEW_DEGREES:
            # Tilted boxes overlap and split lines between blocks: level the page and look again.
            image = await asyncio.to_thread(image.rotate, angle, Image.Resampling.BICUBIC, True, fillcolor="white")
            jpeg = await asyncio.to_thread(geometry.encode, image, "JPEG", quality=95)
            layout, det = await asyncio.gather(self.models.post("/layout", {"image": b64(jpeg)}),
                                               self.models.post("/det", {"image": b64(jpeg)}))
            lines = [{"poly": poly, "box": geometry.bounds(poly)} for poly in det["polys"]]
        else:
            angle = 0.0
        blocks = [Block(box["label"], tuple(box["box"]), box["order"]) for box in layout["boxes"]]
        # An inline formula inside a text block is read with that block.
        blocks = [b for b in blocks if not (b.label == "inline_formula" and any(
            o.label in TEXT and geometry.inside(geometry.center(b.box), o.box, 0) for o in blocks))]
        if lines:
            crops = await asyncio.to_thread(
                lambda: [b64(geometry.encode(geometry.crop_quad(image, line["poly"]))) for line in lines])
            recognized = await self.models.post("/rec", {"images": crops, "top_k": self.settings.top_k, "mask": True})
            for line, result in zip(lines, recognized["lines"]):
                line["chars"] = lattice.from_reader(result["chars"])
        blocks += geometry.cluster(geometry.assign(lines, blocks, lambda l: l["box"], "lines"))
        written = [b for b in blocks if b.label in TEXT and b.lines]
        vertical = sum(map(geometry.is_vertical, written)) > len(written) / 2
        return image, geometry.reading_order(blocks, vertical), angle

    async def blocks(self, image, blocks, use_layer, name, out_dir):
        """Markdown per content block: read everything first, then fill slots with neighbours as context."""
        content = [b for b in blocks if self.settings.furniture or b.label not in FURNITURE]
        readings = await asyncio.gather(*(self.read(image, block, use_layer, f"{name}_{k}", out_dir)
                                          for k, block in enumerate(content)))
        guesses = [r["final"] if r.get("final") is not None else lattice.best(r["segs"]) if "segs" in r else ""
                   for r in readings]

        def context(k):  # Neighbouring blocks' best guesses help the filler choose.
            before = guesses[k - 1][-150:] if k else ""
            after = guesses[k + 1][:150] if k + 1 < len(guesses) else ""
            return " … ".join(part for part in (before, after) if part)

        finished = await asyncio.gather(*(self.finish(reading, context(k)) for k, reading in enumerate(readings)))
        results, report = [], []
        for block, reading, body in zip(content, readings, finished):
            if body:
                results.append((block.label, markdown(block.label, body),
                                reading.get("source", "").startswith("text_layer")))
            report.append({"label": block.label, "box": [round(v) for v in block.box], "lines": len(block.lines),
                           **{k: v for k, v in reading.items() if k in REPORTED}})
        return results, report

    async def read(self, image, block, use_layer, name, out_dir):
        crop = await asyncio.to_thread(geometry.encode, image.crop(tuple(map(round, block.box))))
        if block.label in PICTURES:
            folder = out_dir / "images"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{name}.png").write_bytes(crop)
            return {"final": f"![{block.label}](images/{name}.png)", "source": "image"}
        if block.label in FORMULAS:
            try:
                latex = (await vlm.read(self.vl, crop, vlm.FORMULA, max_tokens=2048))[0].strip()
            except Exception as error:
                return {"final": "", "source": "formula", "error": str(error)}
            return {"final": f"$$\n{latex}\n$$" if block.label == "display_formula" else f"${latex}$",
                    "source": "formula"}
        if block.label == "table":
            return await self.read_table(image, block, crop, use_layer)
        return await self.read_text(image, block, crop, use_layer)

    async def read_text(self, image, block, crop, use_layer):
        code = block.label == "algorithm"
        vertical = geometry.is_vertical(block)
        lines = block.lines
        angle = geometry.skew(lines)
        groups = [geometry.order_lines(tier, lambda l: l["poly"], True, angle)
                  for tier in geometry.tiers(lines, lambda l: l["box"])] if vertical else \
            [geometry.order_lines(lines, lambda l: l["poly"], False, angle)]
        ordered = [line for group in groups for line in group]
        ocr = lattice.join_lines([line["chars"] for line in ordered], newline=code)
        layer = lattice.join_text(geometry.text_layer_lines(block.chars, vertical), code) \
            if use_layer and block.chars else None
        if layer and lattice.agreement(layer, lattice.text(ocr)) >= TRUST_TEXT_LAYER:
            return {"final": layer, "source": "text_layer"}
        thin = all(min(b[2] - b[0], b[3] - b[1]) * 72 / self.settings.dpi < TINY_POINTS
                   for b in (line["box"] for line in lines))
        if lines and (not ocr or block.label == "unclaimed" and thin) and not layer:
            # Lines were found but OCR could read none (lettering inside a chart, say): the VL reader
            # would only invent text here.
            return {"final": "", "source": "unreadable"}
        # Tall blocks are read a few lines at a time: one crop of a whole page-long paragraph is
        # downscaled and the VL reader skips repeated lines.
        chunks = [group[k:k + LINES_PER_READ] for group in groups for k in range(0, len(group), LINES_PER_READ)]
        if len(chunks) > 1:  # (each tier of a vertical block on its own)
            crops = await asyncio.to_thread(lambda: [geometry.encode(image.crop(geometry.padded(
                geometry.bounds([p for line in chunk for p in line["poly"]]), 4, image.size))) for chunk in chunks])
        else:
            crops = [crop]
        error, chars, complete = None, [], True
        try:
            readings = await asyncio.gather(*(vlm.read(self.vl, piece, vlm.OCR, self.text_constraint,
                                                       self.settings.top_k + 1) for piece in crops))
        except Exception as failure:
            readings, error, complete = [], str(failure), False
        for content, tokens, finished in readings:
            if chars:
                chars.append(lattice.Char("\n", 1.0))
            chars += lattice.from_logprobs(content, tokens, self.allowed)
            complete = complete and finished
        content = lattice.text(chars)
        if layer and content and lattice.agreement(layer, content) >= TRUST_TEXT_LAYER:
            return {"final": layer, "source": "text_layer"}
        chars = lattice.strip(chars) if code else lattice.unwrap(chars)
        if vertical:
            # Latin letters lie sideways in vertical text (N looks like Z); OCR reads each line turned
            # upright, so on Latin letters the VL reader barely votes.
            chars = [lattice.Char(c.c, c.p / 10, {k: q / 10 for k, q in c.alts.items()})
                     if c.c.isascii() and c.c.isalnum() else c for c in chars]
        # The VL reader leads (context, reading order) unless it was cut off or clearly skipped lines OCR
        # read; then OCR leads and the VL reading only confirms. (More VL text than OCR usually means the
        # line detector missed lines, e.g. on a warped scan.)
        found = sum(c.c.isalnum() for c in chars)
        seen = sum(c.c.isalnum() for c in ocr)
        vl_leads = bool(chars) and complete and found >= 0.9 * seen
        merged = lattice.merge(chars, ocr) if vl_leads else lattice.merge(ocr, chars)
        segs = lattice.segments(merged, self.settings.threshold, self.settings.top_k)
        reading = {"segs": segs, "crop": crop, "slots": lattice.slots(segs),
                   "source": "ocr+vl" if vl_leads else "ocr" if chars else "ocr_only"}
        if error:
            reading["error"] = error
        return reading

    async def read_table(self, image, block, crop, use_layer):
        if use_layer and block.ruled:  # Drawn grid + text layer: exact text and merged cells, no model.
            table = tables.from_grid(block.ruled["rows"], block.ruled["cols"], block.ruled["cells"])
            if not block.lines or tables.coverage(table, [l["chars"] for l in block.lines]) >= MIN_TABLE_COVERAGE:
                return {"table": table, "cell_segs": [lattice.segments(c.chars) for c in table.cells],
                        "crop": crop, "slots": 0, "source": "text_layer_ruled"}
        try:
            content, tokens, _ = await vlm.read(self.vl, crop, vlm.TABLE, self.table_constraint,
                                                self.settings.top_k + 1, max_tokens=8192)
            table = tables.parse_otsl(content, lattice.from_logprobs(content, tokens, self.allowed))
        except Exception as failure:
            table, error = None, str(failure)
        else:
            error = None if table else "no OTSL table in the reply"
        if table is None:  # Keep the table's text rather than lose it.
            reading = await self.read_text(image, replace(block, label="text"), crop, use_layer)
            reading["error"] = error
            return reading
        # A table that lost text OCR saw (a dropped column, say) is rebuilt as a plain grid from the
        # OCR lines: complete content beats spans.
        ocr_lines = [line["chars"] for line in block.lines]
        if block.lines and tables.coverage(table, ocr_lines) < MIN_TABLE_COVERAGE:
            grid = tables.from_lines(block.lines)
            if tables.coverage(grid, ocr_lines) > tables.coverage(table, ocr_lines):
                table, error = grid, "table model dropped text: plain grid from OCR lines"
        trusted = bool(use_layer and block.chars)
        if trusted:
            tables.reconcile(table, [{"box": box, "chars": lattice.locked(text)}
                                     for text, box in geometry.text_layer_boxes(block.chars, False)], True)
        elif not error:  # A grid rebuilt from OCR lines must not "agree" with those same lines.
            tables.reconcile(table, block.lines)
        cell_segs = [lattice.segments(cell.chars, self.settings.threshold, self.settings.top_k)
                     for cell in table.cells]
        reading = {"table": table, "cell_segs": cell_segs, "crop": crop,
                   "slots": sum(lattice.slots(s) for s in cell_segs),
                   "source": "text_layer+table_vl" if trusted else "ocr+table_vl"}
        if error:
            reading["error"] = error
        return reading

    async def finish(self, reading, context):
        """Final text: fill the slots with the constrained model, or the readers' best guess."""
        if reading.get("final") is not None:
            return reading["final"]
        table = reading.get("table")
        if table:
            fallback = tables.render(table, [tables.escape(lattice.best(s)) for s in reading["cell_segs"]])
        else:
            fallback = lattice.best(reading["segs"])
        if not reading["slots"]:
            return fallback
        if table:
            shown, pattern = tables.template_and_regex(table, reading["cell_segs"])
            prompt = vlm.TABLE_PROMPT
        else:
            shown, pattern = lattice.template(reading["segs"]), lattice.regex(reading["segs"])
            prompt = vlm.FILL_PROMPT
        reading["template"] = shown  # The report keeps every uncertain block for review and tuning.
        if not self.filler:
            return fallback
        try:
            answer = await vlm.fill(self.filler, reading["crop"], shown, pattern, prompt, context)
        except Exception as error:
            reading["error"] = f"fill: {error}"
            return fallback
        valid = re.fullmatch(pattern, answer) and ("table" not in reading or len(tables.parse_html(answer)) == 1)
        reading["filled"] = bool(valid)
        if not valid:
            reading["answer"] = answer
        return answer if valid else fallback


async def parse(settings, inputs, out_dir):
    parser = Parser(settings)
    try:
        await asyncio.gather(*(parser.run(path, out_dir) for path in inputs))
    finally:
        await parser.close()


def main(argv=None):
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="PDF files or folders of PDFs")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--model-server", default=env("PDF_PARSER_MODEL_SERVER", "http://127.0.0.1:8119"))
    parser.add_argument("--vl-url", default=env("PDF_PARSER_VL_URL", "http://127.0.0.1:8118/v1"))
    parser.add_argument("--vl-model", default=env("PDF_PARSER_VL_MODEL", "PaddleOCR-VL-1.6"))
    parser.add_argument("--fill-url", default=env("PDF_PARSER_FILL_URL"),
                        help="instruction-following VLM for slots (default: none, keep the readers' best guess)")
    parser.add_argument("--fill-model", default=env("PDF_PARSER_FILL_MODEL"))
    parser.add_argument("--fill-key", default=env("PDF_PARSER_FILL_KEY"))
    parser.add_argument("--judge-url", default=env("PDF_PARSER_JUDGE_URL"), help="default: no judge")
    parser.add_argument("--judge-model", default=env("PDF_PARSER_JUDGE_MODEL"))
    parser.add_argument("--judge-key", default=env("PDF_PARSER_JUDGE_KEY"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.95, help="confidence that locks a character")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--pages", type=int, default=8, help="pages in flight")
    parser.add_argument("--vl-concurrency", type=int, default=32)
    parser.add_argument("--fill-concurrency", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="page rendering processes")
    parser.add_argument("--no-text-layer", action="store_true")
    parser.add_argument("--furniture", action="store_true", help="keep headers, footers and page numbers")
    args = parser.parse_args(argv)
    inputs = [p for path in args.inputs for p in (sorted(path.glob("*.pdf")) if path.is_dir() else [path])]
    settings = Settings(
        args.model_server, args.vl_url, args.vl_model, args.fill_url, args.fill_model, args.fill_key,
        args.judge_url, args.judge_model, args.judge_key, args.dpi, args.threshold, args.top_k, args.pages,
        args.vl_concurrency, args.fill_concurrency, args.workers, not args.no_text_layer, args.furniture)
    asyncio.run(parse(settings, inputs, args.output))
