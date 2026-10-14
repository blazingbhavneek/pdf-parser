# TODO

Everything agreed so far, so nothing gets lost. [x] = done, [ ] = open.

## DOCX converter (`docx_converter/`)
- [x] Word server for Windows: POST DOCX → zip of `document.pdf` + `pages/page_NNNN.docx` + `pages.json`
- [x] Word side freezes revisions, comments, list numbers (`ConvertNumbersToText`), fields before export
- [x] Client: health check (5 s), then upload; falls back to local LibreOffice on any failure
- [x] LibreOffice fallback (UNO): same layout, numbering frozen, pages rebuilt from paragraphs / table row runs
- [x] Benchmark `--prepare` uses it for `.docx`; `.pptx` stays plain LibreOffice
- [ ] Run `server.py` on the Windows laptop once and compare a few pages (never executed: no Word here)
- [ ] Header/footer text from shapes/watermarks is not captured (text only)
- [ ] Repeated table header rows on continuation pages are drawn in the PDF but not page content
- [ ] LibreOffice fallback: merged-cell tables that cross a page edge become one table per row (gold pages
      then fall out of `strict`)

## Gold answers
- [x] Pandoc (`--to=gfm`) per page DOCX → `gold/<doc>/page_NNNN.md`; `pages.json` agreement vs PDF text, image count, header/footer
- [ ] Optional: user's own DOCX XML parser instead of Pandoc (headings/list numbers already frozen by Word)
- [ ] Split documents into dev (tune on) and sealed test (only at milestones); ~500-page daily subset

## Gold scorer (`benchmark/gold.py`, separate from the LLM-judge `--score`)
- [x] Text: only kana / kanji / Latin letters; digits and symbols ignored; width-folded, NFC
- [x] CER, LCS recall, verbatim run recall; per page records + micro-averaged summary
- [x] Leakage: predicted kanji outside the Japanese allowlist and absent from gold; top substitutions
- [x] Tables separately: TEDS, TEDS-struct, aligned cell CER; images counted; missing/extra tables
- [x] Header/footer text removed from predictions; `strict` = agreement ≥ 0.98 and no images
- [ ] Pair tables by content instead of order when a page has several

## Pipeline (`src/pdf_parser/`)
- [x] Text layer: char boxes, geometric order, garble check, per-block cross-check with OCR (then VL)
- [x] Judge hook: render + judge for text-layer pages, re-read by OCR when unfaithful (needs a judge VLM)
- [x] Layout + reading order: PP-DocLayoutV3; unordered blocks (tables, captions) follow the block above
- [x] PP-OCRv6 medium: masked CTC, top-4 per character (model server)
- [x] PaddleOCR-VL-1.6 on vLLM: allowlist regex constraint + top logprobs; tall blocks read 8 lines at a time
- [x] Lattice: VL leads unless cut off or its length disagrees with OCR; agreement locks; disagreement → slot
- [x] Fill-in: regex from the template, opt-in only (an OCR model as filler chose worse than the best guess)
- [x] Tables: VL OTSL → cells with per-character lattices; reconcile with OCR / trusted text-layer lines;
      constrained fill may change structure, never text; split tables merged across pages
- [x] Code fenced, formulas via VL in `$`/`$$`, figures saved as images, furniture dropped by default
- [x] Async orchestrator: pages in flight, per-backend limits; model server with dynamic batching
- [x] Benchmark pipeline `ours`
- [x] Real-PDF run (2026-10-14, 4 stress pages) and fixes from it:
  - model server: line detection batches only same-size pages (mixed sizes crashed it)
  - ruled tables on text-layer pages come straight from the PDF (PyMuPDF `find_tables`): exact text,
    merged cells, vertical cell text, no model call; blocks inside the grid join the table
  - table cells collect every wrapped line (text layer and OCR); a line spanning cells is a fallback only
  - layout model's reading order used for every block (PaddleX only hides it for tables/captions/headers)
  - disagreements: the candidate both readers back most leads, not always the VL reader's
  - vertical text: tiers (headline above body read separately), Latin letters decided by OCR
    (sideways N reads as Z), unordered blocks placed right-to-left
  - blocks whose lines OCR cannot read at all, and tiny (< 7 pt) lines no layout block claims (lettering
    in a figure the layout model missed), are skipped: the VL reader only invents text there
  - per document, a long katakana word one voicing mark away from a word 3x as frequent is corrected
    (ボリカーボネート → ポリカーボネート); OCR text only, logged in the JSON report
- [ ] Figures the layout model misses (synthetic newspaper): larger lettering in them still becomes text,
      and their captions lose their place in the reading order
- [ ] White-on-colour text (mastheads, banners) is not detected
- [ ] Forms (NTA tax form): a mostly empty 26-column grid; not a target, not tuned
- [ ] Ruled tables from the text layer keep the PDF's letter spacing (`e GFR`, `m L`); scoring ignores it
- [ ] Run fill-in and judge with a capable VLM (remote Gemma-4-31B was unreachable; local GPU too small)
- [ ] Calibrate the 0.95 lock threshold on gold data (`ボリカーボネート`: both readers agree, wrongly)
- [ ] Layout misses: a borderless table's header row became separate text blocks above the table
- [x] Tilted scans are straightened (skew from detected lines, > 0.5°) before layout and detection
- [x] A table whose VL reading lost OCR-visible text (< 90%) is rebuilt as a plain grid from OCR lines
- [ ] Warped scans: unwarp with UVDoc (cached) before reading; today the detector finds few curved lines and
      the VL reader loops on long repetitive blocks (augmented long-paragraph pages lose most text)
- [ ] VL repetition loops on long repeated text: band-wise reading or a mild no-repeat constraint
- [ ] Triton instead of the model server when there is disk/production hardware (same batching role)
- [ ] NDLOCR-Lite PARSeq as an independent Japanese-only reader
- [ ] Batch mode for the benchmark: one process for many pages (startup is ~3 s per page today)
- [ ] Fine-tune PP-OCRv6 rec / PARSeq on line crops from company Word PDFs (exact labels); hold out test docs

## Experiments on existing parsers (cheap, config only)
- [ ] MinerU: `presence_penalty=0` for text → does leakage drop?
- [ ] Paddle: `lang='japan', ocr_version='PP-OCRv3'` (Japanese-only dictionary) baseline
- [ ] Docling: `ocr_mode=full_page`, RapidOCR `model_size=medium`, OCR scale 4
- [ ] Add PaddleOCR-VL-1.6 / MinerU2.5-Pro strongest profiles as baselines; YomiToku benchmark-only (CC BY-NC-SA)
