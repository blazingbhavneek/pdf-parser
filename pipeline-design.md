# Japanese PDF → Markdown pipeline: design

Status (2026-10-14): built in `src/pdf_parser/` (run instructions in README.md); open items in TODO.md.
Changes from the first plan, from testing: fill-in is opt-in (an OCR model used as filler chose worse than
the readers' best guess); tall blocks are read by the VL model 8 lines at a time; the VL reading leads the
merge only when its length matches OCR's; small models run in our own batching model server until Triton
fits (its image needs more disk than this machine has).

## The rule everything follows

Only OCR models may propose characters. Every larger model (VLM/LLM) may only **choose**
between characters the OCR models proposed, and that choice is enforced by the decoder,
not by a prompt. That makes Chinese leakage structurally impossible instead of unlikely,
while still letting a smart model use the page context to pick the right kanji.

## Page flow

```text
PDF page
 ├─ 1. Text layer usable?  ──yes──► text-layer path (step 2) ──judge OK──► Markdown
 │                                                         └─judge fails─┐
 └─ no / garbled ────────────────────────────────────────────────────────┤
                                                                         ▼
    3. Layout + reading order (PP-DocLayoutV3)  → boxes: text, title, table, code, formula, figure
    4. OCR readers on every text box → character lattice (locked chars + ⟦A|B|C|D⟧ slots)
    5. Fill-in VLM per box with slots, decoder constrained to the lattice
    6. Tables: table VLM structure + OCR lattice cells → fill-in VLM → HTML
    7. Code/formula/figure formatting from the layout label
    8. Assemble page, merge tables split across pages → Markdown (HTML tables)
```

## 1–2. Text layer: use it only when proven faithful

Born-digital PDFs (like Word exports) often have perfect text, but the order of the text
stream can be wrong, and some exporters write garbled character maps. So:

1. **Never trust stream order.** Take each character with its bounding box
   (PyMuPDF `rawdict`), assign characters to the layout boxes from step 3, and order by the
   layout model's reading order. Stream order problems disappear.
2. **Cheap checks first:** `U+FFFD`, private-use characters, fonts with no ToUnicode map, mojibake
   signatures (MinerU's classifier checks these; see `/tmp/parsers/reports/mineru.md`).
3. **Per-box cross-check:** run the fast OCR (step 4, top-1 only) on the same box. If text layer
   and OCR agree, accept the box; if they disagree, that box goes through the OCR path. This is
   per box, not per page, so one bad font doesn't throw away a good page.
4. **Your judge:** render the assembled Markdown to an image (`benchmark/datagen/render.py`
   already does this) and ask the judge model whether it faithfully reproduces the page. Fail →
   full OCR pipeline for the page. In production, judge every page at first; once step 3 has
   measured agreement, judge only pages where some boxes disagreed.

## 3. Layout and reading order

**PP-DocLayoutV3** (Apache-2.0): about 25 classes, including `text`, `paragraph_title`,
`doc_title`, `table`, `algorithm` (code), `display_formula`, `header`, `footer`, `seal`,
`vertical_text`, and a **learned reading-order head**. MinerU and PaddleOCR-VL both use the V2/V3
family. Layout is language-independent, so a Chinese origin doesn't matter here.

## 4. OCR readers → character lattice

No single open model is "the best Japanese OCR". The evidence (sources below):

| Reader | Type / license | Japanese evidence | Top-k per character | Role |
|---|---|---|---|---|
| **PP-OCRv6 medium rec** | CTC, ≤34.5M params, Apache-2.0 | Paper: largest gain on its Japanese category (+16.8% vs v5); claims to beat GPT-5.5 / Gemini-3.1-Pro on its OCR set | Yes: per-frame softmax exists, the decoder averages it away today (`text_recognition/processors.py:230-253`) | Fast primary reader |
| **NDLOCR-Lite PARSeq** | autoregressive, Japanese-only vocabulary, CC BY 4.0 (released 2026-02) | Best *recognizer* CER on XFUND-ja in a May 2026 test: 12.08% vs YomiToku 12.27%; CPU-fast, ONNX | Yes: per-step softmax | Second, independent reader; cannot output simplified Chinese at all |
| **PaddleOCR-VL-1.6** (0.9B, vLLM) | VLM, `OCR:` prompt per box | Best open document parser (OmniDocBench 96.34); in this repo it still produced `統計値→統計值` | Yes: vLLM `logprobs` / `top_logprobs` per token | Context-aware third reader, run with a Japanese token mask (`allowed_token_ids`) |
| YomiToku | PARSeq-based, 7,000+ chars, **CC BY-NC-SA** | Best end-to-end on XFUND-ja (22% CER), best local handwriting | — | Benchmark only, unless you buy a commercial license |
| Gemini 3.1 Pro | cloud API | Best on handwriting tests | Partial | Not for confidential documents |

**Japanese allowlist (for decoding masks):** JIS X 0208 can be generated with the standard
library (`ch.encode("shift_jis")` succeeds), plus JIS X 0213 if needed (`euc_jis_2004`),
kana, ASCII, full-width forms and symbols, **plus every character in your company DOCX
corpus**, which is exact. `值` is not in JIS X 0208 and `値` is, so the mask blocks the
`統計值` leak at decode time.

**Building the lattice for one box:**

1. Run 2–3 readers with the mask on and keep top-4 per position.
2. Align their outputs character by character (edit-distance alignment).
3. **Lock** a character when the readers agree and its probability is ≥ 0.95. Agreement between
   independent models is a better confidence signal than one model's softmax. Calibrate the
   threshold on the benchmark later.
4. Everything else becomes a **slot**: the union of the readers' candidates, best 4 by summed
   probability.

Example: `本規程は会社の業務を円滑に遂行するための基本的な事項を⟦定|走|足⟧める。`

Use `⟦ | ⟧` as slot markers because `()` and `（）` are everywhere in Japanese text.

## 5. Fill-in VLM (your design, with a hard guarantee)

Only boxes that contain slots are sent. The request has the box crop, the page image (or
the neighbouring boxes' resolved text), and the template. The VLM writes the whole block, and
**vLLM structured outputs constrain the decoder with a regex built from the template**:
locked text is literal, and each slot is `(定|走|足)`. A small optional Markdown prefix is
allowed (heading `#`, list `- `/`1. `), but heading level comes from the layout label anyway.

```python
extra_body={"structured_outputs": {"regex": template_regex}}   # vLLM ≥ 0.12 name
```

The model physically cannot emit a character outside the lattice, so any strong VLM can do
this job, including Chinese-trained ones. Candidates: Gemma-4-31B (already on your endpoint),
Qwen3-VL. If no candidate fits, the VLM picks the most likely one, and the box is logged as
low confidence for review.

## 6. Tables

MinerU Advanced and PaddleOCR-VL make good tables because a VLM writes the whole table as OTSL
(`<fcel> <lcel> <ucel> <nl>` …), which is converted to HTML with rowspan/colspan. The weak point
is that the VLM also writes the cell text freely. Best published table structure (OmniDocBench
v1.6 TEDS): TeleOCR ~97 (license unclear), PaddleOCR-VL-1.6 94.8, MinerU2.5-Pro 93.4.

1. **Structure:** PaddleOCR-VL-1.6 `Table Recognition:` on the table crop, with the Japanese token
   mask and `top_logprobs`, gives an HTML grid plus its own cell text.
2. **Text:** the OCR readers on the same crop produce a lattice per line.
3. **Reconcile:** align the VLM's cell texts (in reading order) with the OCR lattice. Characters
   where both agree are locked; disagreements become slots in each cell.
4. **Fill-in VLM:** writes the final `<table>` from the image plus the skeleton
   `<td rowspan="2">売上⟦高|髙⟧</td>…`. The regex allows tags and attributes to change between
   cells, but cell text must match the slot templates in order. So it can fix the structure, never
   the characters.
5. **Fallback structure** when step 1 is invalid: table classifier → SLANeXt/UNet + cell detector,
   with OCR boxes placed into cells by ≥70% containment (PP-StructureV3, MinerU `basic`).
6. **Tables split across pages:** Paddle `merge_table.py` heuristics, MinerU cross-page merge.

## 7. Code, formulas, figures

The layout label decides the Markdown. `algorithm`/code gets ``` fences, with lines from OCR and
slots resolved like text. Display formulas go through PP-FormulaNet or the VLM's
`Formula Recognition:` (LaTeX in `$$`). Figures become image links; their caption goes next to
them. The fill-in VLM may wrap a block in a special block (fence, quote) only when the layout
label allows it.

## 8. Serving: vLLM + Triton + one async orchestrator (no Docling)

```text
                ┌────────────── orchestrator (Python asyncio) ──────────────┐
PDFs ─► render pool (CPU procs) ─► layout ─► OCR det/rec ─► lattice ─► fill-in ─► assemble ─► MD
                                     │            │                         │
                               Triton (ONNX/TensorRT)               vLLM servers
                  PP-DocLayoutV3, PP-OCRv6 det+rec,          PaddleOCR-VL-1.6 (OCR/table),
                  NDLOCR-Lite PARSeq, table cls/SLANeXt      fill-in VLM, judge
```

- **Triton** keeps every small model loaded, with **dynamic batching** (`max_queue_delay` a few
  ms) and a fixed `instance_group` and `max_batch_size` per model, so their memory is known in
  advance. Group OCR line crops by width (≤256, ≤512, ≤1024, ≤2048 px) to avoid padding
  waste. PaddleX's own "high-stability serving" is built on Triton.
- **vLLM** does continuous batching for the VLMs; `--gpu-memory-utilization` reserves a fixed
  share of GPU memory, and `--max-num-seqs` caps concurrent requests.
- **Orchestrator:** pages flow through stages connected by **limited-size queues**. Many
  pages (32–64) are in flight, so every model always has work. Each backend has an in-flight
  limit (a semaphore) tuned so its server-side queue is never empty and never huge. Waiting
  pages sit in RAM, never in GPU memory.
- **No out-of-memory crashes, by construction:** fixed budgets per server, summing to ≤ 90% of
  GPU memory; capped batch sizes and input shapes; a warm-up at maximum shapes at startup to
  measure the real peak; and queue limits for admission control.
- **Measure before tuning:** Triton and vLLM both expose Prometheus metrics. The PaddleOCR-VL
  authors measured ~1.1 pages/s on an A10 and 0.54 on an RTX 3060 for their full pipeline
  under vLLM. SGLang notes CPU-side page preprocessing often caps throughput before the GPU
  does, so render pages in a process pool sized to the CPU.

On the 16 GB RTX 5060 Ti: Triton models ~3–4 GB, PaddleOCR-VL ~5 GB; the fill-in VLM won't
fit next to them at 31B, so use the remote endpoint or a quantized ~8B VLM until the production
GPU is known.

## Benchmark scoring (next to build)

Gold comes from `docx_converter` (`gold/<doc>/page_NNNN.md`).

- **Text:** compare only kana, kanji and Latin-letter sequences (digits and symbols ignored,
  as you proposed). Report CER from edit distance plus LCS-based sequence recall, per box and
  per page.
- **Leakage:** predicted characters outside the Japanese allowlist that don't appear in gold.
- **Tables** separately: TEDS, TEDS-struct, and cell-text CER.
- **Images** separately.
- **Header/footer** text is in `gold/<doc>/pages.json`, so it can be excluded from both sides.
- Pages with low `agreement` or `images > 0` are excluded from the strict text score.

## Build order

1. Scorer on gold (above). Everything after is measured with it.
2. Text-layer path + judge.
3. OCR lattice: PP-OCRv6 top-k decode patch with mask, NDLOCR-Lite top-k, PaddleOCR-VL logprobs.
4. Fill-in VLM with regex constraint.
5. Tables.
6. Serving harness.
7. Fine-tune PP-OCRv6 rec / PARSeq on line crops from the company DOCX corpus. The Word
   PDFs have a perfect text layer with character boxes, so you get unlimited exactly labelled
   Japanese lines in your own fonts and layouts. Keep the benchmark's held-out documents out.

## Sources

- PP-OCRv6 paper: https://arxiv.org/abs/2606.13108
- Japanese OCR comparison on XFUND-ja (NDLOCR-Lite, YomiToku, PaddleOCR, RapidOCR, Surya, EasyOCR): https://zenn.dev/acompany/articles/ddf8677f09e252
- NDLOCR-Lite: https://github.com/ndl-lab/ndlocr-lite
- Handwriting comparison of 23 models: https://nyosegawa.com/en/posts/japanese-handwriting-ocr-comparison/
- Japanese OCR comparison by use case (July 2026): https://note.com/daikidomon/n/nf4c45dabc7b4
- Multilingual OCR GPU benchmark: https://voiceping.net/en/blog/research-multilingual-ocr-gpu-benchmark-2026/
- Synth-JDoc (vertical Japanese, ICDAR 2026): https://arxiv.org/abs/2608.28248
- OmniDocBench leaderboard: https://github.com/opendatalab/OmniDocBench
- TeleOCR model card: https://huggingface.co/StarDoc-AI/TeleOCR
- InstructTable: https://arxiv.org/html/2604.02880v1
- vLLM structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs.html
- PaddleOCR-VL-1.5 paper (throughput): https://arxiv.org/pdf/2601.21957
- SGLang PaddleOCR-VL cookbook: https://docs.sglang.io/cookbook/autoregressive/Baidu/PaddleOCR-VL
- PaddleOCR serving (Triton high-stability mode): https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/serving/serving.html
- Code reading reports: `/tmp/parsers/reports/{mineru,paddle,docling}.md`
