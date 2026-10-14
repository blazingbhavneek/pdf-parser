# PDF parser

Custom PDF-to-Markdown pipeline for internal Japanese documents. Design: [pipeline-design.md](pipeline-design.md);
open items: [TODO.md](TODO.md).

| Part | What it does |
|---|---|
| [`src/pdf_parser/`](src/pdf_parser/) | The parser: layout, OCR lattices with a Japanese-only mask, optional constrained fill-in, HTML tables |
| [`docx_converter/`](docx_converter/) | DOCX → Word's PDF + one DOCX per rendered page (Word server on Windows, LibreOffice fallback) |
| [`benchmark/`](benchmark/README.md) | Page variants (original, image-only, augmented), parser runs, LLM-judge mining, exact gold scores |

## Run the parser

Three processes; models stay loaded and requests from many pages are batched together.

```bash
uv sync --extra judge                                   # parser environment (.venv)

# 1. Small models (layout, line detection, recognition) on the GPU, in the Paddle environment
PYTHONPATH=src ~/.venvs/paddlepaddle_venv/bin/python -m pdf_parser.model_server --port 8119

# 2. PaddleOCR-VL-1.6 (block OCR with logprobs, tables, formulas)
~/.venvs/vllmServer/bin/vllm serve ~/.paddlex/official_models/PaddleOCR-VL-1.6 \
    --served-model-name PaddleOCR-VL-1.6 --port 8118 --max-logprobs 20 --gpu-memory-utilization 0.4

# 3. Parse
.venv/bin/python -m pdf_parser report.pdf more/ -o out/
```

Output: `out/<name>.md` (tables as HTML with rowspan/colspan), `out/images/`, and `out/<name>.json`, which
records for every block how it was read and, for uncertain blocks, the `⟦A|B|C|D⟧` template.

Optional, both OpenAI-compatible servers with structured outputs (vLLM):

- `--fill-url/--fill-model`: an instruction-following VLM (Gemma, Qwen3-VL) that resolves uncertain
  characters. Its decoder is held to the lattice, so it can only choose among OCR candidates. Without one,
  the readers' best guess is kept; an OCR model used as filler chose worse than that.
- `--judge-url/--judge-model`: for pages read from the PDF text layer, render the result and ask the
  judge whether it is faithful; if not, the page is re-read by OCR.

## Score against exact answers

`benchmark/main.py --prepare` turns `.docx` sources into per-page gold Markdown; after `--parse`:

```bash
benchmark/.venv/bin/python benchmark/gold.py            # writes score/gold.json, prints per pipeline/variant
```

This is separate from the LLM-judge `--score`, which finds mistakes when there is no gold answer.
