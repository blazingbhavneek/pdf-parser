# Japanese document benchmark

`main.py` prepares single-page test PDFs, runs independent parser environments,
judges their Markdown against the original page, and ranks difficult pages.
The main PDF parser package is separate from this benchmark.

See [Japanese stress cases and datasets](research/japanese_stress_cases.md) for
verified MinerU/Paddle failures and public Japanese benchmark sources.
See [augmentation research and training policy](research/document_augmentation.md)
for the evidence behind the capture profiles and how to use mined failures for fine-tuning.
The [premade setup scripts](datagen/premade/README.md) recreate the current public
stress inputs. [HANDOFF.md](HANDOFF.md) records the current state and next work.

## Setup

The benchmark needs its own lightweight environment. Parser models belong in the
separate environments you are installing, not this one.

```bash
uv venv benchmark/.venv --python 3.13
uv pip install --python benchmark/.venv/bin/python -r benchmark/requirements.txt
PLAYWRIGHT_BROWSERS_PATH="$PWD/benchmark/.browsers" benchmark/.venv/bin/python -m playwright install chromium --no-shell
cp benchmark/.env.example benchmark/.env
```

Edit `benchmark/.env` with the parser venv paths. Relative paths resolve from the
`.env` file's directory; absolute paths are simplest. Shell environment variables
override `.env`. The supplied endpoint/model are already in the example.
The local `.env`, browser files and every document/result under `io` are ignored by Git.
Linux, Japanese fonts (e.g. Noto Sans CJK JP), and either LibreOffice or your Office
conversion endpoint are required. If Chromium reports missing OS libraries, run
Playwright's `install-deps chromium` command using your system package manager privileges.

## Files

```text
benchmark/
  main.py
  .env
  datagen/io/
    raw/
      doc_name.pdf                  # also .docx and .pptx, flat directory
    input/doc_name/
      manifest.json                 # source hash, settings, versions, completion
      source.pdf                    # immutable converted document
      original/
        page_0001.pdf               # original objects and selectable text
        page_0001.png               # high quality original reference
        page_0001.hint.json          # unverified native text + PyMuPDF4LLM Markdown
      chars_rem/
        page_0001.pdf               # one lossless raster, no text layer or fonts
      augmented/
        page_0001_scan_mild.pdf
        page_0001_scan_mild.json     # seed, severity, effects, geometry, review flag
        page_0001_scan_medium.pdf
        page_0001_scan_hard.pdf      # likewise for all eight profiles
    output/doc_name/mineru/          # also docling, paddle, dots
      original/                     # also chars_rem and augmented
        page_0001_attempt1.md
        page_0001_attempt1.run.json  # command, dependency versions, input hash, runtime
        .artifacts/                 # untouched parser outputs, images and logs
        .judge/page_0001_attempt1/
          reference.png
          candidate.png
          candidate.pdf             # exactly one page, original physical dimensions
          response.json             # complete model response for inspection
      augmented/
        page_0001_scan_mild_attempt1.md  # every profile/severity and attempts 2/3
    scores.json
    difficult_pages.json
```

Names use the raw filename stem. If multiple formats share a stem, new folders
get a suffix such as `doc_name__docx`; existing folder identities stay stable.
Ambiguous names are rejected rather than overwritten. Use distinct raw filenames
for different versions of a document.

## 1. Prepare

Place documents in `benchmark/datagen/io/raw/`, then run:

```bash
benchmark/.venv/bin/python benchmark/main.py --prepare
```

Defaults: 300 DPI, seed 42, eight augmentation profiles at mild/medium/hard severity.
Each source page yields **26 PDFs**: original, clean image-only, and 24 augmented
image-only PDFs. For N source pages, this is 26N single-page inputs, before any
retained legacy variants. `chars_rem` is a lossless high resolution rendering of
the original, embedded into a fresh PDF; it removes selectable/hidden text,
fonts, annotations and attachments, while preserving their visible appearance.

```bash
benchmark/.venv/bin/python benchmark/main.py --prepare --augmentations scan photo --severities mild medium
benchmark/.venv/bin/python benchmark/main.py --prepare --documents "report" --limit 1
```

Completed documents are skipped. Interrupted preparation repairs missing files.
Selecting additional profiles/severities appends those exact pairs. Existing
variants remain, even when omitted from a later command; nothing is pruned or
overwritten. Old `page_0001_scan.pdf` inputs and their outputs are retained, and
missing legacy files use the original algorithm when repaired. Changing a raw
file, DPI or global seed is rejected for an existing document: use a fresh
`--io-dir` or rename the raw file so historical parser results stay valid.
`--convert` is an alias for `--prepare`.

For Office conversion, `CONVERTER_URL` receives a POST with multipart field
`file` and must return PDF bytes. `CONVERTER_API_KEY` is an optional bearer token.
If the endpoint is unset, unavailable, or returns an invalid PDF, LibreOffice
is used with a temporary isolated profile. An endpoint with a different contract
will need a small adapter once you provide its API details.

Hints are extracted from each converted PDF page. This keeps DOCX text aligned
to the actual rendered page; whole-document Pandoc text has no reliable DOCX page
boundaries. PyMuPDF4LLM extraction uses no OCR, with native PyMuPDF text as fallback.
Hints are never labeled verified ground truth and are never given to the parsers.

### Augmentation choices

The profiles cover document capture defects described in
[Augraphy](https://arxiv.org/abs/2208.14558),
[Real5-OmniDocBench](https://arxiv.org/abs/2603.04205), and Japanese
[Synth-JDoc](https://github.com/llm-jp/synth-jdoc). These simulations are inspired by
those sources; they do not reproduce a physical scanner or camera exactly.

| Profile | Effects |
| --- | --- |
| scan | Skew fitted inside the canvas, downsample/upscale, defocus, faded contrast, noise, JPEG artifacts |
| motion | Directional motion blur with sampled angle and increasing length |
| photo | Perspective, uneven lighting/shadow, defocus, lower resolution, noise, JPEG artifacts |
| resolution | Isolated downsample/upscale; useful for finding small-kanji and fine-table-line failures |
| compression | Isolated JPEG artifacts at decreasing quality |
| ink | Partial stroke thinning or spreading and fading; useful for dense kanji, punctuation and table rules |
| warp | Smooth page curvature with all source edges fitted inside the canvas |
| screen | Approximate screen-photograph interference, perspective, resampling, contrast, noise and JPEG artifacts |

These are small Pillow/NumPy implementations; no Augraphy installation is needed.
All augmented PDFs contain one raster image and no fonts or text layer. The
original page size and raster dimensions stay fixed. Spatial transforms keep the
entire source rectangle in view. There is no deliberate cropping, text insertion,
mirroring or opaque occlusion. Fine details can still become illegible from
degradation, so **every new variant has `training_review_required: true`**.

Seeds derive from source hash, page, profile and global seed. Severity levels use
the same seed to keep the defect direction/pattern comparable; strength increases,
but parser error counts need not increase monotonically. Parameters, intermediate
resolution, and geometry matrices/mesh are saved alongside each PDF. Pixel blur
lengths scale with the page's raster short edge relative to A4 at 300 DPI.
Rasterizing an existing low-resolution scan at 300 DPI cannot restore detail.

Preparation runs image effects on CPU; it does not load an OCR model or use the
GPU. The complete grid increases storage and parsing time substantially. Start
with `--severities mild medium` or selected profiles when throughput matters.
Use the clean reference and **reviewed, corrected Markdown** as supervision.
Hard cases are stress tests until reviewed; parser outputs and LLM scores are not
ground-truth labels. See the research note for splitting and sampling guidance.

## 2. Parse

Set `MINERU_VENV`, `DOCLING_VENV`, `PADDLE_VENV`, and optionally `DOTS_VENV`.
Then:

```bash
benchmark/.venv/bin/python benchmark/main.py --parse
benchmark/.venv/bin/python benchmark/main.py --parse --pipelines mineru paddle --attempts 4
benchmark/.venv/bin/python benchmark/main.py --parse --pipelines paddle --variants augmented --limit 1
```

The default is three independent attempts per input. Existing successful attempts
are skipped when input and Markdown hashes still match. Nonzero exits, timeouts,
missing Markdown or ambiguous multiple outputs are failures, retain their logs,
and do not become successful attempts. Empty Markdown is allowed: it may be a
blank page or a serious extraction failure, which the judge must assess.

Built-in adapters:

| Name | Invocation |
| --- | --- |
| mineru | Major 2/3: legacy CLI with `-l japan`; major 4+: direct `mineru-kit parse ... -o result.md` |
| docling | CLI, EasyOCR `ja,en`, accurate tables, referenced images |
| paddle | `PPStructureV3(lang="japan")`, explicitly save Markdown and JSON |
| dots | Your `DOTS_COMMAND`, since checkout/module names and inference servers vary |

References: [MinerU](https://github.com/opendatalab/MinerU/blob/master/docs/en/usage/quick_usage.md),
[Docling CLI](https://docling-project.github.io/docling/reference/cli/),
[PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PP-StructureV3.en.md),
[dots](https://github.com/rednote-hilab/dots.ocr).
Install the selected Docling OCR backend (`docling[easyocr]`) in its environment.
For Docling 2.138 with RapidOCR and `onnxruntime` installed, the local `.env` can instead use
`DOCLING_COMMAND='docling convert {input} --to md --output {output} --ocr-engine rapidocr --ocr-lang iso:ja --table-mode accurate --image-export-mode referenced'`.
This explicitly selects Japanese recognition and supports that release's `convert` subcommand.
Paddle's Japanese setting is deliberate: its default recognizer is Chinese/English.
MinerU 4's parser handles language within its model; its direct `mineru-kit` CLI
avoids the document-center server and cache required by `mineru parse`.

`*_COMMAND` overrides support `{python}`, `{venv}`, `{input}`, `{output}`, `{attempt}`.
Commands run without a shell, using the selected environment's Python and executables.
Arguments are split before path substitution so spaces/Japanese filenames are safe.
For example, with an installed dots.ocr module and its inference server running:

```dotenv
DOTS_VENV=/absolute/path/to/dots-env
DOTS_COMMAND='"{python}" -m dots_ocr.parser "{input}" --output "{output}"'
```

Check your checkout's `--help`; newer dots.mocr checkouts use a different module.
The command must emit one full-page `.md` somewhere inside `{output}`. `_nohf.md`
variants are excluded because this benchmark includes headers and footers.
Outputs keep their original Markdown bytes; `.run.json` records the original
artifact directory so the renderer resolves local image paths correctly.
When viewing Markdown outside this benchmark, open its artifact copy to resolve images.

Attempts are sequential to avoid GPU contention. Each attempt starts a new process
and reloads models. This favors environment isolation and restart safety; it is
not a throughput benchmark. Deterministic parsers may produce identical attempts.
Dependency versions, command, and elapsed time are recorded for each successful run.

## 3. Score

```bash
benchmark/.venv/bin/python benchmark/main.py --probe
benchmark/.venv/bin/python benchmark/main.py --score --limit 1
benchmark/.venv/bin/python benchmark/main.py --score
benchmark/.venv/bin/python benchmark/main.py --score --redo
benchmark/.venv/bin/python benchmark/main.py --score --redo "doc_name/mineru/augmented"
```

The supplied OpenAI-compatible endpoint receives one original reference image,
the candidate Markdown text, its rendered image, and unverified extraction hints.
The reference always comes from `original/`, including for degraded inputs.
HTML tables preserve `rowspan`/`colspan`. Rendering uses Japanese fonts, fits all
content onto one page without clipping, and saves the PDF, PNG and fitting scale.
Remote images and document scripts are blocked; missing images fail scoring rather
than silently degrading the comparison. Mathematical TeX is displayed as text.
Very large renders (>40,000 CSS pixels tall or >12,000 wide) require tiled judging
and are left unscored. Default judge resolution is 180 DPI; `--judge-dpi 300` helps
small characters if your model supports the image size.

Each successful judgment is saved immediately and atomically in `scores.json`.
The cache includes original/input PDFs, Markdown, hints, image assets, prompt,
model, endpoint, render dependencies and judge DPI. Changed content is rescored;
failed or malformed judgments are retried on the next run. `--redo` forces a fresh
judgment for all outputs or the supplied output path. One command at a time can
write to an IO directory; independent datasets can run in separate IO directories.

The judge reports substitutions, deletions, insertions, Chinese leakage, table
structure errors (including complex spans), and layout/reading-order errors,
plus example evidence, confidence, uncertainty and table assessments.

```text
character_errors = substitutions + deletions + insertions
score = character_errors + table_structure_errors + layout_errors
```

Lower is better. Leakage is a subset of substitutions and is never added twice.
Japanese kanji and shared Han codepoints are valid; only an incorrect source
character substitution counts as leakage. The code does no Japanese/Chinese
conversion or Unicode normalization. Rendered glyphs use Japanese regional fonts.

LLM counts and `estimated_cer` are estimates for mining difficult examples, not
verified character error rates or GriTS/TEDS metrics. Low-confidence judgments
and Chinese-variant flags need human review. Reflowed Markdown cannot prove exact
source geometry, so the prompt scores logical layout rather than font/margin matching.
Raw error counts favor longer pages; use the saved character-error ratio when
comparing pages of very different lengths. A one-page fitted render can shrink
long tables; original/candidate text and saved scale make this limitation visible.

## 4. Select difficult pages

```bash
benchmark/.venv/bin/python benchmark/main.py --rank --top 200
benchmark/.venv/bin/python benchmark/main.py --rank --top 100 --pipelines mineru paddle
```

Ranking deduplicates to original document/page. It averages attempts within each
pipeline/variant/augmentation-profile group, then averages those group means.
It exports `difficult_pages.json` with original PDF paths, per-group breakdowns,
worst group means and coverage. Stale/error judgments are excluded. Compare pages
with the same pipeline/variant coverage; partial runs can otherwise distort ranking.
`--documents`, `--pipelines`, `--variants`, and `--export` narrow the selection.

## Verify

```bash
benchmark/.venv/bin/python -m unittest benchmark.test_benchmark -v
```

Tests cover lossless text removal, deterministic degradation, interrupted-run
recovery, additive legacy migration, severity and geometry bounds, isolated
attempts, score arithmetic/cache invalidation, original-reference
selection, ranking, and real Chromium rendering with merged table cells/local images.
The OCR model installations and their model downloads are managed separately.
