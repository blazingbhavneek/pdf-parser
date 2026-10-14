# Japanese document benchmark

`main.py` prepares single-page test PDFs, runs independent parser environments,
judges their Markdown against the original page, and ranks difficult pages.
The main PDF parser package is separate from this benchmark.

See [Japanese stress cases and datasets](research/japanese_stress_cases.md) for
verified MinerU/Paddle failures and public Japanese benchmark sources.
See [augmentation research and training policy](research/document_augmentation.md)
for the evidence behind the the capture stack and how to use mined failures for fine-tuning.
The [premade setup scripts](datagen/premade/README.md) recreate the current public
corpus: the four stress inputs plus 20 archived Japanese public-sector PDFs, all
restored into the single `datagen/io/raw/` corpus. [HANDOFF.md](HANDOFF.md) records
the current state and next work.

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
    input/doc_name/                 # PDFs only, one folder per document
      page_0001_original.pdf        # original objects and selectable text
      page_0001_chars_rem.pdf       # one lossless raster, no text layer or fonts
      page_0001_aug.pdf             # one stacked page; all knobs/geometry in cache/input/<doc>.json
    output/mineru/doc_name/         # also docling, paddle, dots; Markdown only
      page_0001_original_attempt1.md
      page_0001_chars_rem_attempt1.md
      page_0001_aug_attempt1.md    # every type and attempts 2/3
    score/
      scores.json                   # judgment database
      ranking.json                  # difficult-page ranking export
      mineru/doc_name/page_0001_aug_attempt1/
        reference.png               # judged reference page
        candidate.png               # rendered candidate
        candidate.pdf               # exactly one page, original physical dimensions
        response.json               # complete model response for inspection
    cache/                          # derived data, kept out of the corpus folders
      input/doc_name.json           # source hash, settings, versions, augment parameters, completion
      convert/doc_name.pdf          # PDF converted from .docx/.pptx, plus its conversion.log
      raster/doc_name/page_0001.png # high quality original reference
      hint/doc_name/page_0001.json  # unverified native text + PyMuPDF4LLM Markdown
      parse/mineru/doc_name/
        page_0001_aug_attempt1.run.json   # command, versions, hashes, runtime, artifact paths
        page_0001_aug_attempt1/<run-id>/  # untouched parser outputs, images and logs
```

Every input and output filename carries the conversion type — `original`,
`chars_rem`, or `aug` — because there are no variant folders.
`cache/raster`, `cache/hint` and `cache/parse` are regenerable and deletable;
`cache/input/<doc>.json` records what `input/<doc>/` contains, so delete that
manifest together with its input folder.

Names use the raw filename stem. If multiple formats share a stem, new folders
get a suffix such as `doc_name__docx`; existing folder identities stay stable.
Ambiguous names are rejected rather than overwritten. Use distinct raw filenames
for different versions of a document.

## 1. Prepare

Place documents in `benchmark/datagen/io/raw/`, then run. Company documents are
named `private_…` so confidential inputs stay recognisable.

```bash
benchmark/.venv/bin/python benchmark/main.py --prepare
```

Defaults: 300 DPI, seed 42, `--workers 4`. Pages of one document are prepared in
parallel processes, so `--workers` is the CPU-core knob (1 disables parallelism;
measured 3.7× on 14 pages at 4 workers). Sampled values do not depend on the worker
count. Parsing stays sequential on purpose — attempts reload models and share the GPU.
Each source page yields **3 PDFs** flat in `input/<document>/`: original, clean
image-only (`chars_rem`), and one stacked `aug` page that has **every** effect applied
at once. For N source pages, this is 3N single-page inputs. Each knob has one range
from the mildest useful value to the strongest still-recoverable one (skew 0.5–15°,
perspective 1.5–15%, curvature 0.8–7.5%, motion 2.5–14 px, resolution 0.22–0.80,
JPEG 20–86, ink 0.25–0.88, moiré 4–28), and every page draws one value per knob
independently and uniformly across the used part of its range (`BIAS = 1`), so every
strength is equally likely. Only the mild half of each range is used (`SPAN = 0.5`), so
the top of every range below is the nominal ceiling and the real maximum is half of it —
which also means a page cannot stack every effect at full strength.
The resolution stage is currently disabled — stacked with the rest it destroyed too
many pages.
`chars_rem` is a lossless high resolution rendering of
the original, embedded into a fresh PDF; it removes selectable/hidden text,
fonts, annotations and attachments, while preserving their visible appearance.

```bash
benchmark/.venv/bin/python benchmark/main.py --prepare --documents "report" --limit 1
benchmark/.venv/bin/python benchmark/main.py --prepare --workers 8   # more cores
```

Completed documents are skipped. Interrupted preparation repairs missing files.
Nothing is ever pruned or overwritten. Changing a raw
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

The stack covers document capture defects described in
[Augraphy](https://arxiv.org/abs/2208.14558),
[Real5-OmniDocBench](https://arxiv.org/abs/2603.04205), and Japanese
[Synth-JDoc](https://github.com/llm-jp/synth-jdoc). These simulations are inspired by
those sources; they do not reproduce a physical scanner or camera exactly.

Applied to every page, in this order, each stage at its own sampled level:

| Stage | Effect |
| --- | --- |
| perspective | Page mapped into an inset quadrilateral, saved homography |
| skew | Rotation fitted inside the canvas, saved homography |
| curvature | Smooth sine curvature with all source edges fitted inside the canvas, saved mesh |
| resolution | **Disabled in version 6** — downsample/upscale was too destructive in the stack |
| defocus + motion | Gaussian defocus, then directional motion blur at a sampled angle |
| ink | Stroke thinning or spreading plus toner fade; dense kanji, punctuation, table rules |
| contrast + shadow + moiré + noise | Faded contrast, uneven lighting, screen-photograph interference, sensor noise |
| compression | JPEG artifacts at a sampled quality and chroma subsampling |

These are small Pillow/NumPy implementations; no Augraphy installation is needed.
All augmented PDFs contain one raster image and no fonts or text layer. The
original page size and raster dimensions stay fixed. Spatial transforms keep the
entire source rectangle in view. There is no deliberate cropping, text insertion,
mirroring or opaque occlusion. Fine details can still become illegible from
degradation, so **every new variant has `training_review_required: true`**.

Seeds derive from source hash, page and global seed, so a page redraws the same
strengths every run; strength varies page to page, but parser
error counts need not increase monotonically. Parameters and geometry
matrices/mesh are saved alongside each PDF. Pixel blur
lengths scale with the page's raster short edge relative to A4 at 300 DPI.
Rasterizing an existing low-resolution scan at 300 DPI cannot restore detail.

Preparation runs image effects on CPU; it does not load an OCR model or use the
GPU. Because every effect now lands on one page, preparation costs roughly the sum of
the old stages per page (measured ~16 s/page at 300 DPI on CPU), dominated by the
motion-blur resampling passes.
Use the clean reference and **reviewed, corrected Markdown** as supervision.
Strongly degraded pages are stress tests until reviewed; parser outputs and LLM scores are not
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
Outputs keep their original Markdown bytes; the `cache/parse/<pipeline>/<document>/*.run.json`
marker records the artifact directory so the renderer resolves local image paths
correctly. Published Markdown links images from that cache directory, so when
viewing it outside this benchmark, open its artifact copy to resolve images.

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

Each successful judgment is saved immediately and atomically in `score/scores.json`.
The cache includes original/input PDFs, Markdown, hints, image assets, prompt,
model, endpoint, render dependencies and judge DPI. Changed content is rescored;
failed or malformed judgments are retried on the next run. `--redo` forces a fresh
judgment for all outputs or the supplied output path. One command at a time can
write to an IO directory; independent datasets can run in separate IO directories.
Scoring submits up to four LLM judgments concurrently. The request does not set an
explicit completion-token limit, so the configured model can use its available
reasoning/output budget.

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
pipeline/variant group, then averages those group means.
It exports `score/ranking.json` with original PDF paths, per-group breakdowns,
worst group means and coverage. Stale/error judgments are excluded. Compare pages
with the same pipeline/variant coverage; partial runs can otherwise distort ranking.
`--documents`, `--pipelines`, `--variants`, and `--export` narrow the selection.

## Verify

```bash
benchmark/.venv/bin/python -m unittest benchmark.test_benchmark -v
```

Tests cover lossless text removal, deterministic degradation, interrupted-run
recovery, full-range knob sampling and geometry bounds, isolated
attempts, score arithmetic/cache invalidation, original-reference
selection, ranking, and real Chromium rendering with merged table cells/local images.
The OCR model installations and their model downloads are managed separately.
