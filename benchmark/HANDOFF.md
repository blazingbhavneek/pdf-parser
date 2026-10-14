# Agent handoff — 2026-10-12

## Goal and user decisions

Build a fast, accurate company Japanese PDF-to-Markdown pipeline. First build a
resumable failure-mining benchmark for PDF/DOCX/PPTX, image-only versions and
realistic augmentations, isolated MinerU/Paddle environments, and eventually a
local multimodal judge. Prioritize Japanese characters, source-specific Chinese
variant substitutions, reading order, table content and complex row/column spans.

The user excluded Docling from further comparisons. They challenged weak/default
Japanese configurations: compare explicit strong profiles fairly, retain the
earlier runs, and never describe default configurations as exhaustive tuning.
They asked to stop for the night after a quick one-page check and this handoff.
Do not leave inference jobs running.

## Workspace and environments

- Repo: `/mnt/common/Code/pdf-parser`; no `.codegraph/` index. The repository is
  currently untracked in Git; no commit or publish was requested.
- Benchmark Python: `benchmark/.venv/bin/python` (Python 3.13).
- MinerU: `/home/blaze/.venvs/mineru_venv/bin/python`; MinerU 4.0.12, CUDA/vLLM,
  local MinerU2.5-Pro-2605-1.2B. Use direct `mineru-kit parse`, tier Advanced for
  strong-profile comparison, and `--ocr-mode ocr` for OCR experiments.
- Paddle: `/home/blaze/.venvs/paddlepaddle_venv/bin/python`; PaddleOCR 3.7.0,
  PaddlePaddle GPU 3.3.1. VL 1.6, PP-OCRv6 Medium and Japanese PP-OCRv3 cached.
- Docling environment exists at `/home/blaze/.venvs/docling_venv`; excluded.
- GPU: RTX 5060 Ti, 16 GB. Run one GPU pipeline at a time.
- Local multimodal endpoint supplied by user: `http://82.79.85.125:42533/v1`,
  model `gemma-4-31B`. Settings live in ignored `benchmark/.env`; do not print keys.

## Implemented benchmark

`benchmark/main.py` provides `--prepare` (`--convert` alias), `--parse`, `--score`,
`--rank`, and `--probe`. CLI help and `benchmark/README.md` document usage.

Preparation converts Office files via optional endpoint then LibreOffice fallback,
splits pages, preserves originals, makes lossless text-free rasters, and saves
unverified native/PyMuPDF4LLM hints. Inputs and outputs are atomic/resumable.
Scoring always references the clean original, sends candidate text plus its
one-page rendered image, and caches JSON judgments. Counts are LLM estimates,
not verified CER/TEDS. The judge is not yet calibrated on these known failures.

New augmentation version 8 writes **one stacked `aug` page per source page**: every
effect is applied to the same image, each at its own random level. There are no
profiles, severities or per-stage files to select. Each knob has one range from the
mildest useful value to the strongest still-recoverable one — skew 0.5–15°,
perspective 1.5–15%, curvature 0.8–7.5%, motion 2.5–14 px, resolution 0.22–0.80,
JPEG 20–86, ink 0.25–0.88, moiré 4–28 — sampled as `mild + (strong-mild) * t` with
drawn uniformly (`BIAS = 1`) rather than leaning mild. `SPAN = 0.5` halves every range,
so no knob reaches the strong end of the table and a page cannot stack everything at full
strength; raise SPAN toward 1.0 for the full documented spread, or BIAS above 1 to favor
gentle pages again.
The resolution reducer is commented out in `datagen/augment.py` (version 5 stacked it
with everything else and destroyed too many pages); re-enabling it means another
VERSION bump because it also consumes a random draw.
Default: **aug + original + chars_rem = 3 PDFs per source page**.
Preparation runs pages of a document through a process pool: `--workers N` (default 4,
1 = serial), measured 3.7× at 4 workers and independent of the sampled values. Parse
stays sequential because attempts reload models onto one GPU.
The prepare manifest in `cache/input/<document>.json` saves seed, exact effects,
geometry matrices/mesh and `training_review_required: true` per page. CPU image processing; no OCR model involved.

Existing inputs/outputs are never pruned or overwritten. Manifests from another
augmentation version are rejected rather than silently mixed with new pixels — prepare
into a fresh IO directory. Global seed, DPI and source changes still require a fresh
IO directory or renamed source. Source hash/page/global seed determine randomness.
Filenames are `page_0001_aug.pdf` / `page_0001_aug_attempt1.md` throughout
preparation, scoring and ranking. Trade-off to remember: one stacked page per page
loses the isolated per-condition variants that used to attribute a failure to a single
defect, and costs roughly the sum of the old stages per page (~16 s/page, 300 DPI CPU).

Research and training recommendations:
`benchmark/research/document_augmentation.md`. Primary sources include Augraphy,
Real5-OmniDocBench, Synth-JDoc and Nougat training transforms. Split by source and
template family before augmentation, preserve verified targets, review legibility,
and mix clean pages with the low half of each range plus a reviewed strong subset. Training gains remain
unmeasured; synthetic distortion alone cannot supply new layout/table diversity.

## Current corpus and evidence — retain it

The four stress PDFs now sit in the shared raw corpus `benchmark/datagen/io/raw/`
(ignored by Git); the per-dataset IO directory and its run artifacts were removed
on 2026-10-14, and `premade/japanese_stress_setup.py` rebuilds the PDFs on demand.

| Document folder | Page |
| --- | --- |
| `jasyn_710_00_vertical` | Synthetic Japanese newspaper with vertical columns |
| `jasyn_746_02_spans` | Business manual, two annotated tables, 67 cells, 11 merged cells |
| `jcs2023_p031_table` | Real JCS guideline PDF page 31, dense medical table |
| `nta2025_tax_form` | Real NTA corporate tax form, tiny text and irregular cells |

`sources.json` has URLs/hashes/transformations; `sources/` retains downloads and
`ground_truth/` the selected JASyn annotations. JASyn images are natively only
794×1123; rendering at 300 DPI cannot restore detail. NTA originally has a
2481×3508-point media box around a same-sized pixel image; fitting it to A4 avoids
a 151 MP render. JASyn is for evaluation only per its card, not fine-tuning.

Eight initial outputs were written as `output/<document>/{mineru,paddle}/chars_rem/`
before the 2026-10-14 layout change; that shape is now
`output/{mineru,paddle}/<document>/page_0001_chars_rem_attempt1.md`.
These used MinerU Standard and PP-StructureV3 with Japanese-capable shared
PP-OCRv5. Do not pool them with stronger profiles as though the settings match.
Initial findings/inspection tools are in `findings.json`, `observations.json`,
`inspect_outputs.py` and `evidence/` under the IO root.

Nine stronger-profile reruns are under `fairness_audit/`:

- MinerU Advanced: newspaper and NTA. Fixed tax row-label association and reading
  order; still `NC加工` → `ZC加工`, `MTTR` → `MITTR` in newspaper body.
- PaddleOCR-VL 1.6: all four pages. Fixed many earlier failures; all 67 manual
  table cell tuples match. Still `統計値` → `統計值`, `NC加工` → `NO加工`,
  `DPP-4阻害薬` → `DPP-4阻帯薬`, tax `繰り越す` → `操り越す`.
- PP-OCRv6 Medium detection/recognition in general and table OCR: manual/medical;
  65/67 manual cell tuples match; `統計值` substitution remains.
- Dedicated Japanese PP-OCRv3 recognizer in general/table paths: manual; fixed
  tested Chinese variants but worse overall (36/67 cells; `MPa` → `山F`).

See `benchmark/research/japanese_stress_cases.md` for source-aligned evidence and
configuration caveats. `fairness_audit/audit.json`, `paddle_profiles.py`, resolved
YAMLs, logs, hashes and `inspect_audit.py` preserve the actual runs.
The audit worker currently reads only `input/<document>/page_0001_chars_rem.pdf`.

Earlier corpora were prepared with the old severity grid. `--prepare` now rejects a
saved augmentation version other than 4, so prepare those documents into a fresh IO
directory rather than appending to them. Existing 38 checked input/output hashes
were unchanged at the time. Preview: `augmentation_preview/jasyn_746_02_spans-hard.jpg`.

## Latest one-page GPU check

PaddleOCR-VL 1.6 parsed manual `augmented/page_0001_scan_medium.pdf` successfully
in **31.63 seconds**, including model initialization. Explicit `gpu:0`, chart
recognition enabled, retain all labels, document orientation/unwarping disabled
to match the clean audit. This does not test preprocessing optimization for scans.

Artifacts: `fairness_audit/augmentation_smoke/` under the IO root:

- `run_smoke.py` reproduces the one-page check with the Paddle Python.
- `vl16/page_0001_scan_medium_0.md`, JSON, config, `completed.json` with hashes/time.
- `comparison.json`: clean VL matched **67/67** annotated cell tuples; medium scan
  matched **66/67**, with `フェーズ間でばらつき増加` → `フェーズ間でぼらつき増加`.
  Additional outside-table errors: `ポリカーボネート` → `ボリカーボネート`,
  `過剰在庫` → `過剥在庫`. Existing `統計値` → `統計值` remains.

These are targeted probes, not a full error count or TEDS metric. MinerU has not
been rerun on the new augmentation grid. No LLM judging was run tonight.

## Premade setup

Both setup scripts feed the single shared corpus `benchmark/datagen/io/raw/`; there
are no per-dataset IO folders.
`benchmark/datagen/premade/japanese_stress_setup.py` recreates all four selected raw
PDFs and the selected annotations from pinned/hash-checked public downloads, caching
its sources under `premade/cache/japanese-stress/` and copying the finished PDFs into
the shared corpus (`--no-publish` skips the copy, `--io-dir` relocates the build).
`benchmark/datagen/premade/japanese_public_setup.py` restores the 20 archived
public-sector Japanese PDFs from pinned Wayback snapshots with SHA-256 checks, using
`japanese_public_sources.json`. `japanese_stress_sources.json` carries the stress
source catalog; `premade/README.md` describes usage. These are selected corpora, not
downloaders for the full official benchmark suites.

```bash
benchmark/.venv/bin/python benchmark/datagen/premade/japanese_stress_setup.py
benchmark/.venv/bin/python benchmark/datagen/premade/japanese_public_setup.py
benchmark/.venv/bin/python benchmark/main.py --prepare
benchmark/.venv/bin/python -m unittest benchmark.test_benchmark -v
```

## Recommended continuation

Final validation: **18 unit/integration tests passed** (including real Chromium
rendering); 48 new augmentation PDFs and sidecars validated; the premade setup
passed both cached resume and fresh four-page creation from the retained sources.
All checked historical input/output hashes remained unchanged. GPU/inference
process checks were empty at handoff.

1. Read this handoff, both research notes, and current `.env` privately. Keep
   Docling excluded and retain all previous evidence.
2. Add explicit reusable strong-profile selection to parser workers/configuration.
   Main's built-in Paddle worker still uses PP-StructureV3 defaults, and MinerU's
   built-in direct CLI defaults to Standard. Do not call those maximal Japanese
   configurations. Keep profile names/config hashes separate in score groups.
3. Benchmark a small matched augmentation subset with MinerU Advanced and VL 1.6;
   consider appropriate orientation/unwarping for genuinely distorted inputs.
   Batch pages within one initialized worker to avoid repeated model startup.
4. Calibrate the local multimodal judge against the source-aligned known mistakes,
   including clean VL and medium-scan outputs. Check misses, false positives,
   table/character judgments and the actual multi-image API. Keep estimates
   separate from verified annotation metrics; use crops if tiny text is missed.
5. Expand the evaluation set and mine legible, diverse hard cases. Use authorized
   internal documents with reviewed Markdown for training, keeping premade held-out
   evaluation sources and all their augmented siblings out of training.

No process needs to remain running overnight. No installation/library upgrade,
commit, external message, publication or model fine-tuning was performed.
