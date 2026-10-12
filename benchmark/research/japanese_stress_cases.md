# Japanese stress cases, 2026-10-12

Both MinerU and Paddle made verified mistakes on these selected pages, including
a row-label association error in MinerU and Chinese character substitutions in
Paddle. This is a failure-discovery set, not an overall model ranking.

## Local run

The data lives in `benchmark/datagen/io/japanese-stress-20261012/`, which is ignored
by Git. The run contains four single-page documents, 300 DPI image-only inputs,
and one attempt per pipeline. All eight MinerU/Paddle conversions completed.
No extra blur/noise was applied to the tested inputs. Docling is excluded from
this comparison at the user's request.

These initial runs used Japanese-capable defaults, not exhaustive tuning or the
strongest available configuration of each project. In particular, Paddle here
means **PP-StructureV3 with PP-OCRv5**, not PaddleOCR-VL, and MinerU used
**Standard**, not Advanced. The failure claims below apply to those exact runs.

| Document | Source | Difficulty |
| --- | --- | --- |
| `jasyn_710_00_vertical` | [OmniDocBench-JASyn, 710_00.png](https://huggingface.co/datasets/stockmark/OmniDocBench-JASyn/blob/73ecb24624682575bd5ebf138026e711589ae5c6/images/710_00.png) | Vertical newspaper columns, horizontal figures, acronyms, percentages |
| `jasyn_746_02_spans` | [OmniDocBench-JASyn, 746_02.png](https://huggingface.co/datasets/stockmark/OmniDocBench-JASyn/blob/73ecb24624682575bd5ebf138026e711589ae5c6/images/746_02.png) | Two tables with 11 merged cells, multilevel headers, parallel sidebars |
| `jcs2023_p031_table` | [Japanese Circulation Society guideline, PDF page 31](https://www.j-circ.or.jp/cms/wp-content/uploads/2023/03/JCS2023_fujiyoshi.pdf#page=31) | Dense table, vertical group labels, nested groups, small text |
| `nta2025_tax_form` | [National Tax Agency corporate tax form](https://www.nta.go.jp/taxes/tetsuzuki/shinsei/annai/hojin/shinkoku/itiran2025/pdf/01-01-s.pdf) | Image-only form, tiny text, irregular cells, labels tied to numbered fields |

The NTA PDF uses a 2481 × 3508 **point** media box around a 2481 × 3508 pixel
image. Its selected page was fitted to A4 without replacing that embedded image;
otherwise a 300 DPI render produces about 151 megapixels. Original downloads are
retained in `sources/`, and `sources.json` records URLs, hashes and transformations.
JASyn images were embedded in A4 PDFs; their native resolution remains a limit.

## Confirmed failures

| Pipeline | Page | Reference → output | Consequence |
| --- | --- | --- | --- |
| MinerU | Vertical newspaper | `NC加工` → `ZC加工`; `MTTR` → `MITTR` | Wrong acronyms |
| MinerU | Vertical newspaper | Main headline appears after body text; paper name/date omitted | Wrong reading order and missing blocks |
| MinerU | NTA form | `欠損金等の当期控除額` belongs to **26 / G48**, but output binds it to **27 / G49** | Wrong association between a label and its field |
| MinerU | NTA form | `翌期へ繰り越す欠損金額`, the actual **27 / G49** label, is absent | Missing field label |
| Paddle | Business manual | `一覧` → `一览`; `単位` → `单位`; `量産` → `量产`; `対処情報` → `对処情報` | Source-specific Chinese variant substitutions |
| Paddle | Vertical newspaper | A manager's `95` becomes `55`; another's `88` loses a digit | Wrong numbers |
| Paddle | Business manual | Troubleshooting header/body text shifts into wrong cells; several actions disappear | Broken cell associations despite matching row/span counts |
| Paddle | Medical table | `DPP-4阻害薬` row label is absent; first row's usage instructions appear under the main-action header | Missing content and incorrect columns |

These are source-aligned substitutions, not a blacklist of Han characters.
Japanese and Chinese share many characters; a shared character alone is not leakage.
Paddle ran with `PPStructureV3(lang="japan")`; its logs show
`PP-OCRv5_server_rec`. This configuration does not guarantee Japanese-only output.

MinerU also had clear successes: it recovered the newspaper's 95/88/72 numbers,
preserved the business manual's 11 row/column spans, and recovered the medical
table's DPP-4 row. Those successes do not eliminate the failures above.

## Japanese configuration audit and targeted reruns

The initial results should not be described as maximally tuned performance.
PaddleOCR is a toolkit containing different parsers: the initial PP-StructureV3
run is not a test of PaddleOCR-VL. Japanese was explicitly enabled, and the
installed PP-StructureV3 source maps `lang="japan"` to `PP-OCRv5_server_rec` in
both general and table OCR. This is a Japanese-capable shared recognizer, as
documented in the [official model table](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html).
It does not restrict decoding to Japanese codepoints.

MinerU 4.0's direct parser has no Japanese language flag. Its installed OCR code
maps the `japan` alias to the shared recognizer; this is upstream behavior, not a
missing flag in our command. The initial Standard tier is below Advanced in the
[official quality tiers](https://opendatalab.github.io/MinerU/usage/tiers/).

Nine targeted reruns used the same image-only inputs, retained the initial
outputs, and ran one GPU pipeline at a time:

| Configuration | Pages | Earlier failures fixed | Failures remaining or introduced |
| --- | --- | --- | --- |
| MinerU **Advanced**, explicit `--ocr-mode ocr` | Vertical newspaper, NTA form | Correct 26/G48 and 27/G49 associations; correct second label; headline precedes body | Body still says `ZC加工` for `NC加工`; `MITTR` for `MTTR` |
| **PaddleOCR-VL 1.6**, complete layout + VLM pipeline | All four selected pages | Correct 95/88/72 numbers; DPP-4 row present; all 67 annotated manual table cells match in text/row/spans | `統計値` → `統計值`; `NC加工` → `NO加工`; `DPP-4阻害薬` → `DPP-4阻帯薬`; NTA `繰り越す` → `操り越す` |
| PP-StructureV3 + **PP-OCRv6 Medium** detection and recognition | Manual, medical table | DPP-4 row recovered correctly; 65/67 manual table cell tuples match | `統計値` → `統計值`; two manual table cell tuples differ |
| PP-StructureV3 + **dedicated Japanese PP-OCRv3 recognizer**, PP-OCRv5 server detector | Manual | `統計値`, `一覧`, `単位` recovered without the tested Chinese variants | `MPa` → `山F`; other garbling/omissions; 36/67 manual table cell tuples match |

The [PaddleOCR-VL documentation](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
requires the complete layout-plus-recognition pipeline; that is what this rerun
uses, with `PP-DocLayoutV3`, native GPU inference, and chart recognition enabled.
The [PP-OCRv6 documentation](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
describes its shared Japanese support. Its model names were explicitly selected
for both general and table OCR. The dedicated Japanese alternative changes the
recognizer in both paths while retaining the detector, so the language model
choice is tested directly.

All Paddle reruns explicitly use `gpu:0`, retain every Markdown layout label
(`markdown_ignore_labels=[]`), and disable document rotation/unwarping on these
already upright clean pages. Text-line orientation remains enabled for
PP-StructureV3. Increasing every option is not automatically better for these
inputs, and this was not an exhaustive threshold/model/preprocessing search.

Evidence is in `fairness_audit/` under the data root: `audit.json`,
`paddle_profiles.py`, resolved YAML configs, logs, separate Markdown/JSON outputs,
per-page input/output hashes, and `observations.json`. Run `inspect_audit.py` with
the benchmark environment to regenerate the source-aligned probes and table
cell comparisons. Cell tuples include row index, rowspan, colspan and text with
whitespace removed; these counts are diagnostics, not CER/TEDS or a full error
count. The Japanese PP-OCRv3 and PP-OCRv6 weights were downloaded into the existing
Paddle model cache; no installed libraries were upgraded.

The fair conclusion is that stronger configurations fix several initial
failures, while confirmed character mistakes remain. Keep these profiles
separate in any ranking or local-LLM judge calibration.

## Evidence and reproduction

Each document has its reference in `input/<document>/original/page_0001.pdf` and
its candidate in `output/<document>/<pipeline>/chars_rem/page_0001_attempt1.md`.
`.run.json` files record versions, input/output hashes, commands and process times.
`ground_truth/` contains the two selected JASyn annotations.
`evidence/nta_rows_25_27.png` is a direct crop of the original form showing the
26/G48 and 27/G49 associations. `findings.json` records the evidence links and checks.

```bash
benchmark/.venv/bin/python benchmark/main.py --parse \
  --io-dir benchmark/datagen/io/japanese-stress-20261012 \
  --pipelines mineru paddle --variants chars_rem --attempts 1
benchmark/.venv/bin/python benchmark/datagen/io/japanese-stress-20261012/inspect_outputs.py
```

The first command resumes and skips matching successful outputs. The inspector
records extracted text lengths, table dimensions/spans and strict annotated-block
matches in `observations.json`. These diagnostics are **not CER or TEDS**; matching
row/span counts alone does not establish correct table content.

Versions: MinerU 4.0.12, PaddleOCR 3.7.0, PaddlePaddle GPU 3.3.1. MinerU used the
standard local model on the RTX 5060 Ti. Paddle also used that GPU. MinerU process
times were 209.4, 71.3, 60.0 and 64.1 seconds; Paddle took 26.4–37.4 seconds.
Processes reload models for each page, so these are not steady-state throughput
measurements. A persistent worker should be the next speed improvement.

## Japanese datasets worth using

| Dataset | Best use here | Ground truth / limitation |
| --- | --- | --- |
| [OmniDocBench-JASyn](https://huggingface.co/datasets/stockmark/OmniDocBench-JASyn) | Closest match for Japanese PDF-to-Markdown evaluation | 518 annotated synthetic images, 14 document types; text, HTML tables, formulas, reading order. Images only. Card says CC BY 4.0 and requests no model training/distillation. Use for evaluation. |
| [JSSODa-test](https://huggingface.co/datasets/llm-jp/JSSODa-test) | Isolate vertical/horizontal Japanese OCR and column-order failures | Synthetic text rendered in one to four columns; dedicated test split, CC BY 4.0. Not a complex-table benchmark. |
| [VJRODa](https://github.com/llm-jp/eval_vertical_ja#vjroda) | Real-world vertical Japanese pages | Companion real-PDF-page evaluation set; obtain from the authors' linked GitLab and retain its source terms. |
| [JDocQA](https://github.com/mizuumi/JDocQA) | Mine difficult real Japanese PDFs and test information recovery | 5,504 PDFs and 11,600 QAs with page/answer-region references. QA annotations, not full-page OCR/table ground truth; annotation license CC BY-SA 4.0. |
| [HakushoBench](https://huggingface.co/datasets/llm-jp/HakushoBench) | Hard government charts, tables and downstream QA | 2,053 image/QA pairs from 33 white papers. Metadata Apache 2.0; image rights/conditions are separate. Not full-page transcription ground truth. |
| [Synth-JDoc](https://github.com/llm-jp/synth-jdoc) | Synthetic generation methods for mixed vertical/horizontal columns | HTML/CSS synthesis, embedded figures and scan/noise scripts. Code Apache 2.0; check data's own terms separately. |

Start with JASyn for table/order failures and JSSODa-test/VJRODa for vertical text,
then add real forms and company documents. Keep failure-mining pages separate
from the held-out set used to claim accuracy improvements. Preserve Unicode
codepoints when measuring leakage, and keep text, cell associations/spans and
reading order as separate measurements. Rendered appearance alone can conceal
missing searchable text.

No official benchmark suite, repeated attempts, or LLM mistake-count scoring was
run for this discovery set. The examples above were checked against annotations
and/or the visible source; they do not establish population error rates.
