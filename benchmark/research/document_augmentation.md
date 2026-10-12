# Document augmentation for Japanese PDF-to-Markdown

Research reviewed on 2026-10-12. This is an engineering policy for this benchmark,
not a measured claim that one augmentation recipe maximizes Japanese fine-tuning
accuracy. That requires training ablations and a held-out Japanese test set.

## What the primary sources support

| Source | Relevant evidence | How we use it |
| --- | --- | --- |
| [Augraphy paper](https://arxiv.org/abs/2208.14558) and [official code](https://github.com/sparkfish/augraphy) | Document-specific print/scan defects include ink, paper and post-processing effects. | Separate ink defects from scan/camera conditions; use bounded combinations instead of arbitrary photographic effects. |
| [Real5-OmniDocBench paper](https://arxiv.org/abs/2603.04205) and [methods](https://arxiv.org/html/2603.04205v1) | Physically recaptured documents cover scanning, warping, screen photography, illumination and skew. Complete semantic correspondence and human quality checks are required; irrecoverably blurred captures are recaptured. | Cover multiple capture conditions, preserve content, and review legibility. Our synthetic effects are approximations, not the paper's physical acquisition process. |
| [Synth-JDoc](https://github.com/llm-jp/synth-jdoc) and its [scan augmentation code](https://github.com/llm-jp/synth-jdoc/blob/main/src/synth/add_noise_scan.py) | Japanese synthetic documents include mixed writing directions and layouts. Its scan simulation records paper texture, rotation, perspective, lighting and blur parameters. | Preserve Japanese layout diversity, record parameters, and treat capture noise separately from layout generation. |
| [Nougat's official training transforms](https://github.com/facebookresearch/nougat/blob/main/nougat/transforms.py) | Document-to-markup training uses probabilistic morphology, affine/geometric distortions, contrast, compression, noise and blur. Its evaluation transform omits these augmentations. | Augment training inputs without turning the clean evaluation set into a moving target. This is precedent, not proof of optimal settings for Japanese business documents. |

The useful objective is **more realistic, recoverable failures**, not simply more
wrong characters. An unreadable page can produce many errors while supplying poor
supervision for exact transcription. Keep hard stress cases, but separate them
from examples approved for training.

## Implemented capture grid

Every PDF, DOCX or PPTX uses the same preparation flow after conversion to PDF.
The original page is immutable. A clean lossless raster copy forces OCR. Each
augmentation starts independently from the clean original raster, rather than
repeatedly degrading another augmentation.

Eight profiles are crossed with three severities. Isolated `resolution`,
`compression`, `motion`, `ink`, and `warp` variants help attribute failures.
`scan`, `photo`, and `screen` combine defects that co-occur during capture.

| Condition | Mild / medium / hard settings before random sampling |
| --- | --- |
| Resolution in scan/photo/screen/resolution | Scale 0.72–0.88 / 0.45–0.62 / 0.25–0.38, then restore raster dimensions |
| JPEG in scan/photo/screen/compression | Quality 78–90 / 48–65 / 22–38, with chroma subsampling |
| Scan skew | Absolute angle 0.2–0.7° / 1–2° / 2.5–4°, fitted inside the page |
| Motion blur | Length 1.5–2.5 / 4–6 / 8–12 pixels at the reference raster size; sampled direction |
| Photo/screen perspective | Inset envelope 1–2.5% / 3–6% / 7–11%; each corner samples 50–100% of that envelope |
| Photo shadow | Maximum attenuation 8–16% / 22–35% / 40–55%, sampled direction |
| Ink | Partial thinning/spreading blend 0.15–0.30 / 0.40–0.55 / 0.65–0.80, plus fading |
| Curvature | Baseline amplitude 0.5–1.2% / 1.8–3% / 4–6.5% of page height, fitted inside the canvas |
| Screen interference | Sinusoidal intensity amplitude 2–5 / 7–12 / 16–24 on an 8-bit scale |

Blur lengths are scaled by raster short-edge length / 2480. This is a practical
A4-at-300-DPI reference, not physical camera calibration. Noise and compression
parameters are in the sidecar JSON. Screen interference and curvature are simple
simulations; validate against actual company scans/photos before tuning their
sampling frequency. No new augmentation dependency is needed.

There is no deliberate text erasure, crop, mirror, fake watermark or opaque
occlusion. These can invalidate an unchanged transcription target. Resampling,
blur and thinning can nevertheless make small glyphs unreadable. All new variants
are marked `training_review_required: true`; this is a review requirement, not an
automatic image-quality classifier.

## Training policy for our dataset

1. **Split original documents first.** All pages and augmented siblings of a
   document belong to the same split. Group repeated forms, near duplicates and
   template families together where possible. Randomly splitting augmented pages
   would put almost the same material in training and testing.
2. **Build verified targets.** Correct the original transcription, reading order,
   table cells, rowspans and colspans. Parser outputs and native PDF extraction are
   proposals; local-LLM judgments are error-mining signals. None is automatically
   ground truth. Do not train on the erroneous extraction we are trying to fix.
3. **Review degraded inputs against their target.** Approve an augmented example
   only if a reader can still recover the target content. Reject or separately
   label unreadable text. For box/region supervision, transform annotations using
   the saved homography or mesh; untransformed original coordinates are wrong.
   This benchmark records transforms but does not export training annotations.
4. **Keep clean and mild examples prominent.** Start with a mixture of clean,
   mild and medium inputs and a smaller reviewed hard subset. Select the final
   mixture through ablations, not a claimed universal ratio. Keep a clean
   validation set and a separate fixed stress set; also evaluate real scans/photos.
5. **Mine diverse failures.** Prefer the weakest degradation that exposes a
   mistake. Balance recognition, reading-order, layout and table-span failures,
   and cap siblings from the same page/template. Do not select solely by raw
   error count: long pages and severely damaged pages naturally score worse.
6. **Preserve Japanese evidence.** Include vertical/horizontal mixtures, ruby,
   dense kanji, kana, small punctuation, seals/signatures already in the source,
   rotated labels, financial forms and irregular merged cells. Preserve source
   codepoints; do not normalize away width/compatibility distinctions or blacklist
   shared Han characters as Chinese. Review actual substituted characters.
7. **Measure the fine-tuning result.** Compare clean-only, clean plus mild/medium,
   and clean plus reviewed hard examples, keeping source pages, training budget
   and held-out tests comparable. Report verified character errors, reading order,
   table content and table structure separately. LLM estimates alone cannot
   establish a fine-tuning gain.

Pixel degradation cannot create new table topologies, vocabulary or layout
diversity. Add genuinely varied company documents or separately generated native
layouts for those gaps. Upsampling an old low-resolution scan also cannot recover
missing detail: record its provenance and inspect the original pixels.

## Running and resuming

```bash
# Full stress grid: 24 augmented inputs plus 2 baselines per source page.
benchmark/.venv/bin/python benchmark/main.py --prepare

# Smaller starting set; later commands append missing pairs without deleting any.
benchmark/.venv/bin/python benchmark/main.py --prepare \
  --augmentations scan photo ink warp --severities mild medium

# Parse with the selected Japanese-capable configurations in benchmark/.env.
benchmark/.venv/bin/python benchmark/main.py --parse \
  --pipelines mineru paddle --variants chars_rem augmented --attempts 1
```

The fixed seed makes the benchmark repeatable. Severity siblings share the same
profile seed; the actual downstream error rate can still be nonmonotonic. Changing
the seed or raster DPI requires a fresh IO directory or renamed raw source, so
existing parser results remain attributable to their exact input.

Keep the settings and package versions with any exported dataset. Increasing the
grid size increases disk use and parser work; 24 siblings do not count as 24
independent documents. Compare matched coverage, retain per-condition results,
and use the original page as the common scoring reference.
