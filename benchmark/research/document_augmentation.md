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
supervision for exact transcription. Keep the strongest samples, but separate them
from examples approved for training.

## Implemented capture grid

Every PDF, DOCX or PPTX uses the same preparation flow after conversion to PDF.
The original page is immutable. A clean lossless raster copy forces OCR. Each
augmentation starts independently from the clean original raster, rather than
repeatedly degrading another augmentation.

One stacked output per page: every effect is applied to the same image, matching how a
real capture degrades everything at once. The trade-off is attribution — a failure on a
stacked page cannot be pinned to one defect, so use the recorded per-knob parameters
when a specific condition has to be isolated again.

There are no profiles and no severity levels. Every knob has one range from the mildest
useful value to the strongest still-recoverable value, and each page draws one value per
knob independently and uniformly (`BIAS = 1`), so no strength is favored within the used
range.

Only `SPAN = 0.5` of each range is sampled, so the strong end below is the nominal
ceiling rather than a reachable level; halve these numbers to get the real maximum, which
also caps how badly one page can stack up.

| Condition | Nominal range mild → strong, sampled over the mild half |
| --- | --- |
| Resolution | Scale 0.80–0.22 — disabled in version 6, too destructive inside the stack |
| JPEG | Quality 86–20, chroma subsampling 0, 1 or 2 |
| Scan skew | Absolute angle 0.5–15°, fitted inside the page |
| Defocus blur | Radius 0.4–2.8 at the reference raster size |
| Motion blur | Length 2.5–14 pixels at the reference raster size; sampled direction |
| Perspective | Inset envelope 1.5–15%; each corner samples 50–100% of that envelope |
| Shadow | Maximum attenuation 14–62%, sampled direction |
| Ink | Partial thinning/spreading blend 0.25–0.88, toner retention 0.60–0.95 |
| Curvature | Baseline amplitude 0.8–7.5% of page height, 1–2 sine cycles, fitted inside the canvas |
| Screen interference | Sinusoidal intensity amplitude 4–28 on an 8-bit scale |
| Contrast | Factor 0.96–0.60 |
| Sensor noise | Gaussian standard deviation 1.0–7.0 on an 8-bit scale |

Blur lengths are scaled by raster short-edge length / 2480. This is a practical
A4-at-300-DPI reference, not physical camera calibration. Noise and compression
parameters are in the prepare manifest, `cache/input/<document>.json`. Screen interference and curvature are simple
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
4. **Keep clean and mild examples prominent.** Start with clean inputs plus the
   mildest half of each range, and add a smaller reviewed subset of the strongest
   samples. Select the final
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
7. **Measure the fine-tuning result.** Compare clean-only, clean plus low-half
   samples, and clean plus reviewed high-end samples, keeping source pages, training budget
   and held-out tests comparable. Report verified character errors, reading order,
   table content and table structure separately. LLM estimates alone cannot
   establish a fine-tuning gain.

Pixel degradation cannot create new table topologies, vocabulary or layout
diversity. Add genuinely varied company documents or separately generated native
layouts for those gaps. Upsampling an old low-resolution scan also cannot recover
missing detail: record its provenance and inspect the original pixels.

## Running and resuming

```bash
# Every raw document: 1 stacked augmented page plus 2 baselines per source page.
benchmark/.venv/bin/python benchmark/main.py --prepare

# Selected documents only.
benchmark/.venv/bin/python benchmark/main.py --prepare --documents "report"

# Parse with the selected Japanese-capable configurations in benchmark/.env.
benchmark/.venv/bin/python benchmark/main.py --parse \
  --pipelines mineru paddle --variants chars_rem augmented --attempts 1
```

The fixed seed makes the benchmark repeatable: a page redraws the same strengths every
run; the actual downstream error rate can still be nonmonotonic. Changing
the seed or raster DPI requires a fresh IO directory or renamed raw source, so
existing parser results remain attributable to their exact input.

Keep the settings and package versions with any exported dataset. Increasing the
grid size increases disk use and parser work; siblings of one page do not count as
independent documents. Compare matched coverage, retain per-condition results,
and use the original page as the common scoring reference.
