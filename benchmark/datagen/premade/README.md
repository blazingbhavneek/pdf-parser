# Premade benchmark inputs

Run setup scripts with the benchmark environment from the repository root.
Each benchmark has a `<benchmark_name>_setup.py` and writes normal `raw/` PDFs,
source downloads/provenance and any available annotations into its IO directory.
Use `benchmark/main.py` for preparation, parsing, scoring and ranking afterward.

```bash
# Reuse the current four-page Japanese stress corpus and all its existing results.
benchmark/.venv/bin/python benchmark/datagen/premade/japanese_stress_setup.py

# Or create a separate copy for another experiment.
benchmark/.venv/bin/python benchmark/datagen/premade/japanese_stress_setup.py \
  --io-dir benchmark/datagen/io/my-japanese-test
benchmark/.venv/bin/python benchmark/main.py --prepare \
  --io-dir benchmark/datagen/io/my-japanese-test
benchmark/.venv/bin/python benchmark/main.py --parse \
  --io-dir benchmark/datagen/io/my-japanese-test --pipelines mineru paddle --attempts 1
```

`japanese_stress_setup.py` carries the current selected pages: two OmniDocBench-JASyn
images, Japanese Circulation Society guideline page 31, and the NTA corporate tax
form. It pins the JASyn revision, validates source SHA-256 hashes, keeps the source
files, embeds images into A4 PDFs, splits the guideline page, and fits the unusually
large NTA media box to A4. Existing raw PDFs, annotations, manifests and outputs
are preserved. Changed cached/downloaded sources fail instead of being replaced.

This is the current four-page discovery corpus, not the full official benchmark
suites. JASyn's dataset card requests evaluation only, without training/distillation;
keep these pages separate from company fine-tuning data. Source URLs, terms notes,
selection reasons and hashes are in `japanese_stress_sources.json`.
