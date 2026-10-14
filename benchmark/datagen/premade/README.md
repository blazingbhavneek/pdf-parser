# Premade benchmark inputs

Every premade setup writes PDFs into the one shared corpus, `benchmark/datagen/io/raw/`.
Setups only add public documents; the `private_…` files are company data and are never
touched. Existing files are kept in both scripts.
Run the setups from the repository root with the benchmark environment, then use
`benchmark/main.py` for preparation, parsing, scoring and ranking.

```bash
# 20 archived Japanese public-sector PDFs, pinned by snapshot and SHA-256.
benchmark/.venv/bin/python benchmark/datagen/premade/japanese_public_setup.py

# Four-page Japanese stress corpus (its sources/annotations are cached under premade/cache/).
benchmark/.venv/bin/python benchmark/datagen/premade/japanese_stress_setup.py

benchmark/.venv/bin/python benchmark/main.py --prepare
benchmark/.venv/bin/python benchmark/main.py --parse --pipelines mineru paddle --attempts 1
```

`japanese_public_setup.py` reads `japanese_public_sources.json`: Japanese
public-sector documents (ministerial reports, agency pamphlets, a prefectural
election notice, a scanned animal-quarantine handbook) discovered through the
Internet Archive CDX index, one or two per publisher. Each entry pins the origin
URL, the archived snapshot, the capture date, page count, whether the file has a
text layer, its vertical-writing block count, SHA-256 and size. Existing files are
never overwritten and a hash mismatch stops the run instead of replacing a file.

`japanese_stress_setup.py` carries the current selected pages: two OmniDocBench-JASyn
images, Japanese Circulation Society guideline page 31, and the NTA corporate tax
form. It pins the JASyn revision, validates source SHA-256 hashes, keeps the source
files under `premade/cache/japanese-stress/`, embeds images into A4 PDFs, splits the
guideline page, and fits the unusually large NTA media box to A4, then copies the
finished PDFs into the shared raw corpus. Existing raw PDFs, annotations, manifests
and outputs are preserved. Changed cached or downloaded sources fail instead of being
replaced. Use `--io-dir` for a separate corpus and `--no-publish` to skip the copy.

This is the current four-page discovery corpus plus the archived public documents,
not the full official benchmark suites. JASyn's dataset card requests evaluation only,
without training/distillation; keep those pages separate from company fine-tuning
data. The public documents stay subject to their publisher's copyright policy
(政府著作権基本方針; many ministries publish under CC BY 4.0) — see `license_note`,
and `japanese_stress_sources.json` for the stress sources' URLs, terms and hashes.
