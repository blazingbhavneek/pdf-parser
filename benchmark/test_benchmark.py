"""Run: benchmark/.venv/bin/python -m unittest benchmark.test_benchmark -v"""

import argparse
import base64
import copy
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import time
import unittest
from unittest.mock import patch

import pymupdf
import numpy as np
from PIL import Image, ImageDraw

from benchmark import main as bench
from benchmark.datagen.augment import SPAN, VERSION, augment
from benchmark.datagen.render import MarkdownRenderer, pdf_image
from benchmark.datagen.premade import japanese_public_setup as public
from benchmark.datagen.premade import japanese_stress_setup as premade
from docx_converter import client as docx_client, server as docx_server


def judgment():
    return {"counts": {name: 0 for name in bench.COUNTS}, "reference_character_count": 20,
            "uncertain_character_count": 0, "confidence": 0.9,
            "evidence": [], "table_assessment": [], "notes": []}


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("raw", "input", "output", "score", "cache"):
            (self.root / name).mkdir()
        self.source = self.root / "raw" / "日本語.pdf"
        with pymupdf.open() as document:
            for text in ("日本語 駅 カタカナ ひらがな", "会社の業務報告書"):
                page = document.new_page(width=300, height=400)
                page.insert_text((20, 50), text, fontname="japan", fontsize=12)
            document.save(self.source)
        self.args = argparse.Namespace(dpi=100, seed=42, workers=1, judge_dpi=100,
                                       documents=None, pipelines=None,
                                       variants=list(bench.VARIANTS), redo=None, limit=None,
                                       export=None, top=200)
        self.config = {"LLM_BASE_URL": "http://unused/v1", "LLM_MODEL": "test"}
        self.document = self.root / "input" / "日本語"

    def prepare(self):
        return bench.prepare_document(self.root, self.source, self.document.name, self.args, {})

    def augmented(self):
        """Page PDFs in input/ that are neither the clean original nor the text-free raster."""
        return sorted(p for p in self.document.glob("page_*.pdf")
                      if bench.page_type(p.stem) not in ("original", "chars_rem"))

    def candidate(self, kind="original"):
        path = self.root / "output" / "mineru" / "日本語" / f"page_0001_{kind}_attempt1.md"
        bench.atomic_write(path, "# 日本語\n\n駅 カタカナ ひらがな\n")
        return path

    def test_prepare_preserves_original_and_removes_all_detectable_text(self):
        self.assertTrue(self.prepare())
        originals = sorted(self.document.glob("page_*_original.pdf"))
        rasters = sorted(self.document.glob("page_*_chars_rem.pdf"))
        self.assertEqual([p.name for p in originals], ["page_0001_original.pdf", "page_0002_original.pdf"])
        self.assertEqual([p.name for p in rasters], ["page_0001_chars_rem.pdf", "page_0002_chars_rem.pdf"])
        self.assertEqual(len(self.augmented()), 2)
        self.assertTrue(all(p.suffix == ".pdf" for p in self.document.iterdir()))
        for original, raster in zip(originals, rasters):
            with pymupdf.open(original) as document:
                self.assertEqual(len(document), 1)
                self.assertTrue(document[0].get_text().strip())
            self.assertEqual(pdf_image(original, 100).tobytes(), pdf_image(raster, 100).tobytes())
        for path in rasters + self.augmented():
            with pymupdf.open(path) as document:
                self.assertEqual(len(document), 1)
                self.assertEqual(document[0].get_text(), "")
                self.assertFalse(document[0].get_fonts())
                self.assertEqual(document[0].rect, pymupdf.Rect(0, 0, 300, 400))
        hint = bench.read_json(self.root / "cache" / "hint" / "日本語" / "page_0001.json")
        self.assertFalse(hint["verified"])
        self.assertIn("駅", hint["native_text"])

    def test_resume_repairs_missing_files_and_rejects_changed_sources(self):
        self.prepare()
        original = self.document / "page_0001_original.pdf"
        timestamp = original.stat().st_mtime_ns
        self.assertFalse(self.prepare())
        missing = self.document / "page_0001_aug.pdf"
        expected = pdf_image(missing, 100).tobytes()
        missing.unlink()
        self.assertTrue(self.prepare())
        self.assertEqual(original.stat().st_mtime_ns, timestamp)
        self.assertEqual(pdf_image(missing, 100).tobytes(), expected)
        self.assertFalse(self.prepare())
        recorded = bench.read_json(self.root / "cache" / "input" / "日本語.json")["pages"][0]["augmentations"]
        self.assertEqual(recorded["version"], VERSION)
        self.args.dpi = 150
        with self.assertRaisesRegex(ValueError, "settings changed"):
            self.prepare()
        self.args.dpi = 100
        with self.source.open("ab") as stream:
            stream.write(b"\n% source changed\n")
        with self.assertRaisesRegex(ValueError, "Source or preparation settings changed"):
            self.prepare()

    def test_partial_manifest_resumes_and_unowned_folder_is_rejected(self):
        self.prepare()
        path = self.root / "cache" / "input" / "日本語.json"
        manifest = bench.read_json(path)
        manifest["complete"] = False
        bench.write_json(path, manifest)
        self.assertTrue(self.prepare())
        path.unlink()
        with self.assertRaisesRegex(ValueError, "without a manifest"):
            self.prepare()

    def test_augmentation_is_seeded_and_keeps_size(self):
        image = Image.new("RGB", (300, 400), "white")
        ImageDraw.Draw(image).text((40, 40), "sample text", fill="black")
        first, settings = augment(image, 42)
        second, settings2 = augment(image, 42)
        self.assertEqual(first.size, image.size)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(settings, settings2)
        self.assertNotEqual(first.tobytes(), image.tobytes())
        self.assertTrue(settings["training_review_required"])
        # Every effect lands on the same page, each with its own recorded strength.
        self.assertEqual(set(settings["geometry"]), {"perspective", "skew", "curvature"})
        self.assertNotIn("resolution_scale", settings)  # Stage is disabled in this version.
        for key in ("rotation_degrees", "gaussian_blur_radius",
                    "motion_blur_pixels", "ink_strength", "ink_retention", "contrast",
                    "shadow_strength", "moire_strength", "noise_std", "jpeg_quality"):
            self.assertIn(key, settings)
        # Metadata must survive JSON round-tripping, including the curvature mesh.
        self.assertEqual(json.dumps(settings), json.dumps(json.loads(json.dumps(settings))))

    def test_stacked_knobs_stay_uniform_in_range_and_never_crop_the_page(self):
        image = Image.new("RGB", (300, 400), "white")
        pixel_scale = 300 / 2480  # Knob bounds are expressed for an A4 page at 300 DPI.
        knobs = {  # knob: (value, mildest, strongest); mild first so t=0 is always the gentle end
            "skew_degrees": (lambda s: abs(s["rotation_degrees"]), 0.5, 15.0),
            "motion_pixels": (lambda s: s["motion_blur_pixels"] / pixel_scale, 2.5, 14.0),
            "jpeg_quality": (lambda s: float(s["jpeg_quality"]), 86.0, 20.0),
            "ink_strength": (lambda s: s["ink_strength"], 0.25, 0.88),
            "moire_strength": (lambda s: s["moire_strength"], 4.0, 28.0),
            "shadow_strength": (lambda s: s["shadow_strength"], 0.14, 0.62),
            "noise_std": (lambda s: s["noise_std"], 1.0, 7.0),
            "curvature_percent": (lambda s: abs(s["geometry"]["curvature"]["amplitude_pixels"]) / 4,
                                  0.8, 7.5),
        }
        corners = np.array([[0, 0, 1], [300, 0, 1], [300, 400, 1], [0, 400, 1]])
        positions = {name: [] for name in knobs}
        for seed in range(80):
            _, settings = augment(image, seed)
            for geometry in settings["geometry"].values():
                if geometry["kind"] == "homography":
                    output = corners @ np.array(geometry["source_to_output"]).T
                    output = output[:, :2] / output[:, 2, None]
                    self.assertTrue(np.all(output >= 0))
                    self.assertTrue(np.all(output <= [300, 400]))
                else:
                    x = np.linspace(0, 300, 100)
                    mx, my = geometry["margin"]
                    sx, sy = geometry["scale"]
                    for y in (0, 400):
                        v = (my + sy * y
                             + geometry["amplitude_pixels"]
                             * np.sin(np.pi * geometry["sine_cycles"] * x / 300))
                        self.assertTrue(np.all((v >= 0) & (v <= 400)))
                    self.assertTrue(np.all((mx + sx * x >= 0) & (mx + sx * x <= 300)))
            page = []
            for name, (value, mild, strong) in knobs.items():
                t = (value(settings) - mild) / (strong - mild) / SPAN  # Of the used range.
                self.assertTrue(0 <= t <= 1, f"{name} out of range: {t}")
                positions[name].append(t)
                page.append(t)
            # SPAN caps the worst case, and a page must never draw every knob at the ceiling.
            self.assertLess(min(page), 0.95, f"seed {seed}")
        for name, t in positions.items():
            self.assertGreater(max(t), 0.8, f"{name} never reaches its strong end")
            # Unbiased: both the mean and the median sit in the middle of the used range.
            self.assertTrue(0.4 < statistics.fmean(t) < 0.6, f"{name} is not uniform: {statistics.fmean(t)}")
            self.assertTrue(0.35 < statistics.median(t) < 0.65, f"{name} is not uniform")

    def test_prepare_writes_one_stacked_page_and_rejects_other_versions(self):
        self.assertTrue(self.prepare())
        self.assertEqual([p.name for p in self.augmented()],
                         ["page_0001_aug.pdf", "page_0002_aug.pdf"])
        manifest_path = self.root / "cache" / "input" / "日本語.json"
        manifest = bench.read_json(manifest_path)
        self.assertEqual(manifest["settings"], {"dpi": 100, "seed": 42,
                                                "augmentation_version": VERSION})
        hashes = {name: bench.digest(self.document / name) for name in manifest["files"]}
        self.assertFalse(self.prepare())
        for name, expected in hashes.items():
            self.assertEqual(bench.digest(self.document / name), expected)
        manifest["settings"]["augmentation_version"] = VERSION - 1
        bench.write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "augmentation version"):
            self.prepare()

    def test_stacked_output_resolves_to_clean_reference(self):
        self.prepare()
        candidate = self.candidate("aug")
        sources = bench.score_sources(self.root, candidate)
        self.assertEqual(sources["profile"], "aug")
        self.assertEqual(sources["reference"], self.document / "page_0001_original.pdf")
        self.assertEqual(sources["input"], self.document / "page_0001_aug.pdf")
        self.assertTrue(sources["input"].is_file())

    def test_parallel_prepare_matches_serial_page_for_page(self):
        self.assertTrue(self.prepare())
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        parallel = Path(other.name)
        for name in ("raw", "input", "output", "score", "cache"):
            (parallel / name).mkdir()
        (parallel / "raw" / self.source.name).write_bytes(self.source.read_bytes())
        self.args.workers = 2
        source = parallel / "raw" / self.source.name
        self.assertTrue(bench.prepare_document(parallel, source, self.document.name, self.args, {}))
        self.assertEqual(bench.read_json(parallel / "cache" / "input" / f"{self.document.name}.json")["pages"],
                         bench.read_json(self.root / "cache" / "input" / f"{self.document.name}.json")["pages"])
        for path in sorted(self.document.glob("page_*.pdf")):
            twin = parallel / "input" / self.document.name / path.name
            self.assertEqual(pdf_image(twin, 100).tobytes(), pdf_image(path, 100).tobytes(), path.name)

    def test_premade_setup_downloads_validates_normalizes_and_preserves_existing_files(self):
        import hashlib
        catalog_dir = self.root / "catalog"
        catalog_dir.mkdir()
        stream = io.BytesIO()
        Image.new("RGB", (80, 100), "white").save(stream, format="PNG")
        image = stream.getvalue()
        annotations = json.dumps([{"page_info": {"image_path": "test.png"}, "layout_dets": []}]).encode()
        catalog = {"jasyn_revision_observed": "pinned-revision",
                   "jasyn_annotation_sha256": hashlib.sha256(annotations).hexdigest(),
                   "documents": {
                       "jasyn_test": {"image": "test.png", "ground_truth": "ground_truth/jasyn_test.json",
                                      "source_sha256": hashlib.sha256(image).hexdigest()},
                       "nta2025_tax_form": {"url": "https://example.test/tax.pdf", "pdf_page_1_based": 1,
                                            "source_sha256": bench.digest(self.source)}}}
        bench.write_json(catalog_dir / "japanese_stress_sources.json", catalog)
        files = {"OmniDocBench_JASyn.json": annotations, "test.png": image, "tax.pdf": self.source.read_bytes()}

        def response(request, **kwargs):
            return io.BytesIO(files[request.full_url.rsplit("/", 1)[-1]])

        target = self.root / "premade"
        with patch.object(premade, "HERE", catalog_dir), patch.object(premade, "urlopen", side_effect=response) as network:
            premade.setup(target)
            self.assertEqual(network.call_count, 3)
            for path in (target / "raw").glob("*.pdf"):
                with pymupdf.open(path) as pdf:
                    self.assertEqual(len(pdf), 1)
                    self.assertAlmostEqual(pdf[0].rect.width, 595.2756, places=3)
            self.assertTrue((target / "ground_truth/jasyn_test.json").is_file())
            preserved = {str(p): bench.digest(p) for p in target.rglob("*") if p.is_file()}
            premade.setup(target)
            self.assertEqual(network.call_count, 3)
            self.assertTrue(all(bench.digest(Path(p)) == value for p, value in preserved.items()))
            (target / "sources/test.png").write_bytes(b"corrupt cache")
            with self.assertRaisesRegex(ValueError, "Cached source changed"):
                premade.setup(target)

    def test_document_names_do_not_overwrite_same_stem(self):
        (self.root / "raw" / "日本語.docx").touch()
        names = [name for _, name in bench.source_documents(self.root)]
        self.assertEqual(set(names), {"日本語__pdf", "日本語__docx"})
        (self.root / "raw" / "日本語__pdf.pdf").touch()
        with self.assertRaisesRegex(ValueError, "collide"):
            bench.source_documents(self.root)

    def test_public_setup_fetches_pinned_hashes_and_keeps_existing_files(self):
        import hashlib
        catalog = self.root / "catalog"
        catalog.mkdir()
        pdf = self.source.read_bytes()
        documents = {"public_a.pdf": {"snapshot": "https://example.test/a.pdf",
                                      "sha256": hashlib.sha256(pdf).hexdigest()},
                     "public_bad.pdf": {"snapshot": "https://example.test/bad.pdf", "sha256": "0" * 64}}
        catalog_file = catalog / "japanese_public_sources.json"
        bench.write_json(catalog_file, {"documents": documents})
        with patch.object(public, "HERE", catalog), patch.object(
                public, "urlopen", side_effect=lambda request, **kwargs: io.BytesIO(pdf)) as network:
            with self.assertRaisesRegex(ValueError, "pinned hash"):
                public.setup(self.root)
            self.assertEqual((self.root / "raw" / "public_a.pdf").read_bytes(), pdf)
            self.assertFalse((self.root / "raw" / "public_bad.pdf").exists())
            self.assertEqual(network.call_count, 2)
            bench.write_json(catalog_file, {"documents": {"public_a.pdf": documents["public_a.pdf"]}})
            public.setup(self.root)
            self.assertEqual(network.call_count, 2)

    def test_existing_document_identity_survives_new_same_stem_format(self):
        self.prepare()
        (self.root / "raw" / "日本語.docx").touch()
        names = {source.suffix: name for source, name in bench.source_documents(self.root)}
        self.assertEqual(names[".pdf"], "日本語")
        self.assertEqual(names[".docx"], "日本語__docx")

    def fake_docx_conversion(self, pages=("日本語 駅 カタカナ ひらがな", "会社の業務報告書")):
        """Stand-in for the Word server: whole PDF plus one Pandoc-made DOCX per page."""
        def convert(source, folder, *args):
            (folder / "pages").mkdir(parents=True)
            (folder / "document.pdf").write_bytes(self.source.read_bytes())
            for number, text in enumerate(pages, 1):
                subprocess.run(["pandoc", "-f", "markdown", "-t", "docx", "-o",
                                str(folder / "pages" / f"page_{number:04d}.docx")], input=text, text=True, check=True)
            bench.write_json(folder / "pages.json", {"converter": "word", "pages": [
                {"page": number, "header": "", "footer": ""} for number in range(1, len(pages) + 1)]})
            return "word"
        return convert

    @unittest.skipUnless(shutil.which("pandoc"), "pandoc is not installed")
    def test_docx_goes_to_word_converter_and_pptx_to_libreoffice(self):
        source = self.root / "raw" / "日本語.docx"
        source.write_bytes(b"fake-office-file")
        target = self.root / "cache" / "convert" / "日本語.pdf"
        config = {"CONVERTER_URL": "http://laptop:8765/convert", "CONVERTER_API_KEY": "test-token"}
        with patch.object(bench, "convert_docx", side_effect=self.fake_docx_conversion()) as convert:
            self.assertEqual(bench.convert_office(source, target, config), "word")
            self.assertEqual(bench.convert_office(source, target, config), "word")  # Reuses the folder.
        self.assertEqual(convert.call_count, 1)
        self.assertEqual(convert.call_args.args[2:4], ("http://laptop:8765/convert", "test-token"))
        bench.valid_pdf(target.read_bytes())

        slides = self.root / "raw" / "日本語.pptx"
        slides.write_bytes(b"fake-office-file")

        def fake_office(command, log, timeout):
            folder = Path(command[command.index("--outdir") + 1])
            (folder / (slides.stem + ".pdf")).write_bytes(self.source.read_bytes())

        with patch.object(bench, "run_process", side_effect=fake_office):
            self.assertEqual(bench.convert_office(slides, self.root / "slides.pdf", config), "libreoffice")

    @unittest.skipUnless(shutil.which("pandoc"), "pandoc is not installed")
    def test_prepare_converts_docx_and_writes_gold_markdown_per_page(self):
        source = self.root / "raw" / "報告書.docx"
        source.write_bytes(b"fake-office-file")
        with patch.object(bench, "convert_docx", side_effect=self.fake_docx_conversion()):
            self.assertTrue(bench.prepare_document(self.root, source, "報告書", self.args, {}))
        self.assertEqual(sorted(p.name for p in (self.root / "input" / "報告書").iterdir()),
                         [f"page_{page:04d}_{kind}.pdf" for page in (1, 2)
                          for kind in ("aug", "chars_rem", "original")])
        manifest = bench.read_json(self.root / "cache" / "input" / "報告書.json")
        self.assertEqual(manifest["converter"], "word")
        gold = self.root / "gold" / "報告書"
        self.assertEqual((gold / "page_0002.md").read_text(encoding="utf-8"), "会社の業務報告書\n")
        self.assertEqual([row["agreement"] for row in bench.read_json(gold / "pages.json")], [1.0, 1.0])
        self.assertFalse(bench.prepare_document(self.root, source, "報告書", self.args, {}))
        self.assertLess(bench.agreement("| 統計値 |", "統計值"), 1)
        self.assertEqual(bench.agreement("本文", "社外秘 本文", furniture="社外秘"), 1.0)

    def test_gold_score_counts_letters_leaks_headers_and_tables(self):
        from benchmark import gold

        gold_page = ("# 第7条 統計値の確認\n\n売上は10%増えた。\n\n"
                     "<table><tr><th>項目</th><th>金額</th></tr><tr><td>売上</td><td>10円</td></tr>"
                     "<tr><td>利益</td><td>3円</td></tr></table>\n")
        predicted = ("社外秘\n\n## 第８条 統計值の確認\n\n売上は１１％増えた。\n\n"
                     "| 項目 | 金額 |\n|---|---|\n| 利益 | 3円 |\n")
        record = gold.score_page(gold_page, predicted, ["社外秘", ""], gold.japanese())
        # Only kana/kanji/Latin count (tables too): 7→８, 10→１１ and %→％ are ignored; 値→值 is one edit
        # and one leak, the missing 売上 | 10円 row three more.
        self.assertEqual((record["gold_letters"], record["edits"]), (24, 4))
        self.assertEqual(record["leaks"], ["值"])
        self.assertEqual(record["substitutions"], [["値", "值"]])
        self.assertEqual(record["runs_found"], record["runs"] - 1)
        table = record["tables"][0]
        self.assertLess(table["teds"], 1)
        self.assertEqual((table["cell_edits"], table["cell_chars"]), (5, 13))  # Only the missing 売上 | 10円 row.

    def test_word_server_round_trip_requires_token(self):
        received = []

        def fake_word(source, output):
            received.append(source.read_bytes())
            (output / "pages").mkdir(parents=True)
            (output / "document.pdf").write_bytes(self.source.read_bytes())
            (output / "pages" / "page_0001.docx").write_bytes(b"page one")
            (output / "pages.json").write_text('{"converter": "word", "pages": []}')

        server = docx_server.HTTPServer(("127.0.0.1", 0), docx_server.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        with patch.object(docx_server.Handler, "token", "secret"), \
                patch.object(docx_server.Handler, "convert", staticmethod(fake_word)):
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_address[1]}/convert"
                upload = self.root / "日本語.docx"
                upload.write_bytes(b"PK\x03\x04 binary \r\n--boundary-like\r\n bytes")
                folder = self.root / "converted"
                self.assertEqual(docx_client.convert(upload, folder, url, "secret"), "word")
                self.assertEqual(received, [upload.read_bytes()])
                self.assertEqual(sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.*")),
                                 ["document.pdf", "pages.json", "pages/page_0001.docx"])
                with self.assertRaises(HTTPError) as denied:
                    urlopen(Request(url, b"x", {"Authorization": "Bearer wrong"}), timeout=5)
                self.assertEqual(denied.exception.code, 401)
            finally:
                server.shutdown()
                server.server_close()

    @unittest.skipUnless(shutil.which("pandoc") and shutil.which("soffice") and subprocess.run(
        ["/usr/bin/python3", "-c", "import uno"], capture_output=True).returncode == 0,
        "needs pandoc, LibreOffice and a Python with uno")
    def test_libreoffice_fallback_splits_real_pages_with_continued_numbering(self):
        lines = ["# 社内規程", ""] + [f"{i}. 第{i}項目の本文です。会社の業務報告書と統計値を確認します。\n"
                                       for i in range(1, 41)]
        lines += ["| 項目 | 金額 |", "|---|---|"] + [f"| 売上高{i} | {i * 100}円 |" for i in range(30)]
        source = self.root / "規程.docx"
        subprocess.run(["pandoc", "-f", "markdown", "-o", str(source)], input="\n".join(lines),
                       text=True, check=True)
        folder = self.root / "cache" / "convert" / "規程"
        self.assertEqual(docx_client.convert(source, folder), "libreoffice")
        bench.write_gold(self.root, "規程", folder)
        rows = bench.read_json(self.root / "gold" / "規程" / "pages.json")
        self.assertGreater(len(rows), 2)
        # The last page loses only the table header row LibreOffice repeats there.
        self.assertTrue(all(row["agreement"] > 0.95 for row in rows), rows)
        second = (self.root / "gold" / "規程" / "page_0002.md").read_text(encoding="utf-8")
        self.assertNotRegex(second.lstrip(), r"^1\\?\. ")  # List numbers continue, not restart.

    def test_pipeline_isolated_attempts_resume_and_retain_assets(self):
        self.prepare()
        worker = self.root / "fake worker.py"
        worker.write_text("import pathlib,sys\n"
                          "out=pathlib.Path(sys.argv[2])\n"
                          "(out/'result.md').write_text('# 駅\\n',encoding='utf-8')\n"
                          "(out/'result_nohf.md').write_text('ignore',encoding='utf-8')\n", encoding="utf-8")
        config = {"MINERU_VENV": str(Path(sys.executable).parent.parent),
                  "MINERU_COMMAND": "{python} " + shlex.quote(str(worker)) + " {input} {output}"}
        setup = bench.pipeline_setup("mineru", config, self.root)
        source = self.document / "page_0001_original.pdf"
        output = self.root / "output" / "mineru" / "日本語"
        scratch = self.root / "cache" / "parse" / "mineru" / "日本語"
        self.assertTrue(bench.parse_page(source, self.root, "mineru", "日本語", setup, 1, config))
        self.assertFalse(bench.parse_page(source, self.root, "mineru", "日本語", setup, 1, config))
        self.assertTrue(bench.parse_page(source, self.root, "mineru", "日本語", setup, 2, config))
        self.assertEqual(sorted(p.name for p in output.iterdir()),
                         ["page_0001_original_attempt1.md", "page_0001_original_attempt2.md"])
        marker = bench.read_json(scratch / "page_0001_original_attempt1.run.json")
        self.assertIn("versions", marker)
        self.assertTrue((scratch / marker["artifact_markdown"]).is_file())
        self.assertTrue((scratch / marker["artifact_root"] / "run.log").is_file())
        self.assertEqual((output / "page_0001_original_attempt1.md").read_text(), "# 駅\n")
        env = bench.isolated_environment(setup[0])
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["VIRTUAL_ENV"], str(setup[0]))

    def test_failed_pipeline_does_not_publish_markdown(self):
        self.prepare()
        venv = Path(sys.executable).parent.parent
        setup = (venv, ["{python}", "-c", "raise SystemExit(1)"], {})
        source = self.document / "page_0001_original.pdf"
        output = self.root / "output" / "mineru" / "日本語"
        scratch = self.root / "cache" / "parse" / "mineru" / "日本語"
        with self.assertRaises(RuntimeError):
            bench.parse_page(source, self.root, "mineru", "日本語", setup, 1, {})
        self.assertFalse((output / "page_0001_original_attempt1.md").exists())
        self.assertEqual(len(list(scratch.glob("**/run.log"))), 1)
        self.assertFalse((scratch / "page_0001_original_attempt1.run.json").exists())

    def test_mineru_four_uses_direct_inference_without_a_server(self):
        venv = Path(sys.executable).parent.parent
        with patch.object(bench.subprocess, "check_output", return_value=b'{"mineru":"4.0.12"}'):
            setup = bench.pipeline_setup("mineru", {"MINERU_VENV": str(venv)}, self.root)
        command = bench.pipeline_command(setup, self.source, self.root / "results", 1)
        self.assertEqual(command[0], str(venv / "bin" / "mineru-kit"))
        self.assertEqual(command[1], "parse")

    def test_score_arithmetic_does_not_double_count_leakage(self):
        result = judgment()
        result["counts"].update(character_substitutions=3, character_deletions=2,
                                character_insertions=1, chinese_leakage_substitutions=2,
                                table_structure_errors=4, layout_errors=1)
        self.assertEqual(bench.validate_judgment(result)["score"], 11)
        self.assertEqual(bench.validate_judgment(result)["estimated_cer"], 0.3)
        for invalid in (True, -1, 1.2, "2"):
            broken = copy.deepcopy(result)
            broken["counts"]["character_deletions"] = invalid
            with self.assertRaises(ValueError):
                bench.validate_judgment(broken)
        result["counts"]["chinese_leakage_substitutions"] = 4
        with self.assertRaisesRegex(ValueError, "subset"):
            bench.validate_judgment(result)

    def test_chat_does_not_set_a_completion_token_limit(self):
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
        config = {"LLM_BASE_URL": "http://unused/v1", "LLM_MODEL": "test",
                  "LLM_MAX_TOKENS": "1"}
        with patch.object(bench, "request_bytes", return_value=json.dumps(response).encode()) as request:
            bench.chat(config, [])
        payload = json.loads(request.call_args.args[1])
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("max_tokens", bench.judge_settings(config, 180))

    def test_score_uses_four_concurrent_judgments(self):
        self.prepare()
        self.args.pipelines = list(bench.PIPELINES)
        self.args.variants = ["original"]
        for pipeline in bench.PIPELINES:
            path = self.root / "output" / pipeline / "日本語" / "page_0001_original_attempt1.md"
            bench.atomic_write(path, "# 日本語\n\n駅 カタカナ ひらがな\n")
        active, maximum = 0, 0
        lock = threading.Lock()

        def fake_chat(config, content):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            result = judgment()
            return result, {"choices": [{"message": {"content": json.dumps(result)}}]}

        class FakeRenderer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def render(self, content, asset_base, asset_root, width, height, dpi, destination):
                Image.new("RGB", (100, 100), "white").save(destination.with_suffix(".png"))
                return {"fit_scale": 1.0}

        with patch.object(bench, "chat", fake_chat), patch.object(bench, "MarkdownRenderer", FakeRenderer):
            self.assertEqual(bench.score_all(self.root, self.args, self.config), 0)
        self.assertEqual(maximum, bench.SCORE_CONCURRENCY)

    def test_score_always_uses_original_resume_redo_and_content_invalidation(self):
        self.prepare()
        markdown = self.candidate("aug")
        sent = []

        def fake_chat(config, content):
            sent.append(content)
            result = judgment()
            return result, {"choices": [{"message": {"content": json.dumps(result)}}]}

        class FakeRenderer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def render(self, content, asset_base, asset_root, width, height, dpi, destination):
                Image.new("RGB", (100, 100), "white").save(destination.with_suffix(".png"))
                return {"fit_scale": 1.0}

        with patch.object(bench, "chat", fake_chat), patch.object(bench, "MarkdownRenderer", FakeRenderer):
            self.assertEqual(bench.score_all(self.root, self.args, self.config), 0)
            reference = next(item for item in sent[0] if item["type"] == "image_url")
            image_data = base64.b64decode(reference["image_url"]["url"].split(",", 1)[1])
            with Image.open(io.BytesIO(image_data)) as received:
                expected = pdf_image(self.document / "page_0001_original.pdf", 100)
                self.assertEqual(received.tobytes(), expected.tobytes())
            self.assertEqual(bench.score_all(self.root, self.args, self.config), 0)
            self.assertEqual(len(sent), 1)
            self.args.redo = str(markdown)
            bench.score_all(self.root, self.args, self.config)
            self.assertEqual(len(sent), 2)
            self.args.redo = None
            bench.atomic_write(markdown, "changed text")
            bench.score_all(self.root, self.args, self.config)
            self.assertEqual(len(sent), 3)
            with patch.object(bench, "chat", side_effect=ValueError("invalid model JSON")):
                self.args.redo = "*"
                self.assertEqual(bench.score_all(self.root, self.args, self.config), 1)
            self.args.redo = None
            bench.score_all(self.root, self.args, self.config)
            self.assertEqual(len(sent), 4)

    def test_rank_groups_variants_into_unique_original_pages(self):
        self.prepare()
        first = self.candidate()
        second = self.candidate("chars_rem")
        settings = bench.judge_settings(self.config, self.args.judge_dpi)
        records = {}
        for markdown, count in ((first, 10), (second, 20)):
            sources = bench.score_sources(self.root, markdown)
            key = str(markdown.relative_to(self.root / "output"))
            records[key] = {**{key: sources[key] for key in ("document", "page", "pipeline", "variant", "profile")},
                            "status": "complete", "score": count,
                            "fingerprint": bench.fingerprint(markdown, sources, settings)}
        bench.write_json(self.root / "score" / "scores.json", {"records": records})
        bench.rank_pages(self.root, self.args, self.config)
        pages = bench.read_json(self.root / "score" / "ranking.json")["pages"]
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["mean_mistakes"], 15)
        bench.atomic_write(second, "changed")
        bench.rank_pages(self.root, self.args, self.config)
        ranked = bench.read_json(self.root / "score" / "ranking.json")
        self.assertEqual(ranked["stale_scores_ignored"], 1)
        self.assertEqual(ranked["pages"][0]["mean_mistakes"], 10)

    def test_renderer_preserves_spans_and_fits_long_content_without_clipping(self):
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(bench.HERE / ".browsers"))
        assets = self.root / "assets"
        base = assets / "nested"
        base.mkdir(parents=True)
        Image.new("RGB", (30, 30), "red").save(assets / "local.png")
        markdown = ("# 日本語 駅\n\n![figure](../local.png)\n\n"
                    '<table><tr><th rowspan="2">部門</th><th colspan="2">売上</th></tr>'
                    '<tr><td>前年</td><td>今年</td></tr></table>\n\n' + "説明文\n\n" * 100)
        with MarkdownRenderer() as renderer:
            metrics = renderer.render(markdown, base, assets, 300, 400, 100, self.root / "render")
            self.assertLess(metrics["fit_scale"], 1)
            with pymupdf.open(self.root / "render.pdf") as document:
                self.assertEqual(len(document), 1)
                self.assertEqual(document[0].rect, pymupdf.Rect(0, 0, 300, 400))
            with self.assertRaisesRegex(ValueError, "Unrenderable"):
                renderer.render("![external](https://example.com/private.png)", base, assets,
                                300, 400, 100, self.root / "external")
            with self.assertRaisesRegex(ValueError, "Unrenderable"):
                renderer.render("![escape](../../outside.png)", base, assets,
                                300, 400, 100, self.root / "escape")


if __name__ == "__main__":
    unittest.main()
