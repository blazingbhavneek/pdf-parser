"""Run: benchmark/.venv/bin/python -m unittest benchmark.test_benchmark -v"""

import argparse
import base64
import copy
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import pymupdf
import numpy as np
from PIL import Image, ImageDraw

from benchmark import main as bench
from benchmark.datagen.augment import LEGACY_PROFILES, PROFILES, SEVERITIES, augment, page_seed
from benchmark.datagen.render import MarkdownRenderer, image_pdf, pdf_image
from benchmark.datagen.premade import japanese_stress_setup as premade


def judgment():
    return {"counts": {name: 0 for name in bench.COUNTS}, "reference_character_count": 20,
            "uncertain_character_count": 0, "confidence": 0.9,
            "evidence": [], "table_assessment": [], "notes": []}


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("raw", "input", "output"):
            (self.root / name).mkdir()
        self.source = self.root / "raw" / "日本語.pdf"
        with pymupdf.open() as document:
            for text in ("日本語 駅 カタカナ ひらがな", "会社の業務報告書"):
                page = document.new_page(width=300, height=400)
                page.insert_text((20, 50), text, fontname="japan", fontsize=12)
            document.save(self.source)
        self.args = argparse.Namespace(dpi=100, seed=42, augmentations=list(LEGACY_PROFILES),
                                       severities=["medium"],
                                       judge_dpi=100, documents=None, pipelines=None,
                                       variants=list(bench.VARIANTS), redo=None, limit=None,
                                       export=None, top=200)
        self.config = {"LLM_BASE_URL": "http://unused/v1", "LLM_MODEL": "test"}
        self.document = self.root / "input" / "日本語"

    def prepare(self):
        return bench.prepare_document(self.source, self.document, self.args, {})

    def candidate(self, variant="original", profile=None):
        folder = self.root / "output" / "日本語" / "mineru" / variant
        stem = "page_0001" + ("_" + profile if profile else "")
        path = folder / f"{stem}_attempt1.md"
        bench.atomic_write(path, "# 日本語\n\n駅 カタカナ ひらがな\n")
        return path

    def test_prepare_preserves_original_and_removes_all_detectable_text(self):
        self.args.augmentations = list(PROFILES)
        self.args.severities = list(SEVERITIES)
        self.assertTrue(self.prepare())
        self.assertEqual(len(list(self.document.glob("original/*.pdf"))), 2)
        self.assertEqual(len(list(self.document.glob("chars_rem/*.pdf"))), 2)
        self.assertEqual(len(list(self.document.glob("augmented/*.pdf"))), 48)
        for original in self.document.glob("original/*.pdf"):
            with pymupdf.open(original) as document:
                self.assertEqual(len(document), 1)
                self.assertTrue(document[0].get_text().strip())
            raster = self.document / "chars_rem" / original.name
            self.assertEqual(pdf_image(original, 100).tobytes(), pdf_image(raster, 100).tobytes())
        for path in [*self.document.glob("chars_rem/*.pdf"), *self.document.glob("augmented/*.pdf")]:
            with pymupdf.open(path) as document:
                self.assertEqual(len(document), 1)
                self.assertEqual(document[0].get_text(), "")
                self.assertFalse(document[0].get_fonts())
                self.assertEqual(document[0].rect, pymupdf.Rect(0, 0, 300, 400))
        hint = bench.read_json(self.document / "original" / "page_0001.hint.json")
        self.assertFalse(hint["verified"])
        self.assertIn("駅", hint["native_text"])

    def test_resume_repairs_missing_files_and_rejects_changed_sources(self):
        self.prepare()
        original = self.document / "original" / "page_0001.pdf"
        timestamp = original.stat().st_mtime_ns
        self.assertFalse(self.prepare())
        missing = self.document / "augmented" / "page_0001_photo_medium.pdf"
        expected = pdf_image(missing, 100).tobytes()
        missing.unlink()
        self.assertTrue(self.prepare())
        self.assertEqual(original.stat().st_mtime_ns, timestamp)
        self.assertEqual(pdf_image(missing, 100).tobytes(), expected)
        pdf_hash = bench.digest(missing)
        missing.with_suffix(".json").unlink()
        self.assertTrue(self.prepare())
        self.assertEqual(bench.digest(missing), pdf_hash)
        self.assertTrue(missing.with_suffix(".json").is_file())
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
        path = self.document / "manifest.json"
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
        for profile in PROFILES:
            for severity in SEVERITIES:
                with self.subTest(profile=profile, severity=severity):
                    first, settings = augment(image, profile, 42, severity)
                    second, settings2 = augment(image, profile, 42, severity)
                    self.assertEqual(first.size, image.size)
                    self.assertEqual(first.tobytes(), second.tobytes())
                    self.assertEqual(settings, settings2)
                    self.assertNotEqual(first.tobytes(), image.tobytes())
                    self.assertTrue(settings["training_review_required"])
                    # Metadata must survive JSON round-tripping, including warp meshes.
                    self.assertEqual(json.dumps(settings), json.dumps(json.loads(json.dumps(settings))))
        for profile, severity in (("unknown", "hard"), ("scan", "unknown"), ("warp", None)):
            with self.assertRaises(ValueError):
                augment(image, profile, 42, severity)

    def test_severity_increases_effect_strength_without_cropping_geometry(self):
        image = Image.new("RGB", (300, 400), "white")
        measures = {"scan": ("rotation_degrees", abs), "motion": ("motion_blur_pixels", float),
                    "resolution": ("resolution_scale", lambda x: -x),
                    "compression": ("jpeg_quality", lambda x: -x),
                    "ink": ("ink_strength", float), "screen": ("moire_strength", float),
                    "photo": ("shadow_strength", float)}
        corners = np.array([[0, 0, 1], [300, 0, 1], [300, 400, 1], [0, 400, 1]])
        for profile in PROFILES:
            strengths = []
            for severity in SEVERITIES:
                _, settings = augment(image, profile, 42, severity)
                geometry = settings["geometry"]
                if geometry["kind"] == "homography":
                    output = corners @ np.array(geometry["source_to_output"]).T
                    output = output[:, :2] / output[:, 2, None]
                    self.assertTrue(np.all(output >= 0))
                    self.assertTrue(np.all(output <= [300, 400]))
                elif geometry["kind"] == "sine_mesh":
                    x = np.linspace(0, 300, 100)
                    mx, my = geometry["margin"]
                    sx, sy = geometry["scale"]
                    for y in (0, 400):
                        v = my + sy * y + geometry["amplitude_pixels"] * np.sin(np.pi * x / 300)
                        self.assertTrue(np.all((v >= 0) & (v <= 400)))
                    self.assertTrue(np.all((mx + sx * x >= 0) & (mx + sx * x <= 300)))
                    strengths.append(abs(geometry["amplitude_pixels"]))
                if profile in measures:
                    key, measure = measures[profile]
                    strengths.append(measure(settings[key]))
            self.assertEqual(strengths, sorted(set(strengths)))

    def test_prepare_appends_exact_pairs_and_migrates_legacy_without_replacing_inputs(self):
        self.args.augmentations = ["scan"]
        self.prepare()
        manifest_path = self.document / "manifest.json"
        manifest = bench.read_json(manifest_path)
        for path in (self.document / "augmented").iterdir():
            path.unlink()
        manifest["files"] = [name for name in manifest["files"] if not name.startswith("augmented/")]
        manifest["settings"] = {"dpi": 100, "seed": 42, "profiles": ["scan"], "augmentation_version": 1}
        for page in manifest["pages"]:
            stem = f"page_{page['page']:04d}"
            image = pdf_image(self.document / "original" / f"{stem}.pdf", 100)
            seed = page_seed(manifest["source_sha256"], page["page"], "scan", 42)
            legacy, parameters = augment(image, "scan", seed)
            target = self.document / "augmented" / f"{stem}_scan.pdf"
            bench.atomic_write(target, image_pdf(legacy, page["width"], page["height"]))
            bench.write_json(target.with_suffix(".json"), parameters)
            page["augmentations"] = {"scan": parameters}
            manifest["files"].extend(str(p.relative_to(self.document)) for p in (target, target.with_suffix(".json")))
        bench.write_json(manifest_path, manifest)
        candidate = self.candidate("augmented", "scan")
        hashes = {name: bench.digest(self.document / name) for name in manifest["files"]}
        markdown_hash = bench.digest(candidate)
        self.args.augmentations, self.args.severities = ["photo"], ["hard"]
        self.assertTrue(self.prepare())
        self.assertEqual({p.stem for p in (self.document / "augmented").glob("*.pdf")},
                         {f"page_{i:04d}_{v}" for i in (1, 2) for v in ("scan", "photo_hard")})
        for name, expected in hashes.items():
            self.assertEqual(bench.digest(self.document / name), expected)
        self.assertEqual(bench.digest(candidate), markdown_hash)
        self.assertFalse(self.prepare())
        missing = self.document / "augmented" / "page_0001_scan.pdf"
        expected_pixels = pdf_image(missing, 100).tobytes()
        missing.unlink()
        self.assertTrue(self.prepare())
        self.assertEqual(pdf_image(missing, 100).tobytes(), expected_pixels)
        self.args.augmentations, self.args.severities = ["ink"], ["mild"]
        self.assertTrue(self.prepare())
        self.assertEqual(set(bench.read_json(manifest_path)["settings"]["variants"]),
                         {"scan", "photo_hard", "ink_mild"})

    def test_severity_output_resolves_to_clean_reference(self):
        self.prepare()
        candidate = self.candidate("augmented", "scan_medium")
        sources = bench.score_sources(self.root, candidate)
        self.assertEqual(sources["profile"], "scan_medium")
        self.assertEqual(sources["reference"], self.document / "original" / "page_0001.pdf")
        self.assertEqual(sources["input"], self.document / "augmented" / "page_0001_scan_medium.pdf")
        self.assertTrue(sources["input"].is_file())

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

    def test_existing_document_identity_survives_new_same_stem_format(self):
        self.prepare()
        (self.root / "raw" / "日本語.docx").touch()
        names = {source.suffix: name for source, name in bench.source_documents(self.root)}
        self.assertEqual(names[".pdf"], "日本語")
        self.assertEqual(names[".docx"], "日本語__docx")

    def test_office_endpoint_multipart_and_invalid_pdf_fallback(self):
        source = self.root / "raw" / "日本語.docx"
        source.write_bytes(b"fake-office-file")
        target = self.root / "converted.pdf"
        config = {"CONVERTER_URL": "http://converter/convert", "CONVERTER_API_KEY": "test-token"}
        with patch.object(bench, "request_bytes", return_value=self.source.read_bytes()) as request:
            self.assertEqual(bench.convert_office(source, target, config), "endpoint")
        self.assertIn(b'name="file"', request.call_args.args[1])
        self.assertIn(b"fake-office-file", request.call_args.args[1])
        self.assertEqual(request.call_args.args[2]["Authorization"], "Bearer test-token")

        def fake_office(command, log, timeout):
            folder = Path(command[command.index("--outdir") + 1])
            (folder / (source.stem + ".pdf")).write_bytes(self.source.read_bytes())

        with patch.object(bench, "request_bytes", return_value=b"not a PDF"), \
                patch.object(bench, "run_process", side_effect=fake_office):
            self.assertEqual(bench.convert_office(source, target, config), "libreoffice")
        bench.valid_pdf(target.read_bytes())

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
        source = self.document / "original" / "page_0001.pdf"
        output = self.root / "output" / "日本語" / "mineru" / "original"
        self.assertTrue(bench.parse_page(source, output, setup, 1, config))
        self.assertFalse(bench.parse_page(source, output, setup, 1, config))
        self.assertTrue(bench.parse_page(source, output, setup, 2, config))
        marker = bench.read_json(output / "page_0001_attempt1.run.json")
        self.assertIn("versions", marker)
        self.assertTrue((output / marker["artifact_markdown"]).is_file())
        self.assertEqual((output / "page_0001_attempt1.md").read_text(), "# 駅\n")
        env = bench.isolated_environment(setup[0])
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["VIRTUAL_ENV"], str(setup[0]))

    def test_failed_pipeline_does_not_publish_markdown(self):
        self.prepare()
        venv = Path(sys.executable).parent.parent
        setup = (venv, ["{python}", "-c", "raise SystemExit(1)"], {})
        source = self.document / "original" / "page_0001.pdf"
        output = self.root / "output" / "日本語" / "mineru" / "original"
        with self.assertRaises(RuntimeError):
            bench.parse_page(source, output, setup, 1, {})
        self.assertFalse((output / "page_0001_attempt1.md").exists())
        self.assertEqual(len(list((output / ".artifacts").glob("*/run.log"))), 1)

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

    def test_score_always_uses_original_resume_redo_and_content_invalidation(self):
        self.prepare()
        markdown = self.candidate("augmented", "photo_medium")
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
                expected = pdf_image(self.document / "original" / "page_0001.pdf", 100)
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
        bench.write_json(self.root / "scores.json", {"records": records})
        bench.rank_pages(self.root, self.args, self.config)
        pages = bench.read_json(self.root / "difficult_pages.json")["pages"]
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["mean_mistakes"], 15)
        bench.atomic_write(second, "changed")
        bench.rank_pages(self.root, self.args, self.config)
        ranked = bench.read_json(self.root / "difficult_pages.json")
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
