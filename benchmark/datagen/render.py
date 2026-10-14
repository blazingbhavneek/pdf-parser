"""Render Markdown and HTML tables onto one page without clipping content."""

import io
import mimetypes
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import pymupdf
from markdown_it import MarkdownIt
from PIL import Image

VERSION = 1
CSS = """
* { box-sizing: border-box; }
body { margin: 0; background: white; color: #111; }
main { padding: 24px; font: 16px/1.6 'Noto Sans CJK JP', 'Noto Sans JP', sans-serif;
       overflow-wrap: anywhere; display: flow-root; }
table { border-collapse: collapse; margin: 12px 0; }
td, th { border: 1px solid #555; padding: 5px 8px; vertical-align: top; }
img, svg { max-width: 100%; height: auto; }
pre { white-space: pre-wrap; }
blockquote { border-left: 3px solid #bbb; margin-left: 0; padding-left: 12px; }
"""


def pdf_image(pdf: Path, dpi: int) -> Image.Image:
    with pymupdf.open(pdf) as document:
        if len(document) != 1:
            raise ValueError(f"Expected a single-page PDF: {pdf}")
        pixmap = document[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def image_pdf(image: Image.Image, width: float, height: float) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    with pymupdf.open() as document:
        page = document.new_page(width=width, height=height)
        page.insert_image(page.rect, stream=stream.getvalue(), keep_proportion=True)
        return document.tobytes(garbage=4, deflate=True)


class MarkdownRenderer:
    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self.playwright = sync_playwright().start()
        try:
            self.browser = self.playwright.chromium.launch(channel="chromium", headless=True)
        except Exception:
            self.playwright.stop()
            raise
        return self

    def __exit__(self, *args):
        self.browser.close()
        self.playwright.stop()

    def render(self, markdown: str, asset_base: Path, asset_root: Path,
               width: float, height: float, dpi: int, destination: Path) -> dict:
        pixels = (round(width * dpi / 72), round(height * dpi / 72))
        viewport = {"width": round(width * 96 / 72), "height": round(height * 96 / 72)}
        page = self.browser.new_page(viewport=viewport, device_scale_factor=dpi / 96,
                                     java_script_enabled=False, service_workers="block")
        missing = []

        def resource(route):
            parsed = urlparse(route.request.url)
            if parsed.scheme == "data":
                route.continue_()
                return
            if parsed.netloc != "benchmark.invalid":
                missing.append(route.request.url)
                route.abort()
                return
            relative = unquote(parsed.path).lstrip("/")
            root = asset_root.resolve()
            # Relative paths resolve inside the run's artifact root; parsers such as
            # Docling's `referenced` export write absolute paths into that same root.
            candidate = next((path for path in ((asset_root / relative).resolve(),
                                                Path("/" + relative).resolve())
                              if path.is_relative_to(root) and path.is_file()), None)
            if candidate is None:
                missing.append(relative)
                route.abort()
                return
            mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            if not mime.startswith("image/"):
                missing.append(relative)
                route.abort()
                return
            route.fulfill(body=candidate.read_bytes(), content_type=mime)

        page.route("**/*", resource)
        body = MarkdownIt("commonmark", {"html": True}).enable("table").render(markdown)
        base = quote(asset_base.resolve().relative_to(asset_root.resolve()).as_posix())
        base = "" if base == "." else base + "/"
        # The CSP allows table HTML and local images, but no active content or remote fetches.
        html = ("<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
                "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; "
                "style-src 'unsafe-inline'; img-src data: https://benchmark.invalid; "
                "font-src 'none'; base-uri https://benchmark.invalid\">"
                "<base href='https://benchmark.invalid/" + base + "'><style>" + CSS + "</style></head>"
                "<body><main>" + body + "</main></body></html>")
        try:
            page.set_content(html, wait_until="load")
            # Playwright evaluation works with document scripts disabled.
            page.evaluate("document.fonts.ready")
            missing.extend(page.locator("img").evaluate_all(
                "imgs => imgs.filter(i => !i.complete || i.naturalWidth === 0).map(i => i.src)"))
            if missing:
                raise ValueError(f"Unrenderable Markdown images: {missing[:5]}")
            dimensions = page.locator("main").evaluate(
                "e => ({width: Math.max(e.scrollWidth, e.offsetWidth), "
                "height: Math.max(e.scrollHeight, e.offsetHeight)})")
            # shortcut: very long outputs are rejected, use tiled judging above 40,000px.
            if dimensions["height"] > 40000 or dimensions["width"] > 12000:
                raise ValueError("Markdown render is too large for one page; requires tiled judging")
            raw = page.screenshot(full_page=True, type="png")
        finally:
            page.close()
        with Image.open(io.BytesIO(raw)) as screenshot:
            image = screenshot.convert("RGB")
        scale = min(pixels[0] / image.width, pixels[1] / image.height)
        fitted = image.resize((max(1, round(image.width * scale)),
                               max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", pixels, "white")
        canvas.paste(fitted, ((pixels[0] - fitted.width) // 2, 0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination.with_suffix(".png"))
        destination.with_suffix(".pdf").write_bytes(image_pdf(canvas, width, height))
        return {"content_pixels": list(image.size), "page_pixels": list(pixels), "fit_scale": scale}
