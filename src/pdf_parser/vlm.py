"""OpenAI-compatible (vLLM) calls: VL reading, constrained fill-in, and the faithfulness judge."""

import asyncio
import base64
import json

import httpx

# PaddleOCR-VL task prompts and pixel budgets (paddlex PaddleOCR-VL pipeline defaults).
OCR, TABLE, FORMULA = "OCR:", "Table Recognition:", "Formula Recognition:"
PIXELS = (112896, 1003520)
OTSL_TAGS = ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>")

FILL_PROMPT = """This image is part of a Japanese business document. Below is the OCR text of it.
Inside each ⟦…|…⟧ the OCR was unsure: choose the one alternative that matches the image and makes \
sense in context (an empty alternative means no character). Copy everything else exactly as given.
{context}OCR text:
{template}"""

TABLE_PROMPT = """This image is a table from a Japanese business document. Below is its HTML as read by OCR.
Inside each ⟦…|…⟧ the OCR was unsure: choose the one alternative that matches the image and makes \
sense in context (an empty alternative means no character). Copy all other text exactly. You may fix \
the table structure (rows, rowspan, colspan, empty cells) to match the image, but keep every cell's \
text in its own cell and in the same order.
OCR HTML:
{template}"""

JUDGE_PROMPT = """The first image is a page from a Japanese document. The second image is a rendering of \
Markdown extracted from it. Is the extraction faithful: every character of the body text and tables \
present and identical (no wrong, missing, extra or reordered text; tables with the same cells)? Ignore \
fonts, styling, line wrapping, images, page headers, footers and page numbers."""

JUDGE_SCHEMA = {"type": "object", "properties": {
    "faithful": {"type": "boolean"},
    "problems": {"type": "array", "items": {"type": "string"}, "maxItems": 10}},
    "required": ["faithful", "problems"]}


def image_part(png):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


class Chat:
    """Minimal async client for /chat/completions with an in-flight limit."""

    def __init__(self, url, model, api_key=None, concurrency=32, timeout=600):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.AsyncClient(base_url=url.rstrip("/") + "/", headers=headers, timeout=timeout,
                                      limits=httpx.Limits(max_connections=concurrency))
        self.model = model
        self.limit = asyncio.Semaphore(concurrency)

    async def complete(self, content, **params):
        body = {"model": self.model, "temperature": 0, **params,
                "messages": [{"role": "user", "content": content}]}
        for attempt in range(3):
            try:
                async with self.limit:
                    response = await self.http.post("chat/completions", json=body)
                if response.status_code < 500:
                    break
            except httpx.TransportError:
                if attempt == 2:
                    raise
            await asyncio.sleep(2 ** attempt)
        if response.status_code >= 400:
            raise RuntimeError(f"{self.model}: HTTP {response.status_code}: {response.text[:500]}")
        return response.json()["choices"][0]

    async def close(self):
        await self.http.aclose()


async def read(chat, png, prompt, constraint=None, top_logprobs=0, max_tokens=4096):
    """One PaddleOCR-VL task on a crop; returns (text, logprob tokens, finished)."""
    params = {"max_completion_tokens": max_tokens, "skip_special_tokens": True,
              "mm_processor_kwargs": {"min_pixels": PIXELS[0], "max_pixels": PIXELS[1]}}
    if constraint:
        params["structured_outputs"] = {"regex": constraint}
    if top_logprobs:
        params.update(logprobs=True, top_logprobs=top_logprobs)
    choice = await chat.complete([image_part(png), {"type": "text", "text": prompt}], **params)
    tokens = (choice.get("logprobs") or {}).get("content") or []
    return choice["message"]["content"] or "", tokens, choice.get("finish_reason") != "length"


async def fill(chat, png, shown, pattern, prompt=FILL_PROMPT, context=""):
    """Resolve the ⟦…⟧ slots in `shown`; the decoder is held to `pattern`, so nothing else can appear."""
    context = f"Surrounding text, for context only: {context}\n" if context else ""
    choice = await chat.complete(
        [image_part(png), {"type": "text", "text": prompt.format(context=context, template=shown)}],
        max_completion_tokens=4 * len(shown) + 64, structured_outputs={"regex": pattern})
    return choice["message"]["content"] or ""


async def judge(chat, original_png, rendered_png):
    choice = await chat.complete(
        [image_part(original_png), image_part(rendered_png), {"type": "text", "text": JUDGE_PROMPT}],
        max_completion_tokens=1024, structured_outputs={"json": JUDGE_SCHEMA})
    return json.loads(choice["message"]["content"])
